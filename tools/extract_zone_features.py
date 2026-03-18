import argparse
import gc
import os
import random
import sys
import time

import numpy as np
import torch
from tqdm import tqdm

root_path = os.path.abspath(__file__)
root_path = "/".join(root_path.split("/")[:-2])
workspace_root = os.path.dirname(root_path)
sys.path.append(root_path)


def resolve_repo_path(*parts):
    local_path = os.path.join(root_path, *parts)
    workspace_path = os.path.join(workspace_root, *parts)
    if os.path.exists(local_path):
        return local_path
    return workspace_path


data_root = resolve_repo_path("data")

from datasets.dataset_nuplan import NuPlanTest
from models.model import TrainTransformers
from modules.tokenizers.model_tokenizer import Tokenizer
from modules.tokenizers.pose_tokenizer import poses_to_indices, yaws_to_indices
from utils.config_utils import Config
from utils.running import load_parameters


CONFIG_PATH = resolve_repo_path("configs", "drivingworld_v1", "gen_videovq_conf_demo.py")
LOAD_PATH = resolve_repo_path("pretrained_models", "world_model.pth")
VQ_CKPT = resolve_repo_path("pretrained_models", "vqvae.pt")

PHYSICS_SAMPLES_PATH = os.path.join(data_root, "physics_samples.pt")
COHERENCE_SAMPLES_PATH = os.path.join(data_root, "coherence_samples.pt")

PHYSICS_FEATURES_SAVE_PATH = os.path.join(data_root, "physics_features.pt")
COHERENCE_FEATURES_SAVE_PATH = os.path.join(data_root, "coherence_features.pt")

PHYSICS_LABEL_KEYS = [
    "past_avg_speed",
    "past_avg_yaw_rate",
    "past_heading_change",
    "future_speed",
    "future_yaw_rate",
    "future_delta_yaw",
]

META_KEYS = [
    "seq_id",
    "window_start",
    "window_end",
    "future_t",
    "negative_type",
    "source_index",
    "cond_source_index",
    "base_index",
    "cond_index",
    "cond_seq_id",
    "cond_window_start",
    "cond_window_end",
    "cond_future_t",
]


def parse_cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, choices=["physics", "coherence"], required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--pool_mode",
        type=str,
        choices=["all_mean", "tokenwise"],
        default="all_mean",
        help="`all_mean` keeps the legacy pooled 1536-d feature; `tokenwise` keeps all last-t tokens as [L, C].",
    )
    parser.add_argument("--data_root", type=str, default=data_root)
    parser.add_argument("--json_root", type=str, default=data_root)
    parser.add_argument("--condition_frames", type=int, default=None)
    parser.add_argument("--downsample_fps", type=int, default=5)
    parser.add_argument("--downsample_size", type=int, default=None)
    parser.add_argument("--max_cached_sequences", type=int, default=2)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--feature_dtype", type=str, choices=["float32", "float16"], default="float32")
    return parser.parse_args()


def build_args():
    args = Config.fromfile(CONFIG_PATH)
    args.load_path = LOAD_PATH
    args.vq_ckpt = VQ_CKPT
    args.exp_name = "extract_zone_features"
    if not hasattr(args, "seed"):
        args.seed = 1234
    return args


def init_environment(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def maybe_print_cuda_memory(device, prefix):
    if device.type != "cuda":
        return
    allocated = torch.cuda.memory_allocated(device) / 1024 ** 3
    reserved = torch.cuda.memory_reserved(device) / 1024 ** 3
    print(f"{prefix} cuda_allocated={allocated:.2f}GB cuda_reserved={reserved:.2f}GB")


def get_runtime_hparams(config_args, cli_args):
    condition_frames = cli_args.condition_frames
    if condition_frames is None:
        condition_frames = getattr(config_args, "condition_frames", 15)

    downsample_size = cli_args.downsample_size
    if downsample_size is None:
        downsample_size = getattr(config_args, "downsample_size", 16)

    image_size = getattr(config_args, "image_size", [256, 512])
    if len(image_size) != 2:
        raise ValueError(f"Unexpected image_size in config: {image_size}")

    return {
        "condition_frames": int(condition_frames),
        "downsample_fps": int(cli_args.downsample_fps),
        "downsample_size": int(downsample_size),
        "image_h": int(image_size[0]),
        "image_w": int(image_size[1]),
    }


def load_probe_samples(task, start_index=0, end_index=None, max_samples=None):
    if task == "physics":
        path = PHYSICS_SAMPLES_PATH
    elif task == "coherence":
        path = COHERENCE_SAMPLES_PATH
    else:
        raise ValueError(f"Unknown task: {task}")

    print(f"loading probe samples from: {path}")
    payload = torch.load(path, map_location="cpu")

    if isinstance(payload, dict) and "samples" in payload:
        samples = payload["samples"]
        sample_format = payload.get("sample_format", "unknown")
        print("probe payload keys =", sorted(payload.keys()))
    else:
        samples = payload
        sample_format = "legacy_tensor_list"
        print("probe payload type = legacy list")

    total_before_slice = len(samples)
    start_index = int(start_index)
    if end_index is None:
        end_index = total_before_slice
    end_index = int(min(end_index, total_before_slice))
    if start_index < 0 or start_index > end_index:
        raise ValueError(f"Invalid slice range: start_index={start_index}, end_index={end_index}")
    samples = samples[start_index:end_index]
    print(f"slice range = [{start_index}, {end_index}) -> {len(samples)} samples")

    if max_samples is not None:
        samples = samples[:max_samples]
        end_index = start_index + len(samples)
        print(f"max_samples active -> truncated to {len(samples)}")

    print(f"probe sample_format = {sample_format}")
    print(f"num samples = {len(samples)}")
    if len(samples) > 0:
        print("first sample keys =", sorted(samples[0].keys()))
    return samples, sample_format, start_index, end_index


class SequenceProvider:
    def __init__(
        self,
        data_root,
        json_root,
        condition_frames,
        downsample_fps,
        downsample_size,
        h,
        w,
        max_cached_sequences=4,
    ):
        print("initializing SequenceProvider ...")
        self.dataset = NuPlanTest(
            data_root=data_root,
            json_root=json_root,
            condition_frames=condition_frames,
            downsample_fps=downsample_fps,
            downsample_size=downsample_size,
            h=h,
            w=w,
        )
        self.max_cached_sequences = max_cached_sequences
        self.cache = {}
        self.cache_order = []
        print(
            "SequenceProvider ready:",
            f"dataset_len={len(self.dataset)}",
            f"max_cached_sequences={self.max_cached_sequences}",
        )

    def _touch(self, seq_id):
        if seq_id in self.cache_order:
            self.cache_order.remove(seq_id)
        self.cache_order.append(seq_id)

    def get_sequence(self, seq_id):
        seq_id = int(seq_id)
        if seq_id not in self.cache:
            self.cache[seq_id] = self.dataset[seq_id]
            self._touch(seq_id)
            while len(self.cache_order) > self.max_cached_sequences:
                old_seq_id = self.cache_order.pop(0)
                self.cache.pop(old_seq_id, None)
        else:
            self._touch(seq_id)
        return self.cache[seq_id]

    def _slice_window(self, seq_id, start, end, future_t):
        imgs, poses, yaws = self.get_sequence(seq_id)
        start = int(start)
        end = int(end)
        future_t = int(future_t)
        past_imgs = imgs[start:end + 1].clone()
        past_poses = poses[start:end + 1].clone()
        past_yaws = yaws[start:end + 1].clone()
        future_img = imgs[future_t].clone()
        future_pose = poses[future_t].clone()
        future_yaw = yaws[future_t].clone()
        return past_imgs, past_poses, past_yaws, future_img, future_pose, future_yaw

    def build_physics_sequence(self, sample):
        return self._slice_window(
            sample["seq_id"],
            sample["window_start"],
            sample["window_end"],
            sample["future_t"],
        )

    def build_coherence_sequence(self, sample):
        past_imgs, past_poses, past_yaws, future_img, _, _ = self._slice_window(
            sample["seq_id"],
            sample["window_start"],
            sample["window_end"],
            sample["future_t"],
        )
        _, _, _, _, future_pose, future_yaw = self._slice_window(
            sample["cond_seq_id"],
            sample["cond_window_start"],
            sample["cond_window_end"],
            sample["cond_future_t"],
        )
        return past_imgs, past_poses, past_yaws, future_img, future_pose, future_yaw


def build_full_sequence(sample, task, sequence_provider):
    if "past_imgs" in sample:
        past_imgs = sample["past_imgs"]
        past_poses = sample["past_poses"]
        past_yaws = sample["past_yaws"]
        future_img = sample["future_img"].unsqueeze(0)

        if task == "physics":
            future_pose = sample["future_pose"].unsqueeze(0)
            future_yaw = sample["future_yaw"].unsqueeze(0)
        elif task == "coherence":
            future_pose = sample["future_pose_cond"].unsqueeze(0)
            future_yaw = sample["future_yaw_cond"].unsqueeze(0)
        else:
            raise ValueError(f"Unknown task: {task}")
    else:
        if sequence_provider is None:
            raise RuntimeError("Lightweight samples require a sequence provider.")

        if task == "physics":
            past_imgs, past_poses, past_yaws, future_img, future_pose, future_yaw = (
                sequence_provider.build_physics_sequence(sample)
            )
        elif task == "coherence":
            past_imgs, past_poses, past_yaws, future_img, future_pose, future_yaw = (
                sequence_provider.build_coherence_sequence(sample)
            )
        else:
            raise ValueError(f"Unknown task: {task}")

        future_img = future_img.unsqueeze(0)
        future_pose = future_pose.unsqueeze(0)
        future_yaw = future_yaw.unsqueeze(0)

    img_seq = torch.cat([past_imgs, future_img], dim=0)
    pose_seq = torch.cat([past_poses, future_pose], dim=0)
    yaw_seq = torch.cat([past_yaws, future_yaw], dim=0)
    return img_seq, pose_seq, yaw_seq


def pool_hidden_state(hidden_item, effective_frames, mode="all_mean"):
    name = hidden_item["name"]
    tensor = hidden_item["tensor"]

    if tensor.ndim == 4:
        last_t = tensor[:, -1, :, :]
        if mode == "all_mean":
            feat = last_t.mean(dim=1).squeeze(0)
        elif mode == "tokenwise":
            feat = last_t.squeeze(0)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        return feat.detach().cpu()

    if tensor.ndim == 3:
        assert tensor.shape[0] == effective_frames, (
            f"{name}: first dim {tensor.shape[0]} != effective_frames {effective_frames}"
        )
        last_t = tensor[-1, :, :]
        if mode == "all_mean":
            feat = last_t.mean(dim=0)
        elif mode == "tokenwise":
            feat = last_t
        else:
            raise ValueError(f"Unknown mode: {mode}")
        return feat.detach().cpu()

    raise ValueError(f"Unexpected hidden tensor ndim for {name}: {tensor.ndim}")


def cast_feature_dtype(x, feature_dtype):
    if feature_dtype == "float16":
        return x.to(torch.float16)
    if feature_dtype == "float32":
        return x.to(torch.float32)
    raise ValueError(f"Unknown feature_dtype: {feature_dtype}")



def build_raw_pose_yaw_baselines(pose_seq, yaw_seq):
    # Keep a lightweight continuous-motion baseline outside the world model.
    step_motion = torch.cat([pose_seq, yaw_seq], dim=1)
    return {
        "raw_pose_yaw_past": step_motion[:-1].reshape(-1).detach().cpu(),
        "raw_pose_yaw_full": step_motion.reshape(-1).detach().cpu(),
    }


def extract_features_streaming(
    tokenizer,
    model_wrapper,
    samples,
    task,
    device,
    effective_frames,
    pool_mode,
    sequence_provider,
    log_every,
    feature_dtype,
):
    print("\n===== streaming feature extraction =====")
    print(
        "settings:",
        f"task={task}",
        f"pool_mode={pool_mode}",
        f"feature_dtype={feature_dtype}",
        f"effective_frames={effective_frames}",
        f"log_every={log_every}",
    )

    layer_features = {}
    labels = {}
    meta = {}

    total_samples = len(samples)
    start_time = time.time()
    last_log_time = start_time

    for i, sample in enumerate(tqdm(samples)):
        img_seq, pose_seq, yaw_seq = build_full_sequence(sample, task, sequence_provider)
        raw_baselines = build_raw_pose_yaw_baselines(pose_seq, yaw_seq)

        img_seq = img_seq.unsqueeze(0)
        pose_seq = pose_seq.unsqueeze(0).to(device)
        yaw_seq = yaw_seq.unsqueeze(0).to(device)

        with torch.no_grad():
            start_token_seq, _ = tokenizer.encode_to_z(img_seq)
            feature_total = tokenizer.vq_model.quantize.embedding(start_token_seq).to(device, non_blocking=True)

        drop_flag = torch.isinf(pose_seq[:, 0, 0])
        pose_indices_total = poses_to_indices(
            pose_seq,
            model_wrapper.pose_x_vocab_size,
            model_wrapper.pose_y_vocab_size,
        )
        yaw_indices_total = yaws_to_indices(
            yaw_seq,
            model_wrapper.yaw_vocab_size,
        )

        with torch.no_grad():
            out = model_wrapper.extract_hidden_states(
                feature_total=feature_total,
                pose_indices_total=pose_indices_total,
                yaw_indices_total=yaw_indices_total,
                drop_flag=drop_flag,
            )

        hidden_states = [
            {
                "name": "tokenizer_last",
                "stage": "tokenizer",
                "layer_idx": 0,
                "tensor": feature_total[:, :-1, ...].contiguous(),
            }
        ] + out["hidden_states"]

        if i == 0:
            print("\nFirst sample info:")
            print("feature_total shape =", tuple(feature_total.shape))
            print("feature_total dtype =", feature_total.dtype)
            print("pose_indices_total shape =", tuple(pose_indices_total.shape))
            print("yaw_indices_total shape =", tuple(yaw_indices_total.shape))
            print("drop_flag shape =", tuple(drop_flag.shape))
            print("img_seq shape =", tuple(img_seq.shape))
            print("pose_seq shape =", tuple(pose_seq.shape))
            print("yaw_seq shape =", tuple(yaw_seq.shape))
            print("num hidden states =", len(hidden_states))
            for k in PHYSICS_LABEL_KEYS:
                if k in sample:
                    print(f"{k} =", sample[k])
            if "coherence_label" in sample:
                print("coherence_label =", sample["coherence_label"])
            for k in META_KEYS:
                if k in sample:
                    print(f"{k} =", sample[k])
            print("raw baseline keys =", sorted(raw_baselines.keys()))

            print("\nFirst sample hidden info:")
            for h in hidden_states:
                print(h["name"], tuple(h["tensor"].shape), h["tensor"].dtype)

            print("\npreallocating output tensors ...")
            for layer_name, feat in raw_baselines.items():
                feat = cast_feature_dtype(feat, feature_dtype)
                layer_features[layer_name] = torch.empty(
                    (total_samples,) + tuple(feat.shape),
                    dtype=feat.dtype,
                )
                layer_features[layer_name][0] = feat
                print(
                    f"allocated {layer_name}: shape={tuple(layer_features[layer_name].shape)} "
                    f"dtype={layer_features[layer_name].dtype}"
                )

            for h in hidden_states:
                layer_name = h["name"]
                feat = pool_hidden_state(h, effective_frames=effective_frames, mode=pool_mode)
                feat = cast_feature_dtype(feat, feature_dtype)
                layer_features[layer_name] = torch.empty(
                    (total_samples,) + tuple(feat.shape),
                    dtype=feat.dtype,
                )
                layer_features[layer_name][0] = feat
                print(
                    f"allocated {layer_name}: shape={tuple(layer_features[layer_name].shape)} "
                    f"dtype={layer_features[layer_name].dtype}"
                )

            for key in PHYSICS_LABEL_KEYS:
                if key in sample:
                    labels[key] = torch.empty(total_samples, dtype=torch.float32)
                    labels[key][0] = float(sample[key])

            if "coherence_label" in sample:
                labels["coherence_label"] = torch.empty(total_samples, dtype=torch.long)
                labels["coherence_label"][0] = int(sample["coherence_label"])

            for k in META_KEYS:
                if k in sample:
                    if isinstance(sample[k], (int, float, np.integer, np.floating)):
                        meta[k] = torch.empty(
                            total_samples,
                            dtype=torch.long if isinstance(sample[k], (int, np.integer)) else torch.float32,
                        )
                        meta[k][0] = sample[k]
                    else:
                        meta[k] = [None] * total_samples
                        meta[k][0] = sample[k]

        if i != 0:
            for h in hidden_states:
                layer_name = h["name"]
                feat = pool_hidden_state(h, effective_frames=effective_frames, mode=pool_mode)
                feat = cast_feature_dtype(feat, feature_dtype)
                layer_features[layer_name][i] = feat

            for key in PHYSICS_LABEL_KEYS:
                if key in sample:
                    labels[key][i] = float(sample[key])

            if "coherence_label" in sample:
                labels["coherence_label"][i] = int(sample["coherence_label"])

            for k in META_KEYS:
                if k in sample:
                    if isinstance(meta[k], torch.Tensor):
                        meta[k][i] = sample[k]
                    else:
                        meta[k][i] = sample[k]

        should_log = (
            i == 0
            or (i + 1) % log_every == 0
            or (i + 1) == total_samples
        )
        if should_log:
            now = time.time()
            elapsed = now - start_time
            step_elapsed = now - last_log_time
            avg_per_sample = elapsed / (i + 1)
            remaining = (total_samples - (i + 1)) * avg_per_sample
            print(
                f"[progress] processed={i + 1}/{total_samples} "
                f"elapsed={elapsed / 60:.2f}min "
                f"since_last_log={step_elapsed:.2f}s "
                f"avg_per_sample={avg_per_sample:.4f}s "
                f"eta={remaining / 60:.2f}min"
            )
            if len(layer_features) > 0:
                first_layer_name = sorted(layer_features.keys())[0]
                print(
                    f"[progress] first_layer={first_layer_name} "
                    f"stored_vectors={i + 1}"
                )
            maybe_print_cuda_memory(device, prefix="[progress]")
            last_log_time = now
            if device.type == "cuda":
                torch.cuda.empty_cache()

        del out
        del hidden_states
        del img_seq
        del pose_seq
        del yaw_seq
        del feature_total
        del pose_indices_total
        del yaw_indices_total
        del drop_flag
        gc.collect()

    print("\nfinal output tensor summary ...")
    for layer_name, feats in layer_features.items():
        print(f"stacked {layer_name}: shape={tuple(feats.shape)} dtype={feats.dtype}")

    print("\nlabels/meta summary ...")
    for k, v in labels.items():
        print(f"label {k}: shape={tuple(v.shape)} dtype={v.dtype}")

    for k, v in meta.items():
        if isinstance(v, torch.Tensor):
            print(f"meta {k}: shape={tuple(v.shape)} dtype={v.dtype}")
        else:
            print(f"meta {k}: list(len={len(v)})")

    print("===== streaming extraction done =====\n")
    return layer_features, labels, meta


def save_feature_file(task, layer_features, labels, meta, pool_mode, sample_format, feature_dtype, save_path, start_index, end_index):
    out = {
        "task": task,
        "feature_mode": f"{pool_mode}_last_t",
        "feature_dtype": feature_dtype,
        "probe_sample_format": sample_format,
        "slice_range": {
            "start_index": int(start_index),
            "end_index": int(end_index),
        },
        "stage_semantics": {
            "tokenizer_last": "Tokenizer/VQ feature embedding before world-model multimodal projection. Uses the last observed condition-frame image tokens.",
            "next_state_hidden": "State after injecting next-state query tokens and image prefix, before the first autoregressive block.",
            "time_space": "Condition-only representation. Built from frames 0..T-1 before injecting next-state query tokens.",
            "ar": "Teacher-forced next-state representation. Built after injecting next yaw/pose query and next-frame image-token prefix.",
        },
        "layer_features": layer_features,
        "labels": labels,
        "meta": meta,
    }

    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    torch.save(out, save_path)
    print(f"Saved feature file to: {save_path}")

    print("\n===== saved feature summary =====")
    print("feature_mode =", out["feature_mode"])
    print("feature_dtype =", out["feature_dtype"])
    print("probe_sample_format =", out["probe_sample_format"])
    for layer_name, feats in layer_features.items():
        print(layer_name, tuple(feats.shape), feats.dtype)
    for label_name, label_tensor in labels.items():
        print(label_name, tuple(label_tensor.shape), label_tensor.dtype)
    for meta_name, meta_value in meta.items():
        if isinstance(meta_value, torch.Tensor):
            print(meta_name, tuple(meta_value.shape), meta_value.dtype)
        else:
            print(meta_name, f"list(len={len(meta_value)})")
    print("=================================\n")


def main():
    cli_args = parse_cli_args()
    args = build_args()
    init_environment(args.seed)
    runtime_hparams = get_runtime_hparams(args, cli_args)

    device = torch.device(cli_args.device)

    print("===== extract_zone_features config =====")
    print("task =", cli_args.task)
    print("device =", device)
    print("pool_mode =", cli_args.pool_mode)
    print("data_root =", cli_args.data_root)
    print("json_root =", cli_args.json_root)
    print("log_every =", cli_args.log_every)
    print("start_index =", cli_args.start_index)
    print("end_index =", cli_args.end_index)
    print("max_samples =", cli_args.max_samples)
    print("output_path =", cli_args.output_path)
    print("feature_dtype =", cli_args.feature_dtype)
    print("max_cached_sequences =", cli_args.max_cached_sequences)
    print("CONFIG_PATH =", CONFIG_PATH)
    print("LOAD_PATH =", LOAD_PATH)
    print("VQ_CKPT =", VQ_CKPT)
    print("condition_frames =", runtime_hparams["condition_frames"])
    print("downsample_fps =", runtime_hparams["downsample_fps"])
    print("image_size =", [runtime_hparams["image_h"], runtime_hparams["image_w"]])
    print("downsample_size =", runtime_hparams["downsample_size"])
    print("========================================")

    print("loading world model ...")
    model = TrainTransformers(args, local_rank=0, condition_frames=runtime_hparams["condition_frames"])
    checkpoint = torch.load(args.load_path, map_location="cpu")
    model.model = load_parameters(model.model, checkpoint)
    del checkpoint
    model = model.to(device)
    model.eval()
    maybe_print_cuda_memory(device, prefix="[after world model load]")

    samples, sample_format, start_index, end_index = load_probe_samples(
        cli_args.task,
        start_index=cli_args.start_index,
        end_index=cli_args.end_index,
        max_samples=cli_args.max_samples,
    )

    sequence_provider = None
    if sample_format != "legacy_tensor_list":
        sequence_provider = SequenceProvider(
            data_root=cli_args.data_root,
            json_root=cli_args.json_root,
            condition_frames=runtime_hparams["condition_frames"],
            downsample_fps=runtime_hparams["downsample_fps"],
            downsample_size=runtime_hparams["downsample_size"],
            h=runtime_hparams["image_h"],
            w=runtime_hparams["image_w"],
            max_cached_sequences=cli_args.max_cached_sequences,
        )
    else:
        print("legacy_tensor_list detected: sample already carries image/pose/yaw tensors.")

    print("loading tokenizer ...")
    tokenizer = Tokenizer(args, local_rank="cpu")
    maybe_print_cuda_memory(device, prefix="[after tokenizer load]")

    effective_frames = runtime_hparams["condition_frames"]
    layer_features, labels, meta = extract_features_streaming(
        tokenizer=tokenizer,
        model_wrapper=model,
        samples=samples,
        task=cli_args.task,
        device=device,
        effective_frames=effective_frames,
        pool_mode=cli_args.pool_mode,
        sequence_provider=sequence_provider,
        log_every=cli_args.log_every,
        feature_dtype=cli_args.feature_dtype,
    )

    if cli_args.output_path is None:
        if cli_args.task == "physics":
            save_path = PHYSICS_FEATURES_SAVE_PATH
        elif cli_args.task == "coherence":
            save_path = COHERENCE_FEATURES_SAVE_PATH
        else:
            raise ValueError(f"Unknown task: {cli_args.task}")
    else:
        save_path = cli_args.output_path

    save_feature_file(
        task=cli_args.task,
        layer_features=layer_features,
        labels=labels,
        meta=meta,
        pool_mode=cli_args.pool_mode,
        sample_format=sample_format,
        feature_dtype=cli_args.feature_dtype,
        save_path=save_path,
        start_index=start_index,
        end_index=end_index,
    )


if __name__ == "__main__":
    main()
