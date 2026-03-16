import argparse
import os
import random
import sys

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
    parser.add_argument("--pool_mode", type=str, default="all_mean")
    parser.add_argument("--data_root", type=str, default=data_root)
    parser.add_argument("--json_root", type=str, default=data_root)
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


def load_probe_samples(task):
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
    else:
        samples = payload
        sample_format = "legacy_tensor_list"

    print(f"probe sample_format = {sample_format}")
    print(f"num samples = {len(samples)}")
    return samples, sample_format


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


def encode_inputs_with_tokenizer(tokenizer, model_wrapper, samples, task, device, sequence_provider):
    cached_inputs = []

    print("\n===== Stage A: encoding samples with tokenizer =====")
    for i, sample in enumerate(tqdm(samples)):
        img_seq, pose_seq, yaw_seq = build_full_sequence(sample, task, sequence_provider)

        img_seq = img_seq.unsqueeze(0).to(device)
        pose_seq = pose_seq.unsqueeze(0)
        yaw_seq = yaw_seq.unsqueeze(0)

        with torch.no_grad():
            start_token_seq, _ = tokenizer.encode_to_z(img_seq)
            start_feature = tokenizer.vq_model.quantize.embedding(start_token_seq)

        posedrop_input_flag = torch.isinf(pose_seq[:, 0, 0])

        pose_indices = poses_to_indices(
            pose_seq,
            model_wrapper.pose_x_vocab_size,
            model_wrapper.pose_y_vocab_size,
        )
        yaw_indices = yaws_to_indices(
            yaw_seq,
            model_wrapper.yaw_vocab_size,
        )

        cache_item = {
            "index": sample["index"] if "index" in sample else i,
            "feature_total": start_feature.detach().cpu(),
            "pose_indices_total": pose_indices.detach().cpu(),
            "yaw_indices_total": yaw_indices.detach().cpu(),
            "drop_flag": posedrop_input_flag.detach().cpu(),
        }

        if task == "physics":
            for k in PHYSICS_LABEL_KEYS:
                if k in sample:
                    cache_item[k] = float(sample[k])
        elif task == "coherence":
            cache_item["coherence_label"] = int(sample["coherence_label"])

        for k in META_KEYS:
            if k in sample:
                cache_item[k] = sample[k]

        cached_inputs.append(cache_item)

        if i == 0:
            print("\nFirst cached sample:")
            print("feature_total shape =", tuple(cache_item["feature_total"].shape))
            print("pose_indices_total shape =", tuple(cache_item["pose_indices_total"].shape))
            print("yaw_indices_total shape =", tuple(cache_item["yaw_indices_total"].shape))
            print("drop_flag shape =", tuple(cache_item["drop_flag"].shape))

            for k in PHYSICS_LABEL_KEYS:
                if k in cache_item:
                    print(f"{k} =", cache_item[k])

            if "coherence_label" in cache_item:
                print("coherence_label =", cache_item["coherence_label"])

            for k in META_KEYS:
                if k in cache_item:
                    print(f"{k} =", cache_item[k])

    print("===== Stage A done =====\n")
    return cached_inputs


def pool_hidden_state(hidden_item, effective_frames, mode="all_mean"):
    name = hidden_item["name"]
    tensor = hidden_item["tensor"]

    if tensor.ndim == 4:
        last_t = tensor[:, -1, :, :]
        if mode == "all_mean":
            feat = last_t.mean(dim=1).squeeze(0)
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
        else:
            raise ValueError(f"Unknown mode: {mode}")
        return feat.detach().cpu()

    raise ValueError(f"Unexpected hidden tensor ndim for {name}: {tensor.ndim}")


def extract_features_with_model(model_wrapper, cached_inputs, device, effective_frames, pool_mode):
    print("\n===== Stage B: extracting hidden states with world model =====")

    layer_feature_lists = {}
    labels = {}
    meta = {}

    for i, item in enumerate(tqdm(cached_inputs)):
        feature_total = item["feature_total"].to(device)
        pose_indices_total = item["pose_indices_total"].to(device)
        yaw_indices_total = item["yaw_indices_total"].to(device)
        drop_flag = item["drop_flag"].to(device)

        with torch.no_grad():
            out = model_wrapper.extract_hidden_states(
                feature_total=feature_total,
                pose_indices_total=pose_indices_total,
                yaw_indices_total=yaw_indices_total,
                drop_flag=drop_flag,
            )

        hidden_states = out["hidden_states"]

        if i == 0:
            print("\nFirst sample hidden info:")
            print("num hidden states =", len(hidden_states))
            for h in hidden_states:
                print(h["name"], tuple(h["tensor"].shape))

        for h in hidden_states:
            layer_name = h["name"]
            feat = pool_hidden_state(h, effective_frames=effective_frames, mode=pool_mode)

            if layer_name not in layer_feature_lists:
                layer_feature_lists[layer_name] = []
            layer_feature_lists[layer_name].append(feat)

        for key in PHYSICS_LABEL_KEYS:
            if key in item:
                labels.setdefault(key, []).append(item[key])

        if "coherence_label" in item:
            labels.setdefault("coherence_label", []).append(item["coherence_label"])

        for k in META_KEYS:
            if k in item:
                meta.setdefault(k, []).append(item[k])

        del out
        del hidden_states
        del feature_total
        del pose_indices_total
        del yaw_indices_total
        del drop_flag

    layer_features = {}
    for layer_name, feats in layer_feature_lists.items():
        layer_features[layer_name] = torch.stack(feats, dim=0)

    for k, v in labels.items():
        if k == "coherence_label":
            labels[k] = torch.tensor(v, dtype=torch.long)
        else:
            labels[k] = torch.tensor(v, dtype=torch.float32)

    for k, v in meta.items():
        if len(v) > 0 and isinstance(v[0], (int, float, np.integer, np.floating)):
            meta[k] = torch.tensor(v)
        else:
            meta[k] = v

    print("===== Stage B done =====\n")
    return layer_features, labels, meta


def save_feature_file(task, layer_features, labels, meta, pool_mode, sample_format):
    if task == "physics":
        save_path = PHYSICS_FEATURES_SAVE_PATH
    elif task == "coherence":
        save_path = COHERENCE_FEATURES_SAVE_PATH
    else:
        raise ValueError(f"Unknown task: {task}")

    out = {
        "task": task,
        "feature_mode": f"{pool_mode}_last_t",
        "probe_sample_format": sample_format,
        "stage_semantics": {
            "time_space": "Condition-only representation. Built from frames 0..T-1 before injecting next-state query tokens.",
            "ar": "Teacher-forced next-state representation. Built after injecting next yaw/pose query and next-frame image-token prefix.",
        },
        "layer_features": layer_features,
        "labels": labels,
        "meta": meta,
    }

    torch.save(out, save_path)
    print(f"Saved feature file to: {save_path}")

    print("\n===== saved feature summary =====")
    print("feature_mode =", out["feature_mode"])
    print("probe_sample_format =", out["probe_sample_format"])
    for layer_name, feats in layer_features.items():
        print(layer_name, tuple(feats.shape))
    for label_name, label_tensor in labels.items():
        print(label_name, tuple(label_tensor.shape))
    for meta_name, meta_value in meta.items():
        if isinstance(meta_value, torch.Tensor):
            print(meta_name, tuple(meta_value.shape))
        else:
            print(meta_name, f"list(len={len(meta_value)})")
    print("=================================\n")


def main():
    cli_args = parse_cli_args()
    args = build_args()
    init_environment(args.seed)

    device = torch.device(cli_args.device)

    print("loading world model on CPU...")
    model = TrainTransformers(args, local_rank=0, condition_frames=args.condition_frames)
    checkpoint = torch.load(args.load_path, map_location="cpu")
    model.model = load_parameters(model.model, checkpoint)
    model.eval()

    samples, sample_format = load_probe_samples(cli_args.task)

    sequence_provider = None
    if sample_format != "legacy_tensor_list":
        print("initializing lazy sequence provider for lightweight samples...")
        sequence_provider = SequenceProvider(
            data_root=cli_args.data_root,
            json_root=cli_args.json_root,
            condition_frames=args.condition_frames,
            downsample_fps=args.downsample_fps,
            downsample_size=args.downsample_size,
            h=args.image_size[0],
            w=args.image_size[1],
        )

    print("loading tokenizer on GPU...")
    tokenizer = Tokenizer(args, local_rank=device)

    cached_inputs = encode_inputs_with_tokenizer(
        tokenizer=tokenizer,
        model_wrapper=model,
        samples=samples,
        task=cli_args.task,
        device=device,
        sequence_provider=sequence_provider,
    )

    print("releasing tokenizer GPU memory...")
    del tokenizer
    torch.cuda.empty_cache()

    print("moving world model to GPU...")
    model = model.to(device)
    model.eval()

    effective_frames = args.condition_frames

    layer_features, labels, meta = extract_features_with_model(
        model_wrapper=model,
        cached_inputs=cached_inputs,
        device=device,
        effective_frames=effective_frames,
        pool_mode=cli_args.pool_mode,
    )

    save_feature_file(
        task=cli_args.task,
        layer_features=layer_features,
        labels=labels,
        meta=meta,
        pool_mode=cli_args.pool_mode,
        sample_format=sample_format,
    )


if __name__ == "__main__":
    main()
