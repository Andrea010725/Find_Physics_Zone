import argparse
import gc
import json
import os
import random
import sys
import time

import matplotlib.pyplot as plt
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


def parse_cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--data_root", type=str, default=data_root)
    parser.add_argument("--json_root", type=str, default=data_root)
    parser.add_argument("--samples_path", type=str, default=PHYSICS_SAMPLES_PATH)
    parser.add_argument("--layer_name", type=str, default="ar_5")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=512)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--max_cached_sequences", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output_dir", type=str, default=os.path.join(data_root, "multimodal_attention"))

    parser.add_argument("--target_speed", type=str, default="future_speed")
    parser.add_argument("--target_offset", type=str, default="future_lateral_offset")
    parser.add_argument("--target_progress", type=str, default="future_forward_progress")
    parser.add_argument("--token_stat", type=str, choices=["norm", "mean", "std"], default="norm")

    parser.add_argument("--condition_frames", type=int, default=None)
    parser.add_argument("--downsample_fps", type=int, default=5)
    parser.add_argument("--downsample_size", type=int, default=None)
    return parser.parse_args()


def build_args():
    args = Config.fromfile(CONFIG_PATH)
    args.load_path = LOAD_PATH
    args.vq_ckpt = VQ_CKPT
    args.exp_name = "visualize_multimodal_attention"
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


def normalize_attention_layer(layer_name):
    if layer_name.startswith("time_space_") and "." not in layer_name:
        return f"{layer_name}.space"
    return layer_name


def load_physics_samples(samples_path, start_index=0, max_samples=None):
    print(f"loading physics samples from: {samples_path}")
    payload = torch.load(samples_path, map_location="cpu")

    if isinstance(payload, dict) and "samples" in payload:
        samples = payload["samples"]
        sample_format = payload.get("sample_format", "unknown")
    else:
        raise ValueError("Expected lightweight physics_samples.pt with key 'samples'.")

    total = len(samples)
    start_index = int(start_index)
    if start_index < 0 or start_index >= total:
        raise ValueError(f"Invalid start_index={start_index}, total={total}")

    sliced = samples[start_index:]
    if max_samples is not None:
        sliced = sliced[:max_samples]

    print(f"sample_format={sample_format} total={total} selected={len(sliced)}")
    if len(sliced) == 0:
        raise ValueError("No samples selected. Please adjust start_index/max_samples.")
    print("first sample keys =", sorted(sliced[0].keys()))
    return sliced


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

    def build_physics_sequence(self, sample):
        imgs, poses, yaws = self.get_sequence(sample["seq_id"])
        start = int(sample["window_start"])
        end = int(sample["window_end"])
        future_t = int(sample["future_t"])

        past_imgs = imgs[start : end + 1].clone()
        past_poses = poses[start : end + 1].clone()
        past_yaws = yaws[start : end + 1].clone()

        future_img = imgs[future_t].clone().unsqueeze(0)
        future_pose = poses[future_t].clone().unsqueeze(0)
        future_yaw = yaws[future_t].clone().unsqueeze(0)

        img_seq = torch.cat([past_imgs, future_img], dim=0)
        pose_seq = torch.cat([past_poses, future_pose], dim=0)
        yaw_seq = torch.cat([past_yaws, future_yaw], dim=0)
        return img_seq, pose_seq, yaw_seq


def extract_last_hidden_frame(hidden_tensor, effective_frames):
    if hidden_tensor.ndim == 4:
        if hidden_tensor.shape[0] != 1:
            raise ValueError(f"Unexpected hidden 4D batch shape: {tuple(hidden_tensor.shape)}")
        return hidden_tensor[0, -1, :, :]

    if hidden_tensor.ndim == 3:
        if hidden_tensor.shape[0] < effective_frames:
            raise ValueError(
                f"hidden first dim={hidden_tensor.shape[0]} < effective_frames={effective_frames}"
            )
        return hidden_tensor[effective_frames - 1, :, :]

    raise ValueError(f"Unexpected hidden ndim: {hidden_tensor.ndim}")


def extract_last_attention_frame(attn_tensor, effective_frames):
    if attn_tensor.ndim != 4:
        raise ValueError(f"Unexpected attention tensor shape: {tuple(attn_tensor.shape)}")
    if attn_tensor.shape[0] < effective_frames:
        raise ValueError(
            f"attention first dim={attn_tensor.shape[0]} < effective_frames={effective_frames}"
        )
    return attn_tensor[effective_frames - 1, :, :, :]


def hidden_to_token_stat(hidden_last, mode):
    if mode == "norm":
        return hidden_last.norm(dim=-1)
    if mode == "mean":
        return hidden_last.mean(dim=-1)
    if mode == "std":
        return hidden_last.std(dim=-1)
    raise ValueError(f"Unknown token_stat mode: {mode}")


def safe_abs_corr(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    corr = np.corrcoef(x, y)[0, 1]
    if np.isnan(corr):
        return 0.0
    return float(abs(corr))


def pick_best_token(token_stat_matrix, target_values, forbidden):
    best_idx = None
    best_corr = -1.0

    for token_idx in range(token_stat_matrix.shape[1]):
        if token_idx in forbidden:
            continue
        curr_corr = safe_abs_corr(token_stat_matrix[:, token_idx], target_values)
        if curr_corr > best_corr:
            best_corr = curr_corr
            best_idx = token_idx

    if best_idx is None:
        raise RuntimeError("Cannot find candidate token index.")
    return int(best_idx), float(best_corr)


def describe_token(token_idx, latent_h, latent_w):
    if token_idx == 0:
        return "yaw"
    if token_idx == 1:
        return "pose_x"
    if token_idx == 2:
        return "pose_y"

    patch_idx = token_idx - 3
    row = patch_idx // latent_w
    col = patch_idx % latent_w
    if row >= latent_h:
        return f"img[{patch_idx}]"
    return f"img[{row},{col}]"


def plot_heatmap(matrix, x_labels, y_labels, title, save_path):
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    im = ax.imshow(matrix, cmap="magma", aspect="auto")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Attention Weight")

    ax.set_xticks(range(len(x_labels)))
    ax.set_yticks(range(len(y_labels)))
    ax.set_xticklabels(x_labels, rotation=30, ha="right")
    ax.set_yticklabels(y_labels)
    ax.set_title(title)
    ax.set_xlabel("Key Token")
    ax.set_ylabel("Query Token")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", color="white", fontsize=9)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_full_attention(attn_matrix, special_indices, title, save_path):
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(attn_matrix, cmap="viridis", aspect="auto")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean Attention Weight")

    for idx in special_indices:
        ax.axhline(idx - 0.5, color="white", linewidth=0.5, alpha=0.7)
        ax.axvline(idx - 0.5, color="white", linewidth=0.5, alpha=0.7)

    ax.set_title(title)
    ax.set_xlabel("Key Token Index")
    ax.set_ylabel("Query Token Index")
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    cli_args = parse_cli_args()
    config_args = build_args()
    init_environment(cli_args.seed)

    runtime_hparams = get_runtime_hparams(config_args, cli_args)
    latent_h = runtime_hparams["image_h"] // runtime_hparams["downsample_size"]
    latent_w = runtime_hparams["image_w"] // runtime_hparams["downsample_size"]

    device = torch.device(cli_args.device)
    attention_layer = normalize_attention_layer(cli_args.layer_name)
    hidden_layer = attention_layer.split(".")[0]

    print("===== visualize_multimodal_attention config =====")
    print("device =", device)
    print("samples_path =", cli_args.samples_path)
    print("layer_name (attention) =", attention_layer)
    print("layer_name (hidden) =", hidden_layer)
    print("start_index =", cli_args.start_index)
    print("max_samples =", cli_args.max_samples)
    print("token_stat =", cli_args.token_stat)
    print("target_speed =", cli_args.target_speed)
    print("target_offset =", cli_args.target_offset)
    print("target_progress =", cli_args.target_progress)
    print("condition_frames =", runtime_hparams["condition_frames"])
    print("downsample_fps =", runtime_hparams["downsample_fps"])
    print("downsample_size =", runtime_hparams["downsample_size"])
    print("image_size =", [runtime_hparams["image_h"], runtime_hparams["image_w"]])
    print("===============================================")

    samples = load_physics_samples(
        cli_args.samples_path,
        start_index=cli_args.start_index,
        max_samples=cli_args.max_samples,
    )

    for key in [cli_args.target_speed, cli_args.target_offset, cli_args.target_progress]:
        if key not in samples[0]:
            raise KeyError(f"Sample is missing target key: {key}")

    print("loading world model ...")
    model_wrapper = TrainTransformers(
        config_args,
        local_rank=0,
        condition_frames=runtime_hparams["condition_frames"],
    )
    checkpoint = torch.load(config_args.load_path, map_location="cpu")
    model_wrapper.model = load_parameters(model_wrapper.model, checkpoint)
    del checkpoint
    model_wrapper = model_wrapper.to(device)
    model_wrapper.eval()
    maybe_print_cuda_memory(device, prefix="[after model load]")

    print("loading tokenizer ...")
    tokenizer = Tokenizer(config_args, local_rank="cpu")
    maybe_print_cuda_memory(device, prefix="[after tokenizer load]")

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

    token_stats = []
    y_speed = []
    y_offset = []
    y_progress = []
    attn_sum = None

    total = len(samples)
    start_time = time.time()
    last_log_time = start_time

    print("\n===== extracting hidden stats + attention =====")
    for i, sample in enumerate(tqdm(samples)):
        img_seq, pose_seq, yaw_seq = sequence_provider.build_physics_sequence(sample)

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

            out = model_wrapper.extract_hidden_states(
                feature_total=feature_total,
                pose_indices_total=pose_indices_total,
                yaw_indices_total=yaw_indices_total,
                drop_flag=drop_flag,
                return_logits=False,
                return_attentions=True,
                attention_layers=[attention_layer],
            )

        hidden_item = None
        for h in out["hidden_states"]:
            if h["name"] == hidden_layer:
                hidden_item = h
                break
        if hidden_item is None:
            available = [h["name"] for h in out["hidden_states"]]
            raise KeyError(f"hidden layer {hidden_layer} not found; available={available}")

        if "attentions" not in out or attention_layer not in out["attentions"]:
            available = list(out.get("attentions", {}).keys())
            raise KeyError(f"attention layer {attention_layer} not found; available={available}")

        hidden_last = extract_last_hidden_frame(
            hidden_item["tensor"],
            effective_frames=runtime_hparams["condition_frames"],
        )
        token_stat_vec = hidden_to_token_stat(hidden_last, cli_args.token_stat)
        token_stats.append(token_stat_vec.detach().cpu().numpy().astype(np.float32))

        attn_tensor = out["attentions"][attention_layer]
        attn_last = extract_last_attention_frame(
            attn_tensor,
            effective_frames=runtime_hparams["condition_frames"],
        )
        attn_last = attn_last.mean(dim=0).detach().cpu().to(torch.float32)

        if attn_sum is None:
            attn_sum = torch.zeros_like(attn_last)
            print("first attention matrix shape =", tuple(attn_last.shape))
            print("first hidden matrix shape =", tuple(hidden_last.shape))

        attn_sum += attn_last

        y_speed.append(float(sample[cli_args.target_speed]))
        y_offset.append(float(sample[cli_args.target_offset]))
        y_progress.append(float(sample[cli_args.target_progress]))

        should_log = (
            i == 0
            or (i + 1) % cli_args.log_every == 0
            or (i + 1) == total
        )
        if should_log:
            now = time.time()
            elapsed = now - start_time
            delta = now - last_log_time
            avg = elapsed / (i + 1)
            eta = (total - (i + 1)) * avg
            print(
                f"[progress] {i + 1}/{total} elapsed={elapsed/60:.2f}min "
                f"since_last={delta:.2f}s avg={avg:.4f}s eta={eta/60:.2f}min"
            )
            maybe_print_cuda_memory(device, prefix="[progress]")
            last_log_time = now

        del out
        del feature_total
        del pose_indices_total
        del yaw_indices_total
        del drop_flag
        del img_seq
        del pose_seq
        del yaw_seq
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    num_used = len(token_stats)
    if num_used == 0:
        raise RuntimeError("No samples were processed.")

    token_stat_matrix = np.stack(token_stats, axis=0)
    y_speed = np.asarray(y_speed, dtype=np.float32)
    y_offset = np.asarray(y_offset, dtype=np.float32)
    y_progress = np.asarray(y_progress, dtype=np.float32)
    attn_mean = attn_sum / float(num_used)

    forbidden = {0}
    speed_idx, speed_corr = pick_best_token(token_stat_matrix, y_speed, forbidden)
    forbidden.add(speed_idx)
    offset_idx, offset_corr = pick_best_token(token_stat_matrix, y_offset, forbidden)
    forbidden.add(offset_idx)
    progress_idx, progress_corr = pick_best_token(token_stat_matrix, y_progress, forbidden)

    semantic_tokens = [
        {"semantic": "yaw", "token_idx": 0, "target_key": None, "abs_corr": None},
        {
            "semantic": "speed",
            "token_idx": int(speed_idx),
            "target_key": cli_args.target_speed,
            "abs_corr": float(speed_corr),
        },
        {
            "semantic": "offset",
            "token_idx": int(offset_idx),
            "target_key": cli_args.target_offset,
            "abs_corr": float(offset_corr),
        },
        {
            "semantic": "progress",
            "token_idx": int(progress_idx),
            "target_key": cli_args.target_progress,
            "abs_corr": float(progress_corr),
        },
    ]

    semantic_plot_labels = []
    semantic_text_labels = []
    semantic_indices = []
    for item in semantic_tokens:
        idx = item["token_idx"]
        token_name = describe_token(idx, latent_h, latent_w)
        item["token_name"] = token_name
        semantic_plot_labels.append(f"{item['semantic']}\n{token_name}")
        semantic_text_labels.append(f"{item['semantic']} ({token_name})")
        semantic_indices.append(idx)

    idx_tensor = torch.tensor(semantic_indices, dtype=torch.long)
    semantic_matrix = attn_mean.index_select(0, idx_tensor).index_select(1, idx_tensor).cpu().numpy()

    os.makedirs(cli_args.output_dir, exist_ok=True)
    layer_tag = attention_layer.replace(".", "_")

    semantic_fig_path = os.path.join(
        cli_args.output_dir,
        f"semantic_attention_{layer_tag}.png",
    )
    full_fig_path = os.path.join(
        cli_args.output_dir,
        f"mean_attention_full_{layer_tag}.png",
    )
    summary_json_path = os.path.join(
        cli_args.output_dir,
        f"semantic_attention_summary_{layer_tag}.json",
    )
    summary_txt_path = os.path.join(
        cli_args.output_dir,
        f"semantic_attention_summary_{layer_tag}.txt",
    )

    plot_heatmap(
        matrix=semantic_matrix,
        x_labels=semantic_plot_labels,
        y_labels=semantic_plot_labels,
        title=f"Semantic Attention ({attention_layer}, N={num_used})",
        save_path=semantic_fig_path,
    )

    plot_full_attention(
        attn_matrix=attn_mean.cpu().numpy(),
        special_indices=semantic_indices,
        title=f"Mean Attention Map ({attention_layer}, N={num_used})",
        save_path=full_fig_path,
    )

    summary = {
        "attention_layer": attention_layer,
        "hidden_layer": hidden_layer,
        "num_samples": int(num_used),
        "token_stat": cli_args.token_stat,
        "targets": {
            "speed": cli_args.target_speed,
            "offset": cli_args.target_offset,
            "progress": cli_args.target_progress,
        },
        "semantic_tokens": semantic_tokens,
        "semantic_attention_matrix": semantic_matrix.tolist(),
        "semantic_labels": semantic_text_labels,
        "output_files": {
            "semantic_heatmap_png": semantic_fig_path,
            "full_heatmap_png": full_fig_path,
        },
    }

    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    lines = [
        f"attention_layer: {attention_layer}",
        f"hidden_layer: {hidden_layer}",
        f"num_samples: {num_used}",
        f"token_stat: {cli_args.token_stat}",
        "",
        "semantic token mapping:",
    ]
    for item in semantic_tokens:
        lines.append(
            f"- {item['semantic']:8s} -> idx={item['token_idx']:4d} "
            f"name={item['token_name']:12s} "
            f"target={item['target_key']} abs_corr={item['abs_corr']}"
        )
    lines.append("")
    lines.append("semantic attention matrix (query rows -> key cols):")
    for i, row in enumerate(semantic_matrix):
        lines.append(f"- {semantic_text_labels[i]}: " + ", ".join([f"{v:.6f}" for v in row]))

    with open(summary_txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("\n===== done =====")
    print("semantic heatmap:", semantic_fig_path)
    print("full heatmap:", full_fig_path)
    print("summary json:", summary_json_path)
    print("summary txt:", summary_txt_path)


if __name__ == "__main__":
    main()
