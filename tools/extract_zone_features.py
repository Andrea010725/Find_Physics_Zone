import os
import sys
import torch
import random
import numpy as np
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

from utils.config_utils import Config
from utils.running import load_parameters
from models.model import TrainTransformers
from modules.tokenizers.model_tokenizer import Tokenizer
from modules.tokenizers.pose_tokenizer import poses_to_indices, yaws_to_indices


# =========================
# 路径和模式
# =========================
CONFIG_PATH = resolve_repo_path("configs", "drivingworld_v1", "gen_videovq_conf_demo.py")
LOAD_PATH = resolve_repo_path("pretrained_models", "world_model.pth")
VQ_CKPT = resolve_repo_path("pretrained_models", "vqvae.pt")

PHYSICS_SAMPLES_PATH = os.path.join(data_root, "physics_samples.pt")
COHERENCE_SAMPLES_PATH = os.path.join(data_root, "coherence_samples.pt")

PHYSICS_FEATURES_SAVE_PATH = os.path.join(data_root, "physics_features.pt")
COHERENCE_FEATURES_SAVE_PATH = os.path.join(data_root, "coherence_features.pt")

# 改成 "physics" 或 "coherence"
TASK = "coherence"

# physics 标签字段：和你新的 physics_samples.pt 对齐
PHYSICS_LABEL_KEYS = [
    "past_avg_speed",
    "past_avg_yaw_rate",
    "past_heading_change",
    "future_speed",
    "future_yaw_rate",
    "future_delta_yaw",
]

# 元信息字段：后面做 GroupKFold / 分类型分析会用到
META_KEYS = [
    "seq_id",
    "window_start",
    "window_end",
    "future_t",
    "negative_type",
    "source_index",
    "cond_source_index",
]


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

    # 如果你更重视可复现性，建议这样设置
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_probe_samples(task):
    if task == "physics":
        path = PHYSICS_SAMPLES_PATH
    elif task == "coherence":
        path = COHERENCE_SAMPLES_PATH
    else:
        raise ValueError(f"Unknown TASK: {task}")

    print(f"Loading probe samples from: {path}")
    samples = torch.load(path, map_location="cpu")
    print(f"num samples = {len(samples)}")
    return samples


def build_full_sequence(sample, task):
    """
    把单条 probe 样本组装成完整的 16 帧序列
    = 15 condition + 1 future
    """
    past_imgs = sample["past_imgs"]     # [15, 3, 256, 512]
    past_poses = sample["past_poses"]   # [15, 2]
    past_yaws = sample["past_yaws"]     # [15, 1]

    future_img = sample["future_img"].unsqueeze(0)  # [1, 3, 256, 512]

    if task == "physics":
        future_pose = sample["future_pose"].unsqueeze(0)   # [1, 2]
        future_yaw = sample["future_yaw"].unsqueeze(0)     # [1, 1]
    elif task == "coherence":
        future_pose = sample["future_pose_cond"].unsqueeze(0)
        future_yaw = sample["future_yaw_cond"].unsqueeze(0)
    else:
        raise ValueError(f"Unknown task: {task}")

    img_seq = torch.cat([past_imgs, future_img], dim=0)     # [16, 3, 256, 512]
    pose_seq = torch.cat([past_poses, future_pose], dim=0)  # [16, 2]
    yaw_seq = torch.cat([past_yaws, future_yaw], dim=0)     # [16, 1]

    return img_seq, pose_seq, yaw_seq


def encode_inputs_with_tokenizer(tokenizer, model_wrapper, samples, task, device):
    """
    Stage A:
    先用 tokenizer 把图像序列编码成 feature_total，
    同时把 pose/yaw 变成 indices。
    这些结果先存在 CPU 内存里，后面再交给 world model。
    """
    cached_inputs = []

    print("\n===== Stage A: encoding samples with tokenizer =====")
    for i, sample in enumerate(tqdm(samples)):
        img_seq, pose_seq, yaw_seq = build_full_sequence(sample, task)

        # 增加 batch 维
        img_seq = img_seq.unsqueeze(0).to(device)   # [1, 16, 3, 256, 512]
        pose_seq = pose_seq.unsqueeze(0)            # [1, 16, 2]
        yaw_seq = yaw_seq.unsqueeze(0)              # [1, 16, 1]

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

        # 保存标签
        if task == "physics":
            for k in PHYSICS_LABEL_KEYS:
                if k in sample:
                    cache_item[k] = float(sample[k])
        elif task == "coherence":
            cache_item["coherence_label"] = int(sample["coherence_label"])

        # 保存 meta
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
    """
    最简单公平版 pooling：
    - 对所有层统一使用同一种规则
    - 取最后一个时间步
    - 对全部 token 做 mean pooling
    输出 [D]
    """
    name = hidden_item["name"]
    tensor = hidden_item["tensor"]

    # time_space_*: [1, T, N, D]
    if tensor.ndim == 4:
        last_t = tensor[:, -1, :, :]   # [1, N, D]
        if mode == "all_mean":
            feat = last_t.mean(dim=1).squeeze(0)   # [D]
        else:
            raise ValueError(f"Unknown mode: {mode}")
        return feat.detach().cpu()

    # ar_*: [T, N, D]
    elif tensor.ndim == 3:
        assert tensor.shape[0] == effective_frames, (
            f"{name}: first dim {tensor.shape[0]} != effective_frames {effective_frames}"
        )
        last_t = tensor[-1, :, :]      # [N, D]
        if mode == "all_mean":
            feat = last_t.mean(dim=0)  # [D]
        else:
            raise ValueError(f"Unknown mode: {mode}")
        return feat.detach().cpu()

    else:
        raise ValueError(f"Unexpected hidden tensor ndim for {name}: {tensor.ndim}")


def extract_features_with_model(model_wrapper, cached_inputs, device, effective_frames):
    """
    Stage B:
    用 world model 提取 hidden states，并对每层做 pooling。
    """
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

        # 逐层 pooling
        for h in hidden_states:
            layer_name = h["name"]
            feat = pool_hidden_state(h, effective_frames=effective_frames, mode="all_mean")

            if layer_name not in layer_feature_lists:
                layer_feature_lists[layer_name] = []
            layer_feature_lists[layer_name].append(feat)

        # 保存 labels
        for key in PHYSICS_LABEL_KEYS:
            if key in item:
                labels.setdefault(key, []).append(item[key])

        if "coherence_label" in item:
            labels.setdefault("coherence_label", []).append(item["coherence_label"])

        # 保存 meta
        for k in META_KEYS:
            if k in item:
                meta.setdefault(k, []).append(item[k])

        # 释放显存
        del out
        del hidden_states
        del feature_total
        del pose_indices_total
        del yaw_indices_total
        del drop_flag

    # stack 成 tensor
    layer_features = {}
    for layer_name, feats in layer_feature_lists.items():
        layer_features[layer_name] = torch.stack(feats, dim=0)  # [N, D]

    # labels 转 tensor
    for k, v in labels.items():
        if k == "coherence_label":
            labels[k] = torch.tensor(v, dtype=torch.long)
        else:
            labels[k] = torch.tensor(v, dtype=torch.float32)

    # meta 转 tensor（如果是数值）
    for k, v in meta.items():
        if len(v) > 0 and isinstance(v[0], (int, float, np.integer, np.floating)):
            meta[k] = torch.tensor(v)
        else:
            meta[k] = v

    print("===== Stage B done =====\n")
    return layer_features, labels, meta


def save_feature_file(task, layer_features, labels, meta):
    if task == "physics":
        save_path = PHYSICS_FEATURES_SAVE_PATH
    elif task == "coherence":
        save_path = COHERENCE_FEATURES_SAVE_PATH
    else:
        raise ValueError(f"Unknown task: {task}")

    out = {
        "task": task,
        "feature_mode": "all_mean_last_t",
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
    args = build_args()
    init_environment(args.seed)

    device = torch.device("cuda:0")

    print("Loading world model on CPU...")
    model = TrainTransformers(args, local_rank=0, condition_frames=args.condition_frames)
    checkpoint = torch.load(args.load_path, map_location="cpu")
    model.model = load_parameters(model.model, checkpoint)
    model.eval()

    samples = load_probe_samples(TASK)

    print("Loading tokenizer on GPU...")
    tokenizer = Tokenizer(args, local_rank=0)

    cached_inputs = encode_inputs_with_tokenizer(
        tokenizer=tokenizer,
        model_wrapper=model,
        samples=samples,
        task=TASK,
        device=device,
    )

    print("Releasing tokenizer GPU memory...")
    del tokenizer
    torch.cuda.empty_cache()

    print("Moving world model to GPU...")
    model = model.to(device)
    model.eval()

    # 目前按 DrivingWorld 当前实现，AR hidden 的时间长度等于 condition_frames
    effective_frames = args.condition_frames

    layer_features, labels, meta = extract_features_with_model(
        model_wrapper=model,
        cached_inputs=cached_inputs,
        device=device,
        effective_frames=effective_frames,
    )

    save_feature_file(
        task=TASK,
        layer_features=layer_features,
        labels=labels,
        meta=meta,
    )


if __name__ == "__main__":
    main()
