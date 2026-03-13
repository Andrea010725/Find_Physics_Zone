import os
import sys
import torch
import random
import numpy as np
from tqdm import tqdm

root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-2])
sys.path.append(root_path)

from utils.config_utils import Config
from utils.running import load_parameters
from models.model import TrainTransformers
from modules.tokenizers.model_tokenizer import Tokenizer
from modules.tokenizers.pose_tokenizer import poses_to_indices, yaws_to_indices


# =========================
# 路径和模式
# =========================
CONFIG_PATH = "/home/zhiwen/DrivingWorld/configs/drivingworld_v1/gen_videovq_conf_demo.py"
LOAD_PATH = "/home/zhiwen/DrivingWorld/pretrained_models/world_model.pth"
VQ_CKPT = "/home/zhiwen/DrivingWorld/pretrained_models/vqvae.pt"

PHYSICS_SAMPLES_PATH = "/home/zhiwen/DrivingWorld/data/physics_samples.pt"
COHERENCE_SAMPLES_PATH = "/home/zhiwen/DrivingWorld/data/coherence_samples.pt"

PHYSICS_FEATURES_SAVE_PATH = "/home/zhiwen/DrivingWorld/data/physics_features.pt"
COHERENCE_FEATURES_SAVE_PATH = "/home/zhiwen/DrivingWorld/data/coherence_features.pt"

# 这里改成 "physics" 或 "coherence"
TASK = "coherence"


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
    torch.backends.cudnn.benchmark = True


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
    （15 condition + 1 future）
    """
    past_imgs = sample["past_imgs"]         # [15, 3, 256, 512]
    past_poses = sample["past_poses"]       # [15, 2]
    past_yaws = sample["past_yaws"]         # [15, 1]

    future_img = sample["future_img"].unsqueeze(0)   # [1, 3, 256, 512]

    if task == "physics":
        future_pose = sample["future_pose"].unsqueeze(0)    # [1, 2]
        future_yaw = sample["future_yaw"].unsqueeze(0)      # [1, 1]
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
    第一阶段：
    先用 tokenizer 把图像序列编码成 start_feature，
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
            start_token_seq, start_feature_seq = tokenizer.encode_to_z(img_seq)
            start_token = start_token_seq
            start_feature = tokenizer.vq_model.quantize.embedding(start_token)

        posedrop_input_flag = torch.isinf(pose_seq[:, 0, 0])

        pose_indices = poses_to_indices(
            pose_seq,
            model_wrapper.pose_x_vocab_size,
            model_wrapper.pose_y_vocab_size
        )
        yaw_indices = yaws_to_indices(
            yaw_seq,
            model_wrapper.yaw_vocab_size
        )

        cache_item = {
            "index": sample["index"] if "index" in sample else i,
            "feature_total": start_feature.detach().cpu(),   # [1, 16, 512, 32]
            "pose_indices_total": pose_indices.detach().cpu(),
            "yaw_indices_total": yaw_indices.detach().cpu(),
            "drop_flag": posedrop_input_flag.detach().cpu(),
        }

        # 标签也一起带着
        if task == "physics":
            cache_item["speed"] = float(sample["speed"])
            cache_item["yaw_rate"] = float(sample["yaw_rate"])
            cache_item["heading_change"] = float(sample["heading_change"])
        elif task == "coherence":
            cache_item["coherence_label"] = int(sample["coherence_label"])

        cached_inputs.append(cache_item)

        if i == 0:
            print("\nFirst cached sample:")
            print("feature_total shape =", tuple(cache_item["feature_total"].shape))
            print("pose_indices_total shape =", tuple(cache_item["pose_indices_total"].shape))
            print("yaw_indices_total shape =", tuple(cache_item["yaw_indices_total"].shape))
            print("drop_flag shape =", tuple(cache_item["drop_flag"].shape))

    print("===== Stage A done =====\n")
    return cached_inputs

def pool_hidden_state(hidden_item, effective_frames):
    """
    把单层 hidden 压成 [1536] 向量

    当前策略：
    - time_space_*: 只取最后一个时间步的前 3 个 token（yaw / pose_x / pose_y），再平均
    - ar_*: 仍然取最后一个时间位置，再对全部 token 平均
    """
    name = hidden_item["name"]
    tensor = hidden_item["tensor"]  # cuda tensor

    # time_space_*: [1, 15, 515, 1536]
    if tensor.ndim == 4:
        # 只取最后一个时间步
        last_t = tensor[:, -1, :, :]          # [1, 515, 1536]

        # 只读前 3 个结构化 token：yaw / pose_x / pose_y
        pose_tokens = last_t[:, 0:3, :]       # [1, 3, 1536]

        # 对这 3 个 token 平均，得到 [1, 1536]，再 squeeze
        feat = pose_tokens.mean(dim=1).squeeze(0)   # [1536]
        return feat.detach().cpu()

    # ar_*: [15, 515, 1536]
    elif tensor.ndim == 3:
        assert tensor.shape[0] == effective_frames, (
            f"{name} first dim {tensor.shape[0]} != {effective_frames}"
        )
        feat = tensor[-1, :, :].mean(dim=0)   # [1536]
        return feat.detach().cpu()

    else:
        raise ValueError(f"Unexpected hidden tensor ndim for {name}: {tensor.ndim}")


def extract_features_with_model(model_wrapper, cached_inputs, device, effective_frames):
    """
    第二阶段：
    用 world model 提取 hidden states，并对每层做 pooling。
    """
    print("\n===== Stage B: extracting hidden states with world model =====")

    layer_feature_lists = {}
    labels = {}

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
                drop_flag=drop_flag
            )

        hidden_states = out["hidden_states"]

        if i == 0:
            print("\nFirst sample hidden info:")
            print("num hidden states =", len(hidden_states))
            for h in hidden_states:
                print(h["name"], tuple(h["tensor"].shape))

        for h in hidden_states:
            layer_name = h["name"]
            feat = pool_hidden_state(h, effective_frames=effective_frames)  # [1536]

            if layer_name not in layer_feature_lists:
                layer_feature_lists[layer_name] = []
            layer_feature_lists[layer_name].append(feat)

        # 保存标签
        if "speed" in item:
            labels.setdefault("speed", []).append(item["speed"])
            labels.setdefault("yaw_rate", []).append(item["yaw_rate"])
            labels.setdefault("heading_change", []).append(item["heading_change"])

        if "coherence_label" in item:
            labels.setdefault("coherence_label", []).append(item["coherence_label"])
            
        # 这一轮样本处理完后，再释放显存
        del out
        del hidden_states
        del feature_total
        del pose_indices_total
        del yaw_indices_total
        del drop_flag
        torch.cuda.empty_cache()

    # stack 成 tensor
    layer_features = {}
    for layer_name, feats in layer_feature_lists.items():
        layer_features[layer_name] = torch.stack(feats, dim=0)  # [N, 1536]

    for k, v in labels.items():
        labels[k] = torch.tensor(v)

    print("===== Stage B done =====\n")
    return layer_features, labels


def save_feature_file(task, layer_features, labels):
    if task == "physics":
        save_path = PHYSICS_FEATURES_SAVE_PATH
    elif task == "coherence":
        save_path = COHERENCE_FEATURES_SAVE_PATH
    else:
        raise ValueError(f"Unknown task: {task}")

    out = {
        "task": task,
        "layer_features": layer_features,
        "labels": labels,
    }

    torch.save(out, save_path)
    print(f"Saved feature file to: {save_path}")

    print("\n===== saved feature summary =====")
    for layer_name, feats in layer_features.items():
        print(layer_name, tuple(feats.shape))
    for label_name, label_tensor in labels.items():
        print(label_name, tuple(label_tensor.shape))
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

    # effective_frames = condition_frames = 15
    effective_frames = args.condition_frames

    layer_features, labels = extract_features_with_model(
        model_wrapper=model,
        cached_inputs=cached_inputs,
        device=device,
        effective_frames=effective_frames,
    )

    save_feature_file(
        task=TASK,
        layer_features=layer_features,
        labels=labels,
    )


if __name__ == "__main__":
    main()
