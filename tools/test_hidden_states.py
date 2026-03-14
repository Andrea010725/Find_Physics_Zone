import os
import random
import sys

import numpy as np
import torch

root_path = os.path.abspath(__file__)
root_path = "/".join(root_path.split("/")[:-2])
workspace_root = os.path.dirname(root_path)
sys.path.append(root_path)

from datasets.dataset_nuplan import NuPlanTest
from models.model import TrainTransformers
from modules.tokenizers.model_tokenizer import Tokenizer
from modules.tokenizers.pose_tokenizer import poses_to_indices, yaws_to_indices
from utils.config_utils import Config
from utils.running import load_parameters



def resolve_repo_path(*parts):
    local_path = os.path.join(root_path, *parts)
    workspace_path = os.path.join(workspace_root, *parts)
    if os.path.exists(local_path):
        return local_path
    return workspace_path


CONFIG_PATH = resolve_repo_path("configs", "drivingworld_v1", "gen_videovq_conf_demo.py")
LOAD_PATH = resolve_repo_path("pretrained_models", "world_model.pth")
VQ_CKPT = resolve_repo_path("pretrained_models", "vqvae.pt")
DATA_ROOT = resolve_repo_path("data")


def build_args():
    args = Config.fromfile(CONFIG_PATH)
    args.load_path = LOAD_PATH
    args.exp_name = "debug_hidden_states"
    args.vq_ckpt = VQ_CKPT
    if not hasattr(args, "seed"):
        args.seed = 1234
    return args


def init_environment(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def print_hidden_info(hidden_states):
    print("\n===== hidden states info =====")
    print("num hidden states =", len(hidden_states))
    for i, hidden_item in enumerate(hidden_states):
        tensor = hidden_item["tensor"]
        print(
            f"[{i}] name={hidden_item['name']}, "
            f"stage={hidden_item['stage']}, "
            f"layer_idx={hidden_item['layer_idx']}, "
            f"shape={tuple(tensor.shape)}, "
            f"dtype={tensor.dtype}, "
            f"device={tensor.device}"
        )
    print("==============================\n")


def main():
    args = build_args()
    init_environment(args.seed)

    device = torch.device("cuda:0")
    local_rank = 0

    print("Loading model...")
    model = TrainTransformers(args, local_rank=local_rank, condition_frames=args.condition_frames)

    print("Loading checkpoint...")
    checkpoint = torch.load(args.load_path, map_location="cpu")
    model.model = load_parameters(model.model, checkpoint)
    model.eval()

    print("Loading tokenizer...")
    tokenizer = Tokenizer(args, local_rank)

    print("Loading dataset...")
    dataset = NuPlanTest(
        data_root=DATA_ROOT,
        json_root=DATA_ROOT,
        condition_frames=args.condition_frames,
        downsample_fps=5,
        downsample_size=16,
        h=256,
        w=512,
    )
    print("dataset len =", len(dataset))

    img, pose, yaw = dataset[0]
    img = img.unsqueeze(0).to(device)
    pose = pose.unsqueeze(0)
    yaw = yaw.unsqueeze(0)

    print("img shape =", tuple(img.shape))
    print("pose shape =", tuple(pose.shape))
    print("yaw shape =", tuple(yaw.shape))

    seq_len = args.condition_frames + 1
    img_seq = img[:, :seq_len, ...]
    pose_seq = pose[:, :seq_len, ...]
    yaw_seq = yaw[:, :seq_len, ...]

    print("Encoding images to VQ tokens/features...")
    with torch.no_grad():
        start_token_seq, _ = tokenizer.encode_to_z(img_seq)
    start_feature = tokenizer.vq_model.quantize.embedding(start_token_seq)

    pose_indices = poses_to_indices(
        pose_seq,
        model.pose_x_vocab_size,
        model.pose_y_vocab_size,
    )
    yaw_indices = yaws_to_indices(
        yaw_seq,
        model.yaw_vocab_size,
    )
    drop_flag = torch.isinf(pose_seq[:, 0, 0])

    print("Moving model and inputs to GPU...")
    model = model.to(device)
    model.eval()

    start_feature = start_feature.to(device)
    pose_indices = pose_indices.to(device)
    yaw_indices = yaw_indices.to(device)
    drop_flag = drop_flag.to(device)

    print("\nCalling model.extract_hidden_states(...) ...")
    with torch.no_grad():
        out = model.extract_hidden_states(
            feature_total=start_feature,
            pose_indices_total=pose_indices,
            yaw_indices_total=yaw_indices,
            drop_flag=drop_flag,
            return_logits=True,
        )

    print("out keys =", sorted(out.keys()))
    assert "hidden_states" in out, "hidden_states not found in output!"
    print_hidden_info(out["hidden_states"])

    print("yaw_logits shape =", tuple(out["yaw_logits"].shape))
    print("pose_x_logits shape =", tuple(out["pose_x_logits"].shape))
    print("pose_y_logits shape =", tuple(out["pose_y_logits"].shape))
    print("img_logits shape =", tuple(out["img_logits"].shape))

    print("\nSanity check passed.")


if __name__ == "__main__":
    main()
