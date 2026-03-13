import os
import sys
import torch
import random
import numpy as np

root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-2])
sys.path.append(root_path)

from utils.config_utils import Config
from utils.running import load_parameters
from models.model import TrainTransformers
from modules.tokenizers.model_tokenizer import Tokenizer
from modules.tokenizers.pose_tokenizer import poses_to_indices, yaws_to_indices
import sys
sys.path.append("/home/zhiwen/DrivingWorld/")
from datasets.dataset_nuplan import NuPlanTest


# =========================
# 写死你的路径
# =========================
CONFIG_PATH = "/home/zhiwen/DrivingWorld/configs/drivingworld_v1/gen_videovq_conf_demo.py"
LOAD_PATH = "/home/zhiwen/DrivingWorld/pretrained_models/world_model.pth"


def build_args():
    """
    不走命令行，直接从 config 文件构造 args，
    并把必要字段补进去。
    """
    args = Config.fromfile(CONFIG_PATH)

    # 补你原来 argparse 会提供的字段
    args.load_path = LOAD_PATH
    args.exp_name = "debug_hidden_states"
    args.vq_ckpt = "/home/zhiwen/DrivingWorld/pretrained_models/vqvae.pt"


    # 如果 config 里没有 seed，就手动补一个
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
    for i, h in enumerate(hidden_states):
        name = h["name"]
        tensor = h["tensor"]
        print(
            f"[{i}] name={name}, "
            f"shape={tuple(tensor.shape)}, "
            f"dtype={tensor.dtype}, "
            f"device={tensor.device}"
        )
    print("==============================\n")


def main():
    args = build_args()
    init_environment(args.seed)

    local_rank = 0

    print("Loading model...")
    model = TrainTransformers(args, local_rank=local_rank, condition_frames=args.condition_frames)

    print("Loading checkpoint...")
    checkpoint = torch.load(args.load_path, map_location="cpu")
    model.model = load_parameters(model.model, checkpoint)
    model.eval()

    device = torch.device("cuda:0")

    print("Loading tokenizer...")
    tokenizer = Tokenizer(args, local_rank)

    print("Loading dataset...")
    test_data = NuPlanTest(
    	data_root="/home/zhiwen/DrivingWorld/data",
    	json_root="/home/zhiwen/DrivingWorld/data",
    	condition_frames=args.condition_frames,
    	downsample_fps=5,
    	downsample_size=16,
    	h=256,
    	w=512,
	)
    print("dataset len =", len(test_data))

    img, pose, yaw = test_data[0]

    # 这里只把图像先放到 GPU，给 tokenizer 用
    img = img.unsqueeze(0).to(device)   # [1, T, 3, H, W]

    # pose / yaw 先留在 CPU，后面再统一搬
    pose = pose.unsqueeze(0)            # [1, T, 2]
    yaw = yaw.unsqueeze(0)              # [1, T, 1]

    print("img shape =", img.shape)
    print("pose shape =", pose.shape)
    print("yaw shape =", yaw.shape)

    condition_frames = args.condition_frames
    print("condition_frames =", condition_frames)

    # 和 demo 保持一致的图像编码流程
    # 这里对 GPT.forward() 测试，要给 condition_frames + 1 帧
    seq_len = condition_frames + 1

    img_seq = img[:, :seq_len, ...]      # [1, 16, 3, H, W]
    pose_now = pose[:, :seq_len, ...]    # [1, 16, 2]
    yaw_now = yaw[:, :seq_len, ...]      # [1, 16, 1]

    print("img_seq shape =", img_seq.shape)
    print("pose_now shape =", pose_now.shape)
    print("yaw_now shape =", yaw_now.shape)

    print("Encoding images to VQ tokens/features...")
    with torch.no_grad():
        start_token_seq, start_feature_seq = tokenizer.encode_to_z(img_seq)

    # 这里不要再裁成 15，直接保留完整 16 帧
    start_token = start_token_seq
    start_feature = tokenizer.vq_model.quantize.embedding(start_token)

    posedrop_input_flag = torch.isinf(pose_now[:, 0, 0])

    pose_indices = poses_to_indices(
    	pose_now,
    	model.pose_x_vocab_size,
    	model.pose_y_vocab_size
	)
    yaw_indices = yaws_to_indices(
    	yaw_now,
    	model.yaw_vocab_size
	)

    posedrop_input_flag = torch.isinf(pose_now[:, 0, 0])

    pose_indices = poses_to_indices(
    	pose_now,
    	model.pose_x_vocab_size,
    	model.pose_y_vocab_size
	)
    yaw_indices = yaws_to_indices(
    	yaw_now,
    	model.yaw_vocab_size
	)

    # 现在才把 world model 和最终输入搬到 GPU
    print("Moving model and inputs to GPU...")
    model = model.to(device)
    model.eval()

    start_feature = start_feature.to(device)
    pose_indices = pose_indices.to(device)
    yaw_indices = yaw_indices.to(device)
    posedrop_input_flag = posedrop_input_flag.to(device)

    print("model device =", next(model.parameters()).device)
    print("start_feature device =", start_feature.device)
    print("pose_indices device =", pose_indices.device)
    print("yaw_indices device =", yaw_indices.device)
    print("posedrop_input_flag device =", posedrop_input_flag.device)

    print("start_token shape =", start_token.shape)
    print("start_feature shape =", start_feature.shape)
    print("pose_now shape =", pose_now.shape)
    print("yaw_now shape =", yaw_now.shape)
    print("pose_indices shape =", pose_indices.shape)
    print("yaw_indices shape =", yaw_indices.shape)
    print("posedrop_input_flag shape =", posedrop_input_flag.shape)

    print("\nCalling model.extract_hidden_states(...) ...")
    with torch.no_grad():
    	out = model.extract_hidden_states(
        	feature_total=start_feature,
        	pose_indices_total=pose_indices,
        	yaw_indices_total=yaw_indices,
        	drop_flag=posedrop_input_flag
    	)

    print("out keys =", out.keys())
    assert "hidden_states" in out, "hidden_states not found in output!"

    print_hidden_info(out["hidden_states"])

    print("yaw_logits shape =", out["yaw_logits"].shape)
    print("pose_x_logits shape =", out["pose_x_logits"].shape)
    print("pose_y_logits shape =", out["pose_y_logits"].shape)
    print("img_logits shape =", out["img_logits"].shape)

    print("\nSanity check passed.")


if __name__ == "__main__":
    main()
