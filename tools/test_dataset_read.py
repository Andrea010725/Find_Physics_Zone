import sys
sys.path.append("/home/zhiwen/DrivingWorld/")
import torch
from datasets.dataset_nuplan import NuPlanTest

dataset = NuPlanTest(
    data_root="/",   # 因为你的 seq_meta 里 data_root 很可能已经是绝对路径
    json_root="/home/zhiwen/DrivingWorld/data",
    condition_frames=3,
    downsample_fps=5,
    downsample_size=16,
    h=256,
    w=512,
)

print("dataset length =", len(dataset))

imgs, poses, yaws = dataset[0]

print("imgs shape =", imgs.shape)
print("poses shape =", poses.shape)
print("yaws shape =", yaws.shape)

print("imgs dtype =", imgs.dtype)
print("poses dtype =", poses.dtype)
print("yaws dtype =", yaws.dtype)

print("imgs min/max =", imgs.min().item(), imgs.max().item())
print("first pose =", poses[0])
print("first yaw =", yaws[0])
