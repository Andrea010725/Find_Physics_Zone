import os
import sys

root_path = os.path.abspath(__file__)
root_path = "/".join(root_path.split("/")[:-2])
workspace_root = os.path.dirname(root_path)


def resolve_repo_path(*parts):
    local_path = os.path.join(root_path, *parts)
    workspace_path = os.path.join(workspace_root, *parts)
    if os.path.exists(local_path):
        return local_path
    return workspace_path


data_root = resolve_repo_path("data")
sys.path.append(root_path)

from datasets.dataset_nuplan import NuPlanTest


dataset = NuPlanTest(
    data_root=data_root,
    json_root=data_root,
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
