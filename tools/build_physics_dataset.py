import os
import argparse
import torch
from tqdm import tqdm
import sys
sys.path.append("/home/zhiwen/DrivingWorld/")
from datasets.dataset_nuplan import NuPlanTest
import numpy as np


def compute_speed(poses, k, fps):
    delta = poses[k - 1] - poses[k - 2]   # poses shape: [T, 2]
    speed = torch.norm(delta).item() * fps
    return float(speed)


def compute_yaw_rate(yaws, k, fps):
    yaw_rate = (yaws[k - 1, 0] - yaws[k - 2, 0]).item() * fps
    return float(yaw_rate)


def compute_heading_change(yaws, k):
    heading_change = (yaws[k - 1, 0] - yaws[0, 0]).item()
    return float(heading_change)


def main(args):
    os.makedirs(args.save_root, exist_ok=True)

    dataset = NuPlanTest(
        data_root=args.data_root,
        json_root=args.json_root,
        condition_frames=args.condition_frames,
        downsample_fps=args.downsample_fps,
        downsample_size=args.downsample_size,
        h=args.h,
        w=args.w,
    )

    fps = args.downsample_fps
    k = args.condition_frames

    samples = []

    for i in tqdm(range(len(dataset))):
        imgs, poses, yaws = dataset[i]

        assert imgs.shape[0] > k
        assert poses.shape[0] > k
        assert yaws.shape[0] > k

        sample = {
            "index": i,
            "past_imgs": imgs[:k].clone(),
            "past_poses": poses[:k].clone(),
            "past_yaws": yaws[:k].clone(),
            "future_img": imgs[k].clone(),
            "future_pose": poses[k].clone(),
            "future_yaw": yaws[k].clone(),
            "speed": compute_speed(poses, k, fps),
            "yaw_rate": compute_yaw_rate(yaws, k, fps),
            "heading_change": compute_heading_change(yaws, k),
        }
        samples.append(sample)

        if i == 0:
            print("first sample:")
            print("past_imgs shape:", sample["past_imgs"].shape)
            print("past_poses shape:", sample["past_poses"].shape)
            print("past_yaws shape:", sample["past_yaws"].shape)
            print("future_pose:", sample["future_pose"])
            print("future_yaw:", sample["future_yaw"])
            print("speed:", sample["speed"])
            print("yaw_rate:", sample["yaw_rate"])
            print("heading_change:", sample["heading_change"])

    save_path = os.path.join(args.save_root, "physics_samples.pt")
    torch.save(samples, save_path)
    print(f"saved to {save_path}")
    print(f"num samples = {len(samples)}")
    
    speeds = [s["speed"] for s in samples]
    yaw_rates = [s["yaw_rate"] for s in samples]
    heading_changes = [s["heading_change"] for s in samples]
    print("\n===== statistics =====")
    print(f"speed          mean={np.mean(speeds):.6f}, std={np.std(speeds):.6f}, min={np.min(speeds):.6f}, max={np.max(speeds):.6f}")
    print(f"yaw_rate       mean={np.mean(yaw_rates):.6f}, std={np.std(yaw_rates):.6f}, min={np.min(yaw_rates):.6f}, max={np.max(yaw_rates):.6f}")
    print(f"heading_change mean={np.mean(heading_changes):.6f}, std={np.std(heading_changes):.6f}, min={np.min(heading_changes):.6f}, max={np.max(heading_changes):.6f}")
    print("======================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="/home/zhiwen/DrivingWorld/data")
    parser.add_argument("--json_root", type=str, default="/home/zhiwen/DrivingWorld/data")
    parser.add_argument("--save_root", type=str, default="/home/zhiwen/DrivingWorld/data")

    parser.add_argument("--condition_frames", type=int, default=15)
    parser.add_argument("--downsample_fps", type=int, default=5)
    parser.add_argument("--downsample_size", type=int, default=16)
    parser.add_argument("--h", type=int, default=256)
    parser.add_argument("--w", type=int, default=512)

    args = parser.parse_args()
    main(args)
