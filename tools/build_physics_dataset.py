import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
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


def safe_yaw_diff(yaw2, yaw1):
    """Return wrapped yaw difference in [-180, 180] degrees."""
    diff = yaw2 - yaw1
    return torch.remainder(diff + 180.0, 360.0) - 180.0


def compute_step_motion(poses, yaws, t, fps):
    """
    Compute one-step motion from t-1 -> t.
    poses: [T, 2]
    yaws:  [T, 1]
    """
    delta_xy = poses[t] - poses[t - 1]
    delta_yaw = safe_yaw_diff(yaws[t, 0], yaws[t - 1, 0])

    speed = torch.norm(delta_xy) * fps
    yaw_rate = delta_yaw * fps

    return {
        "delta_x": float(delta_xy[0].item()),
        "delta_y": float(delta_xy[1].item()),
        "delta_yaw": float(delta_yaw.item()),
        "speed": float(speed.item()),
        "yaw_rate": float(yaw_rate.item()),
    }


def compute_past_summary(poses, yaws, start, k, fps):
    """
    Summarize motion over past window [start, start+k-1].
    """
    end = start + k - 1

    speeds = []
    yaw_rates = []
    for t in range(start + 1, end + 1):
        m = compute_step_motion(poses, yaws, t, fps)
        speeds.append(m["speed"])
        yaw_rates.append(m["yaw_rate"])

    heading_change = safe_yaw_diff(yaws[end, 0], yaws[start, 0])

    return {
        "past_avg_speed": float(np.mean(speeds)),
        "past_final_speed": float(speeds[-1]),
        "past_avg_yaw_rate": float(np.mean(yaw_rates)),
        "past_final_yaw_rate": float(yaw_rates[-1]),
        "past_heading_change": float(heading_change.item()),
    }


def future_turn_class(delta_yaw, thresh=1.0):
    """
    0: left, 1: straight, 2: right
    """
    if delta_yaw > thresh:
        return 0
    elif delta_yaw < -thresh:
        return 2
    else:
        return 1


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

    k = args.condition_frames
    h = args.future_horizon
    fps = args.downsample_fps

    samples = []

    for seq_id in tqdm(range(len(dataset)), desc="building physics samples"):
        imgs, poses, yaws = dataset[seq_id]
        T = imgs.shape[0]

        assert poses.shape[0] == T
        assert yaws.shape[0] == T

        max_start = T - k - h + 1
        if max_start <= 0:
            continue

        for start in range(0, max_start, args.stride):
            past_end = start + k - 1
            future_t = past_end + h

            past_imgs = imgs[start: start + k].clone()
            past_poses = poses[start: start + k].clone()
            past_yaws = yaws[start: start + k].clone()

            future_img = imgs[future_t].clone()
            future_pose = poses[future_t].clone()
            future_yaw = yaws[future_t].clone()

            past_summary = compute_past_summary(poses, yaws, start, k, fps)
            future_motion = compute_step_motion(poses, yaws, future_t, fps)

            sample = {
                "seq_id": int(seq_id),
                "window_start": int(start),
                "window_end": int(past_end),
                "future_t": int(future_t),

                "past_imgs": past_imgs,
                "past_poses": past_poses,
                "past_yaws": past_yaws,

                "future_img": future_img,
                "future_pose": future_pose,
                "future_yaw": future_yaw,

                # past-summary targets
                **past_summary,

                # future-step targets
                "future_delta_x": future_motion["delta_x"],
                "future_delta_y": future_motion["delta_y"],
                "future_delta_yaw": future_motion["delta_yaw"],
                "future_speed": future_motion["speed"],
                "future_yaw_rate": future_motion["yaw_rate"],
                "future_turn_class": future_turn_class(future_motion["delta_yaw"]),
            }
            samples.append(sample)

    save_path = os.path.join(args.save_root, "physics_samples.pt")
    torch.save(samples, save_path)

    print(f"saved to {save_path}")
    print(f"num samples = {len(samples)}")

    for key in [
        "past_avg_speed", "past_final_speed", "past_avg_yaw_rate",
        "past_heading_change", "future_speed", "future_yaw_rate",
        "future_delta_yaw"
    ]:
        vals = [s[key] for s in samples]
        print(f"{key:20s} mean={np.mean(vals):.6f}, std={np.std(vals):.6f}, "
              f"min={np.min(vals):.6f}, max={np.max(vals):.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default=data_root)
    parser.add_argument("--json_root", type=str, default=data_root)
    parser.add_argument("--save_root", type=str, default=data_root)
    parser.add_argument("--condition_frames", type=int, default=15)
    parser.add_argument("--future_horizon", type=int, default=1)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--downsample_fps", type=int, default=5)
    parser.add_argument("--downsample_size", type=int, default=16)
    parser.add_argument("--h", type=int, default=256)
    parser.add_argument("--w", type=int, default=512)
    args = parser.parse_args()
    main(args)
