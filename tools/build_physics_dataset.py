import sys
import os
import argparse

import numpy as np
import torch
from tqdm import tqdm

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


LEGACY_LABEL_KEYS = [
    "past_avg_speed",
    "past_final_speed",
    "past_avg_yaw_rate",
    "past_final_yaw_rate",
    "past_heading_change",
    "future_delta_x",
    "future_delta_y",
    "future_delta_yaw",
    "future_speed",
    "future_yaw_rate",
    "future_turn_class",
]


def build_rollout_head_keys(num_keypoints):
    keys = []
    for idx in range(1, num_keypoints + 1):
        keys.extend([f"rollout_kp{idx}_x", f"rollout_kp{idx}_y"])
    return keys


def build_head_families(num_keypoints):
    return {
        "control_head": [
            "control_delta_v",
            "control_cum_delta_yaw",
        ],
        "endpoint_head": [
            "endpoint_lateral_disp",
            "endpoint_forward_progress",
            "endpoint_heading",
        ],
        "geometry_head": [
            "geometry_mean_curvature",
            "geometry_integrated_curvature",
        ],
        "rollout_head": build_rollout_head_keys(num_keypoints),
    }


def compute_step_motion(poses, yaws, t, fps):
    """
    Compute one-step motion from t-1 -> t.

    Note:
    `NuPlanTest` already returns per-step relative pose/yaw, not absolute trajectory.
    So the step motion at time t is stored directly in poses[t] / yaws[t], rather than
    in the difference between poses[t] and poses[t - 1].

    poses: [T, 2]
    yaws:  [T, 1]
    """
    if t <= 0:
        raise ValueError("Step motion is only defined for t >= 1.")

    delta_xy = poses[t]
    delta_yaw = yaws[t, 0]

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

    heading_change = yaws[start + 1: end + 1, 0].sum()

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


def compose_future_rollout(poses, yaws, start_step, rollout_steps, fps):
    """
    Compose future relative motions into an endpoint/path in the last observed ego frame.
    """
    xs = []
    ys = []
    step_lengths = []
    delta_yaws_deg = []

    x = 0.0
    y = 0.0
    heading_deg = 0.0

    for step_idx in range(start_step + 1, start_step + rollout_steps + 1):
        step_motion = compute_step_motion(poses, yaws, step_idx, fps)
        dx_local = step_motion["delta_x"]
        dy_local = step_motion["delta_y"]
        heading_rad = np.deg2rad(heading_deg)

        dx_ref = np.cos(heading_rad) * dx_local - np.sin(heading_rad) * dy_local
        dy_ref = np.sin(heading_rad) * dx_local + np.cos(heading_rad) * dy_local

        x += dx_ref
        y += dy_ref
        heading_deg += step_motion["delta_yaw"]

        xs.append(float(x))
        ys.append(float(y))
        step_lengths.append(float(np.hypot(dx_local, dy_local)))
        delta_yaws_deg.append(float(step_motion["delta_yaw"]))

    return {
        "xs": xs,
        "ys": ys,
        "step_lengths": step_lengths,
        "delta_yaws_deg": delta_yaws_deg,
        "endpoint_heading_deg": float(heading_deg),
    }


def build_control_targets(poses, yaws, past_end, rollout_steps, fps):
    current_motion = compute_step_motion(poses, yaws, past_end, fps)
    final_motion = compute_step_motion(poses, yaws, past_end + rollout_steps, fps)
    cum_delta_yaw = float(yaws[past_end + 1: past_end + rollout_steps + 1, 0].sum().item())
    return {
        "control_delta_v": float(final_motion["speed"] - current_motion["speed"]),
        "control_cum_delta_yaw": cum_delta_yaw,
    }


def build_endpoint_targets(rollout_state):
    return {
        "endpoint_lateral_disp": float(rollout_state["ys"][-1]),
        "endpoint_forward_progress": float(rollout_state["xs"][-1]),
        "endpoint_heading": float(rollout_state["endpoint_heading_deg"]),
    }


def build_geometry_targets(rollout_state):
    total_arc_length = float(np.sum(rollout_state["step_lengths"]))
    total_heading_rad = float(np.deg2rad(np.sum(rollout_state["delta_yaws_deg"])))
    if total_arc_length < 1e-6:
        mean_curvature = 0.0
    else:
        mean_curvature = total_heading_rad / total_arc_length

    integrated_curvature = float(np.sum(np.abs(np.deg2rad(rollout_state["delta_yaws_deg"]))))
    return {
        "geometry_mean_curvature": float(mean_curvature),
        "geometry_integrated_curvature": integrated_curvature,
    }


def select_rollout_keypoints(rollout_state, rollout_steps, rollout_keypoints):
    keypoint_indices = np.linspace(0, rollout_steps - 1, num=rollout_keypoints)
    keypoint_indices = np.round(keypoint_indices).astype(np.int64)
    out = {}
    for kp_slot, rollout_idx in enumerate(keypoint_indices, start=1):
        out[f"rollout_kp{kp_slot}_x"] = float(rollout_state["xs"][rollout_idx])
        out[f"rollout_kp{kp_slot}_y"] = float(rollout_state["ys"][rollout_idx])
    return out


def main(args):
    os.makedirs(args.save_root, exist_ok=True)

    if args.rollout_steps < 1:
        raise ValueError("--rollout_steps must be >= 1")
    if args.rollout_keypoints < 1:
        raise ValueError("--rollout_keypoints must be >= 1")
    if args.rollout_keypoints > args.rollout_steps:
        raise ValueError("--rollout_keypoints cannot exceed --rollout_steps in the simple evenly-spaced setup.")
    if args.condition_frames < 2:
        raise ValueError("--condition_frames must be >= 2 so past summary / delta_v are well-defined.")

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
    rollout_steps = args.rollout_steps
    required_future = max(h, rollout_steps)
    head_families = build_head_families(args.rollout_keypoints)

    samples = []

    for seq_id in tqdm(range(len(dataset)), desc="building physics samples"):
        imgs, poses, yaws = dataset[seq_id]
        T = imgs.shape[0]

        assert poses.shape[0] == T
        assert yaws.shape[0] == T

        max_start = T - k - required_future + 1
        if max_start <= 0:
            continue

        for start in range(0, max_start, args.stride):
            past_end = start + k - 1
            future_t = past_end + h

            past_summary = compute_past_summary(poses, yaws, start, k, fps)
            future_motion = compute_step_motion(poses, yaws, future_t, fps)
            rollout_state = compose_future_rollout(
                poses=poses,
                yaws=yaws,
                start_step=past_end,
                rollout_steps=rollout_steps,
                fps=fps,
            )
            control_targets = build_control_targets(poses, yaws, past_end, rollout_steps, fps)
            endpoint_targets = build_endpoint_targets(rollout_state)
            geometry_targets = build_geometry_targets(rollout_state)
            rollout_targets = select_rollout_keypoints(
                rollout_state=rollout_state,
                rollout_steps=rollout_steps,
                rollout_keypoints=args.rollout_keypoints,
            )

            sample = {
                "index": int(len(samples)),
                "sample_format": "lightweight_v2",
                "seq_id": int(seq_id),
                "window_start": int(start),
                "window_end": int(past_end),
                "future_t": int(future_t),
                "condition_frames": int(k),
                "future_horizon": int(h),
                "rollout_steps": int(rollout_steps),
                "rollout_keypoints": int(args.rollout_keypoints),

                # past-summary targets
                **past_summary,

                # future-step targets
                "future_delta_x": future_motion["delta_x"],
                "future_delta_y": future_motion["delta_y"],
                "future_delta_yaw": future_motion["delta_yaw"],
                "future_speed": future_motion["speed"],
                "future_yaw_rate": future_motion["yaw_rate"],
                "future_turn_class": future_turn_class(future_motion["delta_yaw"]),

                # planning/action probe targets
                **control_targets,
                **endpoint_targets,
                **geometry_targets,
                **rollout_targets,
            }

            if args.include_tensors:
                sample.update({
                    "past_imgs": imgs[start: start + k].clone(),
                    "past_poses": poses[start: start + k].clone(),
                    "past_yaws": yaws[start: start + k].clone(),
                    "future_img": imgs[future_t].clone(),
                    "future_pose": poses[future_t].clone(),
                    "future_yaw": yaws[future_t].clone(),
                })
            samples.append(sample)

    save_path = os.path.join(args.save_root, "physics_samples.pt")
    out = {
        "sample_format": "lightweight_v2" if not args.include_tensors else "tensor_v1",
        "condition_frames": int(k),
        "future_horizon": int(h),
        "rollout_steps": int(rollout_steps),
        "rollout_keypoints": int(args.rollout_keypoints),
        "downsample_fps": int(fps),
        "num_sequences": int(len(dataset)),
        "num_samples": int(len(samples)),
        "legacy_label_keys": LEGACY_LABEL_KEYS,
        "planning_head_families": head_families,
        "samples": samples,
    }
    torch.save(out, save_path)

    print(f"saved to {save_path}")
    print(f"num samples = {len(samples)}")
    print(f"sample_format = {out['sample_format']}")

    for key in [
        "past_avg_speed", "past_final_speed", "past_avg_yaw_rate",
        "past_heading_change", "future_speed", "future_yaw_rate",
        "future_delta_yaw", "control_delta_v", "control_cum_delta_yaw",
        "endpoint_forward_progress", "endpoint_lateral_disp",
        "geometry_mean_curvature", "geometry_integrated_curvature",
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
    parser.add_argument(
        "--rollout_steps",
        type=int,
        default=5,
        help="How many future relative-motion steps to use for planning/action labels.",
    )
    parser.add_argument(
        "--rollout_keypoints",
        type=int,
        default=5,
        help="How many evenly spaced keypoints to keep for the rollout head.",
    )
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--downsample_fps", type=int, default=5)
    parser.add_argument("--downsample_size", type=int, default=16)
    parser.add_argument("--h", type=int, default=256)
    parser.add_argument("--w", type=int, default=512)
    parser.add_argument(
        "--include_tensors",
        action="store_true",
        help="Also save full image/pose/yaw tensors in each sample. This is very large and mainly for backward compatibility.",
    )
    args = parser.parse_args()
    main(args)
