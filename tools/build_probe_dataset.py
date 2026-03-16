import os
import argparse
import random
from collections import defaultdict

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


def load_physics_samples(save_root):
    physics_path = os.path.join(save_root, "physics_samples.pt")
    print(f"loading physics samples from: {physics_path}")
    payload = torch.load(physics_path, map_location="cpu")

    if isinstance(payload, dict) and "samples" in payload:
        samples = payload["samples"]
        sample_format = payload.get("sample_format", "unknown")
    else:
        samples = payload
        sample_format = "legacy_tensor_list"

    print(f"physics sample_format = {sample_format}")
    print(f"num physics samples = {len(samples)}")
    return samples


def sample_random_negative(i, num_samples, rng):
    j = rng.randrange(num_samples - 1)
    if j >= i:
        j += 1
    return j


def precompute_arrays(samples):
    return {
        "past_heading_change": np.asarray([s["past_heading_change"] for s in samples], dtype=np.float32),
        "past_avg_speed": np.asarray([s["past_avg_speed"] for s in samples], dtype=np.float32),
        "future_delta_yaw": np.asarray([s["future_delta_yaw"] for s in samples], dtype=np.float32),
    }


def sample_hard_negative(i, arrays, rng, heading_thresh=5.0, speed_thresh=0.50):
    heading = arrays["past_heading_change"]
    speed = arrays["past_avg_speed"]
    future_yaw = arrays["future_delta_yaw"]

    mask = (
        (np.abs(heading - heading[i]) < heading_thresh)
        & (np.abs(speed - speed[i]) < speed_thresh)
        & (np.abs(future_yaw - future_yaw[i]) > heading_thresh)
    )
    mask[i] = False
    candidates = np.flatnonzero(mask)

    if candidates.size == 0:
        return sample_random_negative(i, len(heading), rng)
    return int(candidates[rng.randrange(candidates.size)])


def build_one_pair(base_index, cond_index, samples, label, neg_type):
    base_sample = samples[base_index]
    cond_sample = samples[cond_index]

    pair = {
        "index": int(base_index * 3 + (1 if neg_type == "positive" else 2 if neg_type == "random_mismatch" else 3)),
        "sample_format": "lightweight_v2",
        "base_index": int(base_index),
        "cond_index": int(cond_index),
        "seq_id": int(base_sample["seq_id"]),
        "window_start": int(base_sample["window_start"]),
        "window_end": int(base_sample["window_end"]),
        "future_t": int(base_sample["future_t"]),
        "cond_seq_id": int(cond_sample["seq_id"]),
        "cond_window_start": int(cond_sample["window_start"]),
        "cond_window_end": int(cond_sample["window_end"]),
        "cond_future_t": int(cond_sample["future_t"]),
        "coherence_label": int(label),
        "negative_type": neg_type,
        "source_index": int(base_sample["seq_id"]),
        "cond_source_index": int(cond_sample["seq_id"]),
        "source_window_start": int(base_sample["window_start"]),
        "cond_window_start_ref": int(cond_sample["window_start"]),
        "abs_future_delta_yaw_gap": float(abs(cond_sample["future_delta_yaw"] - base_sample["future_delta_yaw"])),
        "abs_past_heading_gap": float(abs(cond_sample["past_heading_change"] - base_sample["past_heading_change"])),
        "abs_past_speed_gap": float(abs(cond_sample["past_avg_speed"] - base_sample["past_avg_speed"])),
    }

    # Backward compatibility for older extraction code paths.
    if "future_pose" in cond_sample:
        pair["future_pose_cond"] = cond_sample["future_pose"].clone()
    if "future_yaw" in cond_sample:
        pair["future_yaw_cond"] = cond_sample["future_yaw"].clone()
    if "past_imgs" in base_sample:
        pair["past_imgs"] = base_sample["past_imgs"].clone()
        pair["past_poses"] = base_sample["past_poses"].clone()
        pair["past_yaws"] = base_sample["past_yaws"].clone()
        pair["future_img"] = base_sample["future_img"].clone()

    return pair


def main(args):
    rng = random.Random(args.seed)
    samples = load_physics_samples(args.save_root)
    arrays = precompute_arrays(samples)

    out = []
    num_samples = len(samples)

    for i in tqdm(range(num_samples), desc="building coherence samples"):
        out.append(build_one_pair(i, i, samples, 1, "positive"))

        j_rand = sample_random_negative(i, num_samples, rng)
        out.append(build_one_pair(i, j_rand, samples, 0, "random_mismatch"))

        j_hard = sample_hard_negative(
            i,
            arrays,
            rng,
            heading_thresh=args.heading_thresh,
            speed_thresh=args.speed_thresh,
        )
        out.append(build_one_pair(i, j_hard, samples, 0, "hard_mismatch"))

    save_path = os.path.join(args.save_root, "coherence_samples.pt")
    payload = {
        "sample_format": "lightweight_v2",
        "source": "physics_samples.pt",
        "num_pairs": int(len(out)),
        "num_base_samples": int(num_samples),
        "samples": out,
    }
    torch.save(payload, save_path)

    print(f"saved to {save_path}")
    print(f"num coherence samples = {len(out)}")
    print(f"sample_format = {payload['sample_format']}")

    type_count = defaultdict(int)
    labels = []
    for x in out:
        type_count[x["negative_type"]] += 1
        labels.append(x["coherence_label"])

    print("label stats:")
    print(f"positive = {sum(labels)}")
    print(f"negative = {len(labels) - sum(labels)}")

    print("type stats:")
    for k, v in sorted(type_count.items()):
        print(f"{k}: {v}")

    for key in ["abs_future_delta_yaw_gap", "abs_past_heading_gap", "abs_past_speed_gap"]:
        vals = [x[key] for x in out if x["coherence_label"] == 0]
        print(
            f"{key:25s} mean={np.mean(vals):.6f}, std={np.std(vals):.6f}, "
            f"min={np.min(vals):.6f}, max={np.max(vals):.6f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_root", type=str, default=data_root)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--heading_thresh", type=float, default=5.0)
    parser.add_argument("--speed_thresh", type=float, default=0.50)
    args = parser.parse_args()
    main(args)
