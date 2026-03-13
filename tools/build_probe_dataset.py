import os
import argparse
import torch
import random
import numpy as np
from collections import defaultdict


def sample_random_negative(i, samples, rng):
    j = rng.randrange(len(samples) - 1)
    if j >= i:
        j += 1
    return j


def sample_hard_negative(i, samples, rng, heading_thresh=0.10, speed_thresh=0.50):
    target = samples[i]
    candidates = []

    for j in range(len(samples)):
        if j == i:
            continue

        cond1 = abs(samples[j]["past_heading_change"] - target["past_heading_change"]) < heading_thresh
        cond2 = abs(samples[j]["past_avg_speed"] - target["past_avg_speed"]) < speed_thresh

        # 过去相似，但未来转向不同
        cond3 = abs(samples[j]["future_delta_yaw"] - target["future_delta_yaw"]) > heading_thresh

        if cond1 and cond2 and cond3:
            candidates.append(j)

    if len(candidates) == 0:
        return sample_random_negative(i, samples, rng)
    return rng.choice(candidates)


def build_one_pair(base_sample, cond_sample, label, neg_type):
    return {
        "seq_id": base_sample["seq_id"],
        "window_start": base_sample["window_start"],
        "window_end": base_sample["window_end"],
        "future_t": base_sample["future_t"],

        "past_imgs": base_sample["past_imgs"].clone(),
        "past_poses": base_sample["past_poses"].clone(),
        "past_yaws": base_sample["past_yaws"].clone(),

        "future_img": base_sample["future_img"].clone(),
        "future_pose_cond": cond_sample["future_pose"].clone(),
        "future_yaw_cond": cond_sample["future_yaw"].clone(),

        "coherence_label": int(label),
        "negative_type": neg_type,

        "source_index": int(base_sample["seq_id"]),
        "cond_source_index": int(cond_sample["seq_id"]),
        "source_window_start": int(base_sample["window_start"]),
        "cond_window_start": int(cond_sample["window_start"]),

        # 方便后面分析负样本难度
        "abs_future_delta_yaw_gap": float(abs(cond_sample["future_delta_yaw"] - base_sample["future_delta_yaw"])),
        "abs_past_heading_gap": float(abs(cond_sample["past_heading_change"] - base_sample["past_heading_change"])),
        "abs_past_speed_gap": float(abs(cond_sample["past_avg_speed"] - base_sample["past_avg_speed"])),
    }


def main(args):
    rng = random.Random(args.seed)
    physics_path = os.path.join(args.save_root, "physics_samples.pt")
    samples = torch.load(physics_path)

    out = []

    for i in range(len(samples)):
        base = samples[i]

        # 正样本
        out.append(build_one_pair(base, base, 1, "positive"))

        # random negative
        j_rand = sample_random_negative(i, samples, rng)
        out.append(build_one_pair(base, samples[j_rand], 0, "random_mismatch"))

        # hard negative
        j_hard = sample_hard_negative(i, samples, rng)
        out.append(build_one_pair(base, samples[j_hard], 0, "hard_mismatch"))

    save_path = os.path.join(args.save_root, "coherence_samples.pt")
    torch.save(out, save_path)

    print(f"saved to {save_path}")
    print(f"num coherence samples = {len(out)}")

    type_count = defaultdict(int)
    labels = []
    for x in out:
        type_count[x["negative_type"]] += 1
        labels.append(x["coherence_label"])

    print("label stats:")
    print(f"positive = {sum(labels)}")
    print(f"negative = {len(labels) - sum(labels)}")

    print("type stats:")
    for k, v in type_count.items():
        print(f"{k}: {v}")

    for key in ["abs_future_delta_yaw_gap", "abs_past_heading_gap", "abs_past_speed_gap"]:
        vals = [x[key] for x in out if x["coherence_label"] == 0]
        print(f"{key:25s} mean={np.mean(vals):.6f}, std={np.std(vals):.6f}, "
              f"min={np.min(vals):.6f}, max={np.max(vals):.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_root", type=str, default="/home/zhiwen/DrivingWorld/data")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    main(args)
