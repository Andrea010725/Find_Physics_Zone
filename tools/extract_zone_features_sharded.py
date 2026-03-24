import argparse
import os
import subprocess
import sys

import torch


root_path = os.path.abspath(__file__)
root_path = "/".join(root_path.split("/")[:-2])
workspace_root = os.path.dirname(root_path)


def resolve_repo_path(*parts):
    local_path = os.path.join(root_path, *parts)
    workspace_path = os.path.join(workspace_root, *parts)
    if os.path.exists(local_path):
        return local_path
    return workspace_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, choices=["physics", "coherence"], required=True)
    parser.add_argument("--samples_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--shard_size", type=int, default=200)
    parser.add_argument("--shard_dir", type=str, default=None)
    parser.add_argument("--keep_shards", action="store_true")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--pool_mode", type=str, choices=["all_mean", "tokenwise"], default="all_mean")
    parser.add_argument("--feature_dtype", type=str, choices=["float32", "float16"], default="float32")
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--json_root", type=str, default=None)
    parser.add_argument("--condition_frames", type=int, default=None)
    parser.add_argument("--downsample_fps", type=int, default=5)
    parser.add_argument("--downsample_size", type=int, default=None)
    parser.add_argument("--max_cached_sequences", type=int, default=2)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--max_samples", type=int, default=None)
    return parser.parse_args()


def append_optional_arg(cmd, name, value):
    if value is None:
        return
    cmd.extend([name, str(value)])


def load_num_samples(samples_path):
    payload = torch.load(samples_path, map_location="cpu")
    if isinstance(payload, dict) and "samples" in payload:
        return len(payload["samples"])
    return len(payload)


def build_shard_path(shard_dir, start, end):
    return os.path.join(shard_dir, f"features_{start:06d}_{end:06d}.pt")


def main():
    args = parse_args()
    extract_script = resolve_repo_path("tools", "extract_zone_features.py")
    merge_script = resolve_repo_path("tools", "merge_feature_shards.py")

    if args.shard_size < 1:
        raise ValueError("--shard_size must be >= 1")

    total_samples = load_num_samples(args.samples_path)
    if args.max_samples is not None:
        total_samples = min(total_samples, int(args.max_samples))

    output_dir = os.path.dirname(os.path.abspath(args.output_path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    if args.shard_dir is None:
        output_stem = os.path.splitext(os.path.basename(args.output_path))[0]
        shard_dir = os.path.join(output_dir, f"{output_stem}_shards")
    else:
        shard_dir = args.shard_dir
    os.makedirs(shard_dir, exist_ok=True)

    shard_paths = []
    for start in range(0, total_samples, args.shard_size):
        end = min(start + args.shard_size, total_samples)
        shard_path = build_shard_path(shard_dir, start, end)
        shard_paths.append(shard_path)

        cmd = [
            sys.executable,
            extract_script,
            "--task", args.task,
            "--samples_path", args.samples_path,
            "--output_path", shard_path,
            "--start_index", str(start),
            "--end_index", str(end),
            "--device", args.device,
            "--pool_mode", args.pool_mode,
            "--feature_dtype", args.feature_dtype,
            "--downsample_fps", str(args.downsample_fps),
            "--max_cached_sequences", str(args.max_cached_sequences),
            "--log_every", str(args.log_every),
        ]
        append_optional_arg(cmd, "--data_root", args.data_root)
        append_optional_arg(cmd, "--json_root", args.json_root)
        append_optional_arg(cmd, "--condition_frames", args.condition_frames)
        append_optional_arg(cmd, "--downsample_size", args.downsample_size)

        print(
            f"\n===== shard {len(shard_paths)} =====\n"
            f"range = [{start}, {end})\n"
            f"output = {shard_path}\n"
            f"cmd = {' '.join(cmd)}\n"
        )
        subprocess.run(cmd, check=True)

    merge_cmd = [
        sys.executable,
        merge_script,
        "--pattern", os.path.join(shard_dir, "features_*.pt"),
        "--output_path", args.output_path,
    ]
    print(f"\nMerging shards with: {' '.join(merge_cmd)}\n")
    subprocess.run(merge_cmd, check=True)

    if not args.keep_shards:
        for path in shard_paths:
            os.remove(path)
        if not os.listdir(shard_dir):
            os.rmdir(shard_dir)
        print(f"Removed shard files under: {shard_dir}")

    print(f"Saved merged feature file to: {args.output_path}")


if __name__ == "__main__":
    main()
