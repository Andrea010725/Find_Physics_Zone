import argparse
import glob
import os

import torch


def load_shards(pattern):
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No shard files matched: {pattern}")
    print("matched shard files:")
    for path in paths:
        print(" ", path)
    return paths


def concat_meta(values):
    if not values:
        return values
    first = values[0]
    if isinstance(first, torch.Tensor):
        return torch.cat(values, dim=0)
    merged = []
    for x in values:
        merged.extend(x)
    return merged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    shard_paths = load_shards(args.pattern)
    shard_data = [torch.load(path, map_location="cpu") for path in shard_paths]

    task = shard_data[0]["task"]
    feature_mode = shard_data[0]["feature_mode"]
    feature_dtype = shard_data[0]["feature_dtype"]
    probe_sample_format = shard_data[0]["probe_sample_format"]
    stage_semantics = shard_data[0]["stage_semantics"]

    layer_names = sorted(shard_data[0]["layer_features"].keys())
    for shard in shard_data[1:]:
        if sorted(shard["layer_features"].keys()) != layer_names:
            raise ValueError("Layer names differ across shards")

    merged_layer_features = {}
    for layer_name in layer_names:
        merged_layer_features[layer_name] = torch.cat(
            [shard["layer_features"][layer_name] for shard in shard_data],
            dim=0,
        )
        print(f"merged {layer_name}: {tuple(merged_layer_features[layer_name].shape)}")

    label_names = sorted(shard_data[0]["labels"].keys())
    merged_labels = {}
    for label_name in label_names:
        merged_labels[label_name] = torch.cat(
            [shard["labels"][label_name] for shard in shard_data],
            dim=0,
        )
        print(f"merged label {label_name}: {tuple(merged_labels[label_name].shape)}")

    meta_names = sorted(shard_data[0]["meta"].keys())
    merged_meta = {}
    for meta_name in meta_names:
        merged_meta[meta_name] = concat_meta([shard["meta"][meta_name] for shard in shard_data])
        if isinstance(merged_meta[meta_name], torch.Tensor):
            print(f"merged meta {meta_name}: {tuple(merged_meta[meta_name].shape)}")
        else:
            print(f"merged meta {meta_name}: list(len={len(merged_meta[meta_name])})")

    out = {
        "task": task,
        "feature_mode": feature_mode,
        "feature_dtype": feature_dtype,
        "probe_sample_format": probe_sample_format,
        "stage_semantics": stage_semantics,
        "layer_features": merged_layer_features,
        "labels": merged_labels,
        "meta": merged_meta,
        "merged_from": shard_paths,
    }

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    torch.save(out, args.output_path)
    print(f"saved merged feature file to: {args.output_path}")


if __name__ == "__main__":
    main()
