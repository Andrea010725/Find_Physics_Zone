import argparse
import os
import re
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import torch


DEFAULT_PT_PATH = os.path.join("data", "physics_probe_results.pt")
DEFAULT_TXT_PATH = os.path.join("data", "physics_probe_results.txt")
DEFAULT_OUTPUT_DIR = os.path.join("data", "physics_probe_viz")

AGS_TARGETS = [
    "future_speed",
    "future_yaw_rate",
    "future_delta_yaw",
    "future_forward_progress",
    "future_lateral_offset",
    "future_speed_delta",
    "future_acc",
]

PLOT_TARGETS = [
    "future_speed",
    "future_yaw_rate",
    "future_delta_yaw",
    "future_forward_progress",
    "future_lateral_offset",
    "future_speed_delta",
    "future_acc",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt_path", type=str, default=DEFAULT_PT_PATH)
    parser.add_argument("--txt_path", type=str, default=DEFAULT_TXT_PATH)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--targets", nargs="*", default=None)
    parser.add_argument("--show_baselines", action="store_true")
    return parser.parse_args()


def layer_order_key(name):
    if name == "tokenizer_last":
        return (0, 0)
    if name.startswith("time_space_"):
        return (1, int(name.split("_")[-1]))
    if name == "next_state_hidden":
        return (2, 0)
    if name.startswith("ar_"):
        return (3, int(name.split("_")[-1]))
    if name == "raw_pose_yaw_past":
        return (4, 0)
    if name == "raw_pose_yaw_full":
        return (4, 1)
    return (5, name)


def get_layer_stage(layer_name):
    """Return stage name for a layer."""
    if layer_name == "tokenizer_last":
        return "tokenizer"
    if layer_name.startswith("time_space_"):
        return "spatiotemporal"
    if layer_name == "next_state_hidden":
        return "next_state"
    if layer_name.startswith("ar_"):
        return "autoregressive"
    if layer_name.startswith("raw_pose_yaw"):
        return "raw_baseline"
    return "unknown"


def get_stage_boundaries(layers):
    """Return indices where stage changes occur."""
    boundaries = []
    prev_stage = None
    for idx, layer in enumerate(layers):
        stage = get_layer_stage(layer)
        if stage != prev_stage and prev_stage is not None:
            boundaries.append(idx)
        prev_stage = stage
    return boundaries


def get_stage_colors():
    """Return colorblind-friendly color map for stages."""
    return {
        "tokenizer": "#E69F00",
        "spatiotemporal": "#56B4E9",
        "next_state": "#009E73",
        "autoregressive": "#F0E442",
        "raw_baseline": "#999999",
        "unknown": "#CC79A7",
    }


def get_stage_labels(layers):
    """Return stage labels and their x positions for annotation."""
    stage_ranges = {}
    for idx, layer in enumerate(layers):
        stage = get_layer_stage(layer)
        if stage not in stage_ranges:
            stage_ranges[stage] = [idx, idx]
        else:
            stage_ranges[stage][1] = idx

    labels = []
    for stage, (start, end) in stage_ranges.items():
        mid = (start + end) / 2
        labels.append((mid, stage))
    return labels


def summary_metric(metric_dict):
    if isinstance(metric_dict, dict) and "best_token" in metric_dict:
        return metric_dict["best_token"]
    return metric_dict


def load_from_pt(pt_path):
    payload = torch.load(pt_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError("PT result file is not a dict payload")

    if "per_target" in payload:
        per_target = payload["per_target"]
        target_groups = payload.get("target_groups", {"ags": AGS_TARGETS})
        group_summaries = payload.get("group_summaries", {})
    else:
        per_target = payload
        target_groups = {"ags": AGS_TARGETS}
        group_summaries = {}

    layers = set()
    for target_name, layer_dict in per_target.items():
        for layer_name in layer_dict:
            layers.add(layer_name)
    ordered_layers = sorted(layers, key=layer_order_key)

    normalized = {}
    for target_name, layer_dict in per_target.items():
        normalized[target_name] = {}
        for layer_name, metric_dict in layer_dict.items():
            normalized[target_name][layer_name] = summary_metric(metric_dict)

    return {
        "source": "pt",
        "per_target": normalized,
        "target_groups": target_groups,
        "group_summaries": group_summaries,
        "layers": ordered_layers,
    }


SECTION_TARGET_RE = re.compile(r"^===== summary sorted by r2: (.+) =====$")
SECTION_GROUP_RE = re.compile(r"^===== ([A-Z]+) group summary sorted by mean r2 =====$")
METRIC_RE = re.compile(
    r"^(?P<layer>[^|]+?)\s*\|\s*"
    r"(?:(?:best_token=.*?\|\s*))?"
    r"mse=(?P<mse>-?\d+\.\d+)\s*\|\s*"
    r"mae=(?P<mae>-?\d+\.\d+)\s*\|\s*"
    r"nmae=(?P<nmae>-?\d+\.\d+)\s*\|\s*"
    r"r2=(?P<r2>-?\d+\.\d+)\s*\|\s*"
    r"pearson=(?P<pearson>-?\d+\.\d+)"
)
GROUP_METRIC_RE = re.compile(
    r"^(?P<layer>[^|]+?)\s*\|\s*"
    r"mean_mse=(?P<mean_mse>-?\d+\.\d+)\s*\|\s*"
    r"mean_mae=(?P<mean_mae>-?\d+\.\d+)\s*\|\s*"
    r"mean_nmae=(?P<mean_nmae>-?\d+\.\d+)\s*\|\s*"
    r"mean_r2=(?P<mean_r2>-?\d+\.\d+)\s*\|\s*"
    r"mean_pearson=(?P<mean_pearson>-?\d+\.\d+)"
)


def load_from_txt(txt_path):
    with open(txt_path, "r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f]

    per_target = {}
    group_summaries = {}
    current_target = None
    current_group = None

    for line in lines:
        line = line.strip()
        if not line:
            continue

        target_match = SECTION_TARGET_RE.match(line)
        if target_match:
            current_target = target_match.group(1)
            current_group = None
            per_target[current_target] = {}
            continue

        group_match = SECTION_GROUP_RE.match(line)
        if group_match:
            current_group = group_match.group(1).lower()
            current_target = None
            group_summaries[current_group] = {}
            continue

        if line.startswith("=====") or line.startswith("==="):
            continue

        if current_target is not None:
            metric_match = METRIC_RE.match(line)
            if metric_match:
                layer = metric_match.group("layer").strip()
                per_target[current_target][layer] = {
                    "mse": float(metric_match.group("mse")),
                    "mae": float(metric_match.group("mae")),
                    "normalized_mae": float(metric_match.group("nmae")),
                    "r2": float(metric_match.group("r2")),
                    "pearson": float(metric_match.group("pearson")),
                }
            continue

        if current_group is not None:
            group_metric_match = GROUP_METRIC_RE.match(line)
            if group_metric_match:
                layer = group_metric_match.group("layer").strip()
                group_summaries[current_group][layer] = {
                    "mean_mse": float(group_metric_match.group("mean_mse")),
                    "mean_mae": float(group_metric_match.group("mean_mae")),
                    "mean_normalized_mae": float(group_metric_match.group("mean_nmae")),
                    "mean_r2": float(group_metric_match.group("mean_r2")),
                    "mean_pearson": float(group_metric_match.group("mean_pearson")),
                }

    layers = set()
    for layer_dict in per_target.values():
        layers.update(layer_dict.keys())
    ordered_layers = sorted(layers, key=layer_order_key)

    return {
        "source": "txt",
        "per_target": per_target,
        "target_groups": {"ags": AGS_TARGETS},
        "group_summaries": group_summaries,
        "layers": ordered_layers,
    }


def load_results(pt_path, txt_path):
    if os.path.exists(pt_path):
        try:
            return load_from_pt(pt_path)
        except Exception as exc:
            print(f"[warn] failed to load pt file: {exc}")
    if os.path.exists(txt_path):
        return load_from_txt(txt_path)
    raise FileNotFoundError("Could not load either PT or TXT result files")


def compute_group_stats(per_target, group_targets, layers):
    available_targets = [target for target in group_targets if target in per_target]
    mean_r2 = []
    std_r2 = []
    mean_nmae = []

    for layer in layers:
        layer_metrics = [per_target[target][layer] for target in available_targets if layer in per_target[target]]
        r2s = [m["r2"] for m in layer_metrics]
        nmaes = [m["normalized_mae"] for m in layer_metrics]
        mean_r2.append(float(np.mean(r2s)) if r2s else np.nan)
        std_r2.append(float(np.std(r2s)) if r2s else np.nan)
        mean_nmae.append(float(np.mean(nmaes)) if nmaes else np.nan)

    return {
        "targets": available_targets,
        "mean_r2": np.array(mean_r2, dtype=np.float32),
        "std_r2": np.array(std_r2, dtype=np.float32),
        "mean_nmae": np.array(mean_nmae, dtype=np.float32),
    }


def plot_layerwise_r2(output_dir, layers, per_target, group_stats, targets):
    x = np.arange(len(layers))
    plt.figure(figsize=(16, 7))
    for target in targets:
        if target not in per_target:
            continue
        values = [per_target[target][layer]["r2"] for layer in layers]
        plt.plot(x, values, marker="o", linewidth=2, label=target)

    plt.plot(x, group_stats["mean_r2"], marker="o", linewidth=3, linestyle="--", color="black", label="AGS group mean R²")
    plt.xticks(x, layers, rotation=45, ha="right")
    plt.ylabel("R²")
    plt.title("Layerwise R² curves for AGS targets")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "layerwise_r2_curves.png"), dpi=200)
    plt.close()


def plot_per_target_r2(output_dir, layers, per_target, targets):
    per_target_dir = os.path.join(output_dir, "per_target_r2")
    os.makedirs(per_target_dir, exist_ok=True)
    x = np.arange(len(layers))

    for target in targets:
        if target not in per_target:
            continue
        values = [per_target[target][layer]["r2"] for layer in layers]
        best_idx = int(np.argmax(values))
        best_layer = layers[best_idx]
        best_r2 = values[best_idx]

        plt.figure(figsize=(14, 6))
        plt.plot(x, values, marker="o", linewidth=2)
        plt.axhline(0.0, color="gray", linestyle="--", linewidth=1)
        plt.scatter([best_idx], [best_r2], color="red", s=80, zorder=5)
        plt.annotate(
            f"best: {best_layer}\nR²={best_r2:.3f}",
            (best_idx, best_r2),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
        )
        plt.xticks(x, layers, rotation=45, ha="right")
        plt.ylabel("R²")
        plt.title(f"Layerwise R² for {target}")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(per_target_dir, f"{target}_r2.png"), dpi=200)
        plt.close()


def plot_ags_heatmap(output_dir, layers, per_target, ags_targets):
    main_dir = os.path.join(output_dir, "main_figures")
    os.makedirs(main_dir, exist_ok=True)

    targets_with_best = []
    for target in ags_targets:
        if target not in per_target:
            continue
        best_layer, best_metric = max(per_target[target].items(), key=lambda item: item[1]["r2"])
        best_stage = get_layer_stage(best_layer)
        targets_with_best.append((target, best_layer, best_stage, best_metric["r2"]))

    stage_order = {"tokenizer": 0, "spatiotemporal": 1, "next_state": 2, "autoregressive": 3, "raw_baseline": 4}
    targets_with_best.sort(key=lambda x: (stage_order.get(x[2], 5), -x[3]))
    sorted_targets = [t[0] for t in targets_with_best]

    matrix = []
    for target in sorted_targets:
        row = [per_target[target][layer]["r2"] for layer in layers]
        matrix.append(row)
    matrix = np.array(matrix)

    fig, ax = plt.subplots(figsize=(18, 8))
    im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-1.0, vmax=1.0)

    ax.set_xticks(np.arange(len(layers)))
    ax.set_yticks(np.arange(len(sorted_targets)))
    ax.set_xticklabels(layers, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels([f"{t} (R²={targets_with_best[i][3]:.2f})" for i, t in enumerate(sorted_targets)], fontsize=10)

    boundaries = get_stage_boundaries(layers)
    for b in boundaries:
        ax.axvline(b - 0.5, color="black", linewidth=2)

    stage_labels = get_stage_labels(layers)
    for mid, stage in stage_labels:
        stage_name_map = {
            "tokenizer": "Tokenizer",
            "spatiotemporal": "Spatiotemporal",
            "next_state": "Next State",
            "autoregressive": "Autoregressive",
            "raw_baseline": "Raw Motion",
        }
        ax.text(mid, -1.5, stage_name_map.get(stage, stage), ha="center", va="top", fontsize=11, fontweight="bold")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("R²", rotation=270, labelpad=20, fontsize=12)

    ax.set_title("AGS Probe Heatmap: Layer-wise R² for Action-Geometric State Variables", fontsize=14, pad=20)
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("AGS Target", fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(main_dir, "ags_heatmap.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(main_dir, "ags_heatmap.pdf"), bbox_inches="tight")
    plt.close()


def plot_best_layer_and_baseline(output_dir, layers, per_target, ags_targets):
    main_dir = os.path.join(output_dir, "main_figures")
    os.makedirs(main_dir, exist_ok=True)

    target_data = []
    for target in ags_targets:
        if target not in per_target:
            continue
        best_layer, best_metric = max(per_target[target].items(), key=lambda item: item[1]["r2"])
        best_r2 = best_metric["r2"]
        best_stage = get_layer_stage(best_layer)

        raw_past_r2 = per_target[target].get("raw_pose_yaw_past", {}).get("r2", np.nan)
        raw_full_r2 = per_target[target].get("raw_pose_yaw_full", {}).get("r2", np.nan)

        target_data.append({
            "target": target,
            "best_layer": best_layer,
            "best_r2": best_r2,
            "best_stage": best_stage,
            "raw_past_r2": raw_past_r2,
            "raw_full_r2": raw_full_r2,
        })

    target_data.sort(key=lambda x: x["best_r2"], reverse=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    y_pos = np.arange(len(target_data))
    stage_colors = get_stage_colors()
    colors = [stage_colors.get(d["best_stage"], "#999999") for d in target_data]

    ax1.barh(y_pos, [d["best_r2"] for d in target_data], color=colors, alpha=0.8)
    ax1.axvline(0, color="gray", linestyle="--", linewidth=1)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([d["target"] for d in target_data], fontsize=10)
    ax1.set_xlabel("R² at Best Layer", fontsize=12)
    ax1.set_title("Best Layer Analysis", fontsize=13, fontweight="bold")
    ax1.grid(axis="x", alpha=0.3)

    for i, d in enumerate(target_data):
        ax1.text(d["best_r2"] + 0.02, i, f"{d['best_layer']}", va="center", fontsize=9)

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=stage_colors[stage], label=stage.capitalize())
                      for stage in ["tokenizer", "spatiotemporal", "next_state", "autoregressive", "raw_baseline"]]
    ax1.legend(handles=legend_elements, loc="lower right", fontsize=9)

    x_pos = np.arange(len(target_data))
    width = 0.25

    ax2.bar(x_pos - width, [d["best_r2"] for d in target_data], width, label="Best Hidden Layer", color="#56B4E9", alpha=0.9)
    ax2.bar(x_pos, [d["raw_past_r2"] for d in target_data], width, label="raw_pose_yaw_past", color="#999999", alpha=0.7)
    ax2.bar(x_pos + width, [d["raw_full_r2"] for d in target_data], width, label="raw_pose_yaw_full", color="#CCCCCC", alpha=0.7)

    ax2.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([d["target"] for d in target_data], rotation=45, ha="right", fontsize=9)
    ax2.set_ylabel("R²", fontsize=12)
    ax2.set_title("Baseline Comparison", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(main_dir, "best_layer_and_baseline.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(main_dir, "best_layer_and_baseline.pdf"), bbox_inches="tight")
    plt.close()


def plot_ags_stability(output_dir, layers, group_stats):
    x = np.arange(len(layers))
    plt.figure(figsize=(16, 7))
    plt.errorbar(x, group_stats["mean_r2"], yerr=group_stats["std_r2"], fmt="-o", capsize=4, linewidth=2)
    plt.xticks(x, layers, rotation=45, ha="right")
    plt.ylabel("Mean R² ± std")
    plt.title("AGS group stability across layers")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ags_group_stability.png"), dpi=200)
    plt.close()


def plot_best_layer_distribution(output_dir, per_target, targets):
    best_layers = []
    for target in targets:
        if target not in per_target:
            continue
        best_layer = max(per_target[target].items(), key=lambda item: item[1]["r2"])[0]
        best_layers.append(best_layer)

    counts = Counter(best_layers)
    ordered_layers = sorted(counts.keys(), key=layer_order_key)
    values = [counts[layer] for layer in ordered_layers]

    plt.figure(figsize=(12, 6))
    plt.bar(ordered_layers, values)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("# AGS targets with this best layer")
    plt.title("Best-layer distribution for AGS targets")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ags_best_layer_distribution.png"), dpi=200)
    plt.close()

    return counts


def plot_ags_mean_ranking(output_dir, layers, group_stats):
    pairs = list(zip(layers, group_stats["mean_r2"], group_stats["mean_nmae"]))
    pairs = sorted(pairs, key=lambda item: item[1], reverse=True)
    layer_names = [p[0] for p in pairs]
    mean_r2_vals = [p[1] for p in pairs]
    mean_nmae_vals = [p[2] for p in pairs]

    fig, ax1 = plt.subplots(figsize=(14, 7))
    ax1.bar(layer_names, mean_r2_vals, alpha=0.8, label="mean R²")
    ax1.set_ylabel("Mean R²")
    ax1.set_xticklabels(layer_names, rotation=45, ha="right")

    ax2 = ax1.twinx()
    ax2.plot(layer_names, mean_nmae_vals, color="red", marker="o", linewidth=2, label="mean nMAE")
    ax2.set_ylabel("Mean normalized MAE")

    ax1.set_title("AGS mean metric ranking by layer")
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, "ags_mean_metric_ranking.png"), dpi=200)
    plt.close(fig)


def assess_goals(output_dir, per_target, group_stats, best_layer_counts, ags_targets):
    ags_best_layers = {}
    for target in ags_targets:
        if target not in per_target:
            continue
        ags_best_layers[target] = max(per_target[target].items(), key=lambda item: item[1]["r2"])

    supported_layers = [layer for _, (layer, _) in ags_best_layers.items() if layer == "next_state_hidden" or layer.startswith("ar_")]
    supported_ratio = len(supported_layers) / max(len(ags_best_layers), 1)

    layer_names = list(best_layer_counts.keys())
    sorted_group_layers = sorted(
        zip(layer_names, [group_stats["mean_r2"][sorted(set(layer_names) | set())] if False else 0 for _ in layer_names]),
        key=lambda item: item[1],
        reverse=True,
    )

    layers = list(per_target[next(iter(per_target))].keys())
    ordered_layers = sorted(layers, key=layer_order_key)
    ranked_group_layers = sorted(
        zip(ordered_layers, group_stats["mean_r2"], group_stats["std_r2"], group_stats["mean_nmae"]),
        key=lambda item: item[1],
        reverse=True,
    )
    top3_group = ranked_group_layers[:3]
    top3_supported = sum(1 for layer, _, _, _ in top3_group if layer == "next_state_hidden" or layer.startswith("ar_"))

    if supported_ratio >= 0.7 and top3_supported >= 2:
        verdict = "strongly supported"
    elif supported_ratio >= 0.3 or top3_supported >= 1:
        verdict = "partially supported"
    else:
        verdict = "not supported"

    recommended_layers = ranked_group_layers[:3]

    lines = []
    lines.append(f"Verdict: {verdict}")
    lines.append("")
    lines.append("Goal A: Are AGS best layers concentrated in middle/late layers?")
    lines.append(f"- AGS targets with best layer in next_state_hidden/ar_*: {len(supported_layers)}/{len(ags_best_layers)} ({supported_ratio:.2%})")
    lines.append(f"- Top-3 AGS group mean R² layers in late-stage set: {top3_supported}/3")
    lines.append("")
    lines.append("Goal B: Does AGS behave like an intermediate decision state?")
    if top3_group:
        best_layer, best_mean_r2, best_std_r2, best_mean_nmae = top3_group[0]
        lines.append(f"- Best AGS group layer: {best_layer}")
        lines.append(f"- Best AGS group mean R²: {best_mean_r2:.6f}")
        lines.append(f"- Best AGS group std(R² across AGS targets): {best_std_r2:.6f}")
        lines.append(f"- Best AGS group mean normalized MAE: {best_mean_nmae:.6f}")
    lines.append("- Strong support targets are those whose best layer falls in next_state_hidden/ar_*.")
    for target, (layer, metric) in sorted(ags_best_layers.items()):
        lines.append(f"  - {target}: best_layer={layer}, r2={metric['r2']:.6f}, pearson={metric['pearson']:.6f}")
    lines.append("")
    lines.append("Goal C: Candidate layers for policy-facing head")
    for idx, (layer, mean_r2, std_r2, mean_nmae) in enumerate(recommended_layers, start=1):
        lines.append(
            f"- Top {idx}: {layer} | mean_r2={mean_r2:.6f} | std_r2={std_r2:.6f} | mean_nmae={mean_nmae:.6f}"
        )
    lines.append("")
    lines.append("Interpretation:")
    if verdict == "strongly supported":
        lines.append("- AGS signals are clearly concentrated in middle/late layers and support the policy-facing intermediate-state claim.")
    elif verdict == "partially supported":
        lines.append("- AGS group ranking favors some ar_* / next-state layers, but not all AGS targets consistently peak there.")
        lines.append("- This supports a partial intermediate-state story rather than a clean universal concentration claim.")
    else:
        lines.append("- Current AGS targets do not show convincing concentration in middle/late layers.")

    summary_path = os.path.join(output_dir, "goal_assessment.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return verdict, summary_path


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    data = load_results(args.pt_path, args.txt_path)
    per_target = data["per_target"]
    ags_targets = args.targets or [target for target in AGS_TARGETS if target in per_target]
    layers = data["layers"]
    group_stats = compute_group_stats(per_target, ags_targets, layers)

    plot_ags_heatmap(args.output_dir, layers, per_target, ags_targets)
    plot_best_layer_and_baseline(args.output_dir, layers, per_target, ags_targets)
    plot_layerwise_r2(args.output_dir, layers, per_target, group_stats, ags_targets)
    plot_per_target_r2(args.output_dir, layers, per_target, sorted(per_target.keys()))
    plot_ags_stability(args.output_dir, layers, group_stats)
    best_layer_counts = plot_best_layer_distribution(args.output_dir, per_target, ags_targets)
    plot_ags_mean_ranking(args.output_dir, layers, group_stats)
    verdict, summary_path = assess_goals(args.output_dir, per_target, group_stats, best_layer_counts, ags_targets)

    print(f"\nSaved plots to: {args.output_dir}")
    print(f"  Main figures: {os.path.join(args.output_dir, 'main_figures')}")
    print(f"  Per-target: {os.path.join(args.output_dir, 'per_target_r2')}")
    print(f"Saved assessment to: {summary_path}")
    print(f"Verdict: {verdict}")


if __name__ == "__main__":
    main()