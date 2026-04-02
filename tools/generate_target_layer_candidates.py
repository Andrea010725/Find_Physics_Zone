import argparse
import json
import os
import re


SECTION_TARGET_RE = re.compile(r"^===== summary sorted by r2: (.+) =====$")
METRIC_RE = re.compile(
    r"^(?P<layer>[^|]+?)\s*\|\s*"
    r"(?:(?:best_token=.*?\|\s*))?"
    r"mse=(?P<mse>-?\d+\.\d+)\s*\|\s*"
    r"mae=(?P<mae>-?\d+\.\d+)\s*\|\s*"
    r"nmae=(?P<nmae>-?\d+\.\d+)\s*\|\s*"
    r"r2=(?P<r2>-?\d+\.\d+)\s*\|\s*"
    r"pearson=(?P<pearson>-?\d+\.\d+)"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--txt_path",
        type=str,
        default="/home/yanda/Find_Physics_Zone/data/physics_probe_results.txt",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Top-K readout layers by R2 to keep as candidates.",
    )
    parser.add_argument(
        "--r2_margin",
        type=float,
        default=0.02,
        help="Also keep layers within (best_r2 - r2_margin).",
    )
    parser.add_argument(
        "--out_json",
        type=str,
        default="/home/yanda/Find_Physics_Zone/data/target_layer_candidates.json",
    )
    parser.add_argument(
        "--out_txt",
        type=str,
        default="/home/yanda/Find_Physics_Zone/data/target_layer_candidates.txt",
    )
    return parser.parse_args()


def is_readout_layer(layer_name):
    if layer_name.startswith("time_space_"):
        return True
    if layer_name == "next_state_hidden":
        return True
    if layer_name.startswith("ar_"):
        return True
    return False


def load_per_target_metrics(txt_path):
    with open(txt_path, "r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f]

    per_target = {}
    current_target = None

    for raw in lines:
        line = raw.strip()
        if not line:
            continue

        m_target = SECTION_TARGET_RE.match(line)
        if m_target:
            current_target = m_target.group(1)
            per_target[current_target] = {}
            continue

        if line.startswith("===== AGS GROUP SUMMARY"):
            current_target = None
            continue

        if line.startswith("=====") or line.startswith("==="):
            continue

        if current_target is None:
            continue

        m_metric = METRIC_RE.match(line)
        if not m_metric:
            continue
        layer = m_metric.group("layer").strip()
        per_target[current_target][layer] = {
            "r2": float(m_metric.group("r2")),
            "pearson": float(m_metric.group("pearson")),
            "mse": float(m_metric.group("mse")),
            "mae": float(m_metric.group("mae")),
            "nmae": float(m_metric.group("nmae")),
        }

    return per_target


def choose_candidates(per_target, top_k, r2_margin):
    output = {}

    for target, layer_dict in sorted(per_target.items()):
        all_sorted = sorted(layer_dict.items(), key=lambda item: item[1]["r2"], reverse=True)
        readout_sorted = [item for item in all_sorted if is_readout_layer(item[0])]

        if not all_sorted:
            output[target] = {
                "best_overall": None,
                "best_readout": None,
                "top_k_readout_layers": [],
                "margin_readout_layers": [],
                "recommended_layers": [],
                "note": "No layers found in report section.",
            }
            continue

        best_overall_layer, best_overall_metric = all_sorted[0]
        best_readout_layer, best_readout_metric = (None, None)
        if readout_sorted:
            best_readout_layer, best_readout_metric = readout_sorted[0]

        top_k_readout = readout_sorted[:top_k]
        if best_readout_metric is not None:
            margin_threshold = best_readout_metric["r2"] - r2_margin
            margin_readout = [item for item in readout_sorted if item[1]["r2"] >= margin_threshold]
        else:
            margin_readout = []

        recommended = margin_readout[:top_k] if margin_readout else top_k_readout
        if not recommended and best_readout_metric is not None:
            recommended = [readout_sorted[0]]

        rec_positive = [item for item in recommended if item[1]["r2"] > 0.0]
        if rec_positive:
            recommended = rec_positive

        note = ""
        if best_readout_metric is None:
            note = "No readout layers (time_space/next_state_hidden/ar) found."
        elif best_readout_metric["r2"] <= 0.0:
            note = "Best readout-layer R2 <= 0; treat this target as hard/noisy for readout."

        output[target] = {
            "best_overall": {
                "layer": best_overall_layer,
                "r2": best_overall_metric["r2"],
                "pearson": best_overall_metric["pearson"],
            },
            "best_readout": None
            if best_readout_metric is None
            else {
                "layer": best_readout_layer,
                "r2": best_readout_metric["r2"],
                "pearson": best_readout_metric["pearson"],
            },
            "top_k_readout_layers": [
                {"layer": layer, "r2": metric["r2"], "pearson": metric["pearson"]}
                for layer, metric in top_k_readout
            ],
            "margin_readout_layers": [
                {"layer": layer, "r2": metric["r2"], "pearson": metric["pearson"]}
                for layer, metric in margin_readout
            ],
            "recommended_layers": [
                {"layer": layer, "r2": metric["r2"], "pearson": metric["pearson"]}
                for layer, metric in recommended
            ],
            "recommended_layer_names": [layer for layer, _ in recommended],
            "note": note,
        }

    return output


def save_outputs(out_json, out_txt, payload):
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    lines = []
    lines.append(f"source_txt: {payload['source_txt']}")
    lines.append(f"criteria: top_k={payload['criteria']['top_k']}, r2_margin={payload['criteria']['r2_margin']}")
    lines.append("")

    for target, result in payload["targets"].items():
        lines.append(f"=== {target} ===")
        best_overall = result["best_overall"]
        lines.append(
            f"best_overall: {best_overall['layer']} | r2={best_overall['r2']:.6f} | pearson={best_overall['pearson']:.6f}"
        )
        if result["best_readout"] is None:
            lines.append("best_readout: None")
        else:
            lines.append(
                "best_readout: "
                f"{result['best_readout']['layer']} | "
                f"r2={result['best_readout']['r2']:.6f} | "
                f"pearson={result['best_readout']['pearson']:.6f}"
            )
        lines.append("recommended_layers: " + ", ".join(result["recommended_layer_names"]))
        if result["note"]:
            lines.append("note: " + result["note"])
        lines.append("")

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).strip() + "\n")


def main():
    args = parse_args()
    per_target = load_per_target_metrics(args.txt_path)
    targets = choose_candidates(
        per_target=per_target,
        top_k=args.top_k,
        r2_margin=args.r2_margin,
    )
    payload = {
        "source_txt": args.txt_path,
        "criteria": {
            "top_k": int(args.top_k),
            "r2_margin": float(args.r2_margin),
            "readout_layer_filter": "time_space_* | next_state_hidden | ar_*",
        },
        "targets": targets,
    }
    save_outputs(args.out_json, args.out_txt, payload)
    print(f"Saved JSON: {args.out_json}")
    print(f"Saved TXT : {args.out_txt}")

    print("\n===== Candidate Summary =====")
    for target, result in payload["targets"].items():
        rec = result["recommended_layer_names"]
        print(f"{target:24s} -> {', '.join(rec) if rec else '[]'}")


if __name__ == "__main__":
    main()
