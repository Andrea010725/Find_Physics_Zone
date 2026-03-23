import argparse
import os
import re

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


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

DEFAULT_FEATURE_PATH = os.path.join(data_root, "physics_features.pt")
DEFAULT_RESULT_PATH = os.path.join(data_root, "planning_family_probe_results.pt")
DEFAULT_REPORT_PATH = os.path.join(data_root, "planning_family_probe_results.txt")
DEFAULT_ALPHA = 1.0
DEFAULT_N_SPLITS = 5


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_path", type=str, default=DEFAULT_FEATURE_PATH)
    parser.add_argument("--result_path", type=str, default=DEFAULT_RESULT_PATH)
    parser.add_argument("--report_path", type=str, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--n_splits", type=int, default=DEFAULT_N_SPLITS)
    parser.add_argument(
        "--families",
        nargs="*",
        default=None,
        help="Optional subset of planning head families to evaluate.",
    )
    return parser.parse_args()


def pearson_corr(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def sanitize_features(X):
    X = np.asarray(X, dtype=np.float32)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def build_groups(meta):
    if "seq_id" not in meta:
        raise KeyError("physics_features.pt is missing meta['seq_id']; group-aware evaluation cannot run.")
    groups = meta["seq_id"]
    if isinstance(groups, torch.Tensor):
        groups = groups.cpu().numpy()
    groups = np.asarray(groups)
    unique_groups = np.unique(groups)
    if unique_groups.size < 2:
        raise ValueError(
            "GroupKFold needs at least 2 distinct seq_id groups. "
            "Current feature file only contains one sequence, so the result would leak heavily."
        )
    return groups


def maybe_grid_token_name(prefix, patch_idx, num_patches):
    if num_patches == 512:
        row, col = divmod(patch_idx, 32)
        return f"{prefix}[{row},{col}]"
    return f"{prefix}[{patch_idx}]"


def describe_token(layer_name, token_idx, num_tokens):
    if layer_name == "tokenizer_last":
        return maybe_grid_token_name("img", token_idx, num_tokens)

    if layer_name.startswith("time_space_") or layer_name == "next_state_hidden" or layer_name.startswith("ar_"):
        if token_idx == 0:
            return "yaw"
        if token_idx == 1:
            return "pose_x"
        if token_idx == 2:
            return "pose_y"
        return maybe_grid_token_name("img", token_idx - 3, num_tokens - 3)

    return f"token_{token_idx}"


def infer_planning_head_families(label_keys):
    label_key_set = set(label_keys)
    rollout_keys = sorted(
        [key for key in label_keys if re.fullmatch(r"rollout_kp\d+_[xy]", key)],
        key=lambda key: (
            int(re.search(r"\d+", key).group()),
            0 if key.endswith("_x") else 1,
        ),
    )

    families = {
        "control_head": [
            key for key in ["control_delta_v", "control_cum_delta_yaw"]
            if key in label_key_set
        ],
        "endpoint_head": [
            key for key in ["endpoint_lateral_disp", "endpoint_forward_progress", "endpoint_heading"]
            if key in label_key_set
        ],
        "geometry_head": [
            key for key in ["geometry_mean_curvature", "geometry_integrated_curvature"]
            if key in label_key_set
        ],
        "rollout_head": rollout_keys,
    }
    return {name: keys for name, keys in families.items() if keys}


def build_targets(labels, head_families):
    targets = {}
    target_stats = {}

    for family_name, family_keys in head_families.items():
        family_arrays = []
        target_stats[family_name] = {}
        for key in family_keys:
            if key not in labels:
                raise KeyError(f"Missing label '{key}' in feature file.")
            y = labels[key].cpu().numpy().astype(np.float32)
            family_arrays.append(y)
            target_stats[family_name][key] = {
                "mean": float(y.mean()),
                "std": float(y.std()),
                "min": float(y.min()),
                "max": float(y.max()),
            }
        targets[family_name] = np.stack(family_arrays, axis=1)

    return targets, target_stats


def evaluate_family_regression(X, Y, target_keys, groups, alpha=1.0, n_splits=5):
    X = sanitize_features(X)
    unique_groups = np.unique(groups)
    n_splits = min(n_splits, unique_groups.size)
    splitter = GroupKFold(n_splits=n_splits)

    y_true_all = []
    y_pred_all = []

    for train_idx, test_idx in splitter.split(X, Y, groups):
        X_train = X[train_idx]
        X_test = X[test_idx]
        y_train = Y[train_idx]
        y_test = Y[test_idx]

        model = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=alpha)),
        ])
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        y_true_all.append(y_test)
        y_pred_all.append(y_pred)

    y_true_all = np.concatenate(y_true_all, axis=0).astype(np.float32)
    y_pred_all = np.concatenate(y_pred_all, axis=0).astype(np.float32)

    per_target = {}
    per_target_r2 = []
    per_target_pearson = []
    for target_idx, target_name in enumerate(target_keys):
        y_true = y_true_all[:, target_idx]
        y_pred = y_pred_all[:, target_idx]
        target_r2 = float(r2_score(y_true, y_pred))
        target_pearson = pearson_corr(y_true, y_pred)
        per_target[target_name] = {
            "mse": float(mean_squared_error(y_true, y_pred)),
            "r2": target_r2,
            "pearson": target_pearson,
        }
        per_target_r2.append(target_r2)
        per_target_pearson.append(target_pearson)

    return {
        "mse": float(mean_squared_error(y_true_all, y_pred_all)),
        "r2_uniform": float(r2_score(y_true_all, y_pred_all, multioutput="uniform_average")),
        "r2_variance_weighted": float(r2_score(y_true_all, y_pred_all, multioutput="variance_weighted")),
        "mean_target_r2": float(np.mean(per_target_r2)),
        "mean_target_pearson": float(np.mean(per_target_pearson)),
        "num_groups": int(unique_groups.size),
        "num_folds": int(n_splits),
        "per_target": per_target,
    }


def evaluate_tokenwise_family(layer_name, X, Y, target_keys, groups, alpha=1.0, n_splits=5, log_every_tokens=64):
    num_tokens = X.shape[1]
    token_results = []

    for token_idx in range(num_tokens):
        token_result = evaluate_family_regression(
            X=X[:, token_idx, :],
            Y=Y,
            target_keys=target_keys,
            groups=groups,
            alpha=alpha,
            n_splits=n_splits,
        )
        token_result["token_idx"] = int(token_idx)
        token_result["token_name"] = describe_token(layer_name, token_idx, num_tokens)
        token_results.append(token_result)

        should_log = (
            token_idx == 0
            or (token_idx + 1) % log_every_tokens == 0
            or (token_idx + 1) == num_tokens
        )
        if should_log:
            print(
                f"  token_progress={token_idx + 1}/{num_tokens} "
                f"current={token_result['token_name']} "
                f"r2_vw={token_result['r2_variance_weighted']:.6f}"
            )

    best_token = max(token_results, key=lambda item: item["r2_variance_weighted"])
    return {
        "feature_ndim": 3,
        "num_tokens": int(num_tokens),
        "token_results": token_results,
        "best_token": dict(best_token),
    }


def get_summary_metric(metric_dict):
    if "best_token" in metric_dict:
        return metric_dict["best_token"]
    return metric_dict


def format_metric_line(layer_name, metric_dict):
    summary_metric = get_summary_metric(metric_dict)
    prefix = f"{layer_name:15s} | "
    if "token_idx" in summary_metric:
        prefix += f"best_token={summary_metric['token_idx']:3d} ({summary_metric['token_name']}) | "
    return (
        prefix
        + f"mse={summary_metric['mse']:.6f} | "
        + f"r2_vw={summary_metric['r2_variance_weighted']:.6f} | "
        + f"r2_mean={summary_metric['mean_target_r2']:.6f} | "
        + f"pearson_mean={summary_metric['mean_target_pearson']:.6f}"
    )


def build_report_text(feature_path, result_path, results, target_stats, groups, feature_mode, head_families):
    lines = []
    lines.append(f"feature_path: {feature_path}")
    lines.append(f"result_path: {result_path}")
    lines.append(f"num_samples: {len(groups)}")
    lines.append(f"num_groups: {len(np.unique(groups))}")
    lines.append(f"feature_mode: {feature_mode}")
    lines.append("")
    lines.append("planning head families:")
    for family_name, family_keys in head_families.items():
        lines.append(f"  {family_name}: {', '.join(family_keys)}")

    for family_name, stat_dict in target_stats.items():
        lines.append("")
        lines.append(f"target stats: {family_name}")
        for target_name, stats in stat_dict.items():
            lines.append(
                f"  {target_name}: mean={stats['mean']:.6f}, std={stats['std']:.6f}, "
                f"min={stats['min']:.6f}, max={stats['max']:.6f}"
            )

    for family_name, layer_result_dict in results.items():
        lines.append("")
        lines.append(f"===== summary sorted by r2_vw: {family_name} =====")
        sorted_items = sorted(
            layer_result_dict.items(),
            key=lambda item: get_summary_metric(item[1])["r2_variance_weighted"],
            reverse=True,
        )
        for layer_name, metric_dict in sorted_items:
            lines.append(format_metric_line(layer_name, metric_dict))
        lines.append("============================================")

    return "\n".join(lines) + "\n"


def main():
    args = parse_args()

    print(f"Loading feature file from: {args.feature_path}")
    data = torch.load(args.feature_path, map_location="cpu")

    layer_features = data["layer_features"]
    labels = data["labels"]
    meta = data.get("meta", {})
    feature_mode = data.get("feature_mode", "unknown")
    groups = build_groups(meta)

    head_families = data.get("planning_head_families") or infer_planning_head_families(labels.keys())
    if args.families is not None:
        head_families = {name: keys for name, keys in head_families.items() if name in set(args.families)}
    if not head_families:
        raise KeyError("No planning head families found in feature file.")

    targets, target_stats = build_targets(labels, head_families)

    print("num samples =", targets[next(iter(targets))].shape[0])
    print("num groups =", len(np.unique(groups)))
    print("feature_mode =", feature_mode)
    print("head_families =", head_families)

    results = {}
    print("\n===== training planning family probes (GroupKFold by seq_id) =====")
    for family_name, Y in targets.items():
        print(f"\n--- family: {family_name} ---")
        results[family_name] = {}
        target_keys = head_families[family_name]

        for layer_name, feats in layer_features.items():
            X = feats.numpy().astype(np.float32)
            if X.ndim not in (2, 3):
                raise ValueError(f"Layer {layer_name}: expected 2D or 3D features, got {X.shape}")
            if X.shape[0] != Y.shape[0]:
                raise ValueError(f"Layer {layer_name}: X.shape[0]={X.shape[0]} != Y.shape[0]={Y.shape[0]}")

            if X.ndim == 2:
                layer_result = evaluate_family_regression(
                    X=X,
                    Y=Y,
                    target_keys=target_keys,
                    groups=groups,
                    alpha=args.alpha,
                    n_splits=args.n_splits,
                )
            else:
                print(f"{layer_name:15s} | tokenwise layer with {X.shape[1]} tokens")
                layer_result = evaluate_tokenwise_family(
                    layer_name=layer_name,
                    X=X,
                    Y=Y,
                    target_keys=target_keys,
                    groups=groups,
                    alpha=args.alpha,
                    n_splits=args.n_splits,
                )
            results[family_name][layer_name] = layer_result
            print(format_metric_line(layer_name, layer_result))

    payload = {
        "feature_path": args.feature_path,
        "feature_mode": feature_mode,
        "head_families": head_families,
        "target_stats": target_stats,
        "results": results,
    }
    torch.save(payload, args.result_path)
    print(f"\nSaved planning family probe results to: {args.result_path}")

    report_text = build_report_text(
        feature_path=args.feature_path,
        result_path=args.result_path,
        results=results,
        target_stats=target_stats,
        groups=groups,
        feature_mode=feature_mode,
        head_families=head_families,
    )
    with open(args.report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"Saved planning family probe report to: {args.report_path}")


if __name__ == "__main__":
    main()
