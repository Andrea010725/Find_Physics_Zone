import os

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

FEATURE_PATH = os.path.join(data_root, "physics_features.pt")
RESULT_PATH = os.path.join(data_root, "physics_probe_results.pt")
REPORT_PATH = os.path.join(data_root, "physics_probe_results.txt")

TARGET_KEYS = [
    "past_avg_speed",
    "past_avg_yaw_rate",
    "past_heading_change",
    "future_speed",
    "future_yaw_rate",
    "future_delta_yaw",
]

N_SPLITS = 5


def pearson_corr(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


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


def evaluate_one_layer_regression(X, y, groups, alpha=1.0, n_splits=5):
    unique_groups = np.unique(groups)
    n_splits = min(n_splits, unique_groups.size)
    splitter = GroupKFold(n_splits=n_splits)

    y_true_all = []
    y_pred_all = []

    for train_idx, test_idx in splitter.split(X, y, groups):
        X_train = X[train_idx]
        X_test = X[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        model = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=alpha)),
        ])
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        y_true_all.extend(y_test.tolist())
        y_pred_all.extend(y_pred.tolist())

    y_true_all = np.array(y_true_all, dtype=np.float32)
    y_pred_all = np.array(y_pred_all, dtype=np.float32)

    mse = mean_squared_error(y_true_all, y_pred_all)
    r2 = r2_score(y_true_all, y_pred_all)
    corr = pearson_corr(y_true_all, y_pred_all)

    return {
        "mse": float(mse),
        "r2": float(r2),
        "pearson": float(corr),
        "num_groups": int(unique_groups.size),
        "num_folds": int(n_splits),
    }


def maybe_grid_token_name(prefix, patch_idx, num_patches):
    if num_patches == 512:
        row, col = divmod(patch_idx, 32)
        return f"{prefix}[{row},{col}]"
    return f"{prefix}[{patch_idx}]"


def describe_token(layer_name, token_idx, num_tokens):
    if layer_name == "tokenizer_last":
        return maybe_grid_token_name("img", token_idx, num_tokens)

    if layer_name.startswith("time_space_"):
        if token_idx == 0:
            return "yaw"
        if token_idx == 1:
            return "pose_x"
        if token_idx == 2:
            return "pose_y"
        return maybe_grid_token_name("img", token_idx - 3, num_tokens - 3)

    if layer_name == "next_state_hidden" or layer_name.startswith("ar_"):
        if token_idx == 0:
            return "yaw"
        if token_idx == 1:
            return "pose_x"
        if token_idx == 2:
            return "pose_y"
        return maybe_grid_token_name("img", token_idx - 3, num_tokens - 3)

    return f"token_{token_idx}"


def evaluate_tokenwise_layer_regression(layer_name, X, y, groups, alpha=1.0, n_splits=5, log_every_tokens=64):
    num_tokens = X.shape[1]
    token_results = []

    for token_idx in range(num_tokens):
        token_result = evaluate_one_layer_regression(
            X=X[:, token_idx, :],
            y=y,
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
                f"r2={token_result['r2']:.6f}"
            )

    best_token = max(token_results, key=lambda item: item["r2"])
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
    if "token_idx" in summary_metric:
        return (
            f"{layer_name:15s} | "
            f"best_token={summary_metric['token_idx']:3d} ({summary_metric['token_name']}) | "
            f"mse={summary_metric['mse']:.6f} | "
            f"r2={summary_metric['r2']:.6f} | "
            f"pearson={summary_metric['pearson']:.6f}"
        )
    return (
        f"{layer_name:15s} | "
        f"mse={summary_metric['mse']:.6f} | "
        f"r2={summary_metric['r2']:.6f} | "
        f"pearson={summary_metric['pearson']:.6f}"
    )


def build_report_text(results, targets, groups):
    lines = []
    lines.append(f"feature_path: {FEATURE_PATH}")
    lines.append(f"result_path: {RESULT_PATH}")
    lines.append(f"num_samples: {len(next(iter(targets.values())))}")
    lines.append(f"num_groups: {len(np.unique(groups))}")
    lines.append("")
    lines.append("target stats:")
    for name, y in targets.items():
        lines.append(
            f"  {name}: mean={y.mean():.6f}, std={y.std():.6f}, "
            f"min={y.min():.6f}, max={y.max():.6f}"
        )

    for target_name, layer_result_dict in results.items():
        lines.append("")
        lines.append(f"===== summary sorted by r2: {target_name} =====")
        sorted_items = sorted(
            layer_result_dict.items(),
            key=lambda item: get_summary_metric(item[1])["r2"],
            reverse=True,
        )
        for layer_name, metric_dict in sorted_items:
            lines.append(format_metric_line(layer_name, metric_dict))
        lines.append("==========================================")

    return "\n".join(lines) + "\n"


def main():
    print(f"Loading feature file from: {FEATURE_PATH}")
    data = torch.load(FEATURE_PATH, map_location="cpu")

    layer_features = data["layer_features"]
    labels = data["labels"]
    meta = data.get("meta", {})
    feature_mode = data.get("feature_mode", "unknown")
    groups = build_groups(meta)

    targets = {
        key: labels[key].numpy().astype(np.float32)
        for key in TARGET_KEYS
        if key in labels
    }
    if not targets:
        raise KeyError(f"No physics targets found. Expected any of: {TARGET_KEYS}")

    print("num samples =", len(next(iter(targets.values()))))
    print("num groups =", len(np.unique(groups)))
    print("feature_mode =", feature_mode)
    print("target stats:")
    for name, y in targets.items():
        print(
            f"  {name}: mean={y.mean():.6f}, std={y.std():.6f}, "
            f"min={y.min():.6f}, max={y.max():.6f}"
        )

    results = {}
    print("\n===== training physics probes (GroupKFold by seq_id) =====")
    for target_name, y in targets.items():
        print(f"\n--- target: {target_name} ---")
        results[target_name] = {}

        for layer_name, feats in layer_features.items():
            X = feats.numpy().astype(np.float32)
            if X.ndim not in (2, 3):
                raise ValueError(f"Layer {layer_name}: expected 2D or 3D features, got {X.shape}")
            if X.shape[0] != len(y):
                raise ValueError(f"Layer {layer_name}: X.shape[0]={X.shape[0]} != len(y)={len(y)}")

            if X.ndim == 2:
                layer_result = evaluate_one_layer_regression(
                    X=X,
                    y=y,
                    groups=groups,
                    alpha=1.0,
                    n_splits=N_SPLITS,
                )
            else:
                print(f"{layer_name:15s} | tokenwise layer with {X.shape[1]} tokens")
                layer_result = evaluate_tokenwise_layer_regression(
                    layer_name=layer_name,
                    X=X,
                    y=y,
                    groups=groups,
                    alpha=1.0,
                    n_splits=N_SPLITS,
                )
            results[target_name][layer_name] = layer_result

            print(format_metric_line(layer_name, layer_result))

    torch.save(results, RESULT_PATH)
    print(f"\nSaved physics probe results to: {RESULT_PATH}")

    report_text = build_report_text(results, targets, groups)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"Saved physics probe report to: {REPORT_PATH}")

    for target_name, layer_result_dict in results.items():
        print(f"\n===== summary sorted by r2: {target_name} =====")
        sorted_items = sorted(
            layer_result_dict.items(),
            key=lambda item: get_summary_metric(item[1])["r2"],
            reverse=True,
        )
        for layer_name, metric_dict in sorted_items:
            print(format_metric_line(layer_name, metric_dict))
        print("==========================================")


if __name__ == "__main__":
    main()
