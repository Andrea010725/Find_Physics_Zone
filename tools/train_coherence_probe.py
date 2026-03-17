import os

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
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

FEATURE_PATH = os.path.join(data_root, "coherence_features.pt")
RESULT_PATH = os.path.join(data_root, "coherence_probe_results.pt")
REPORT_PATH = os.path.join(data_root, "coherence_probe_results.txt")

N_SPLITS = 5
RANDOM_STATE = 1234


def build_groups(meta):
    for key in ("seq_id", "source_index"):
        if key in meta:
            groups = meta[key]
            if isinstance(groups, torch.Tensor):
                groups = groups.cpu().numpy()
            groups = np.asarray(groups)
            unique_groups = np.unique(groups)
            if unique_groups.size < 2:
                raise ValueError(
                    "StratifiedGroupKFold needs at least 2 distinct groups. "
                    "Current feature file only contains one base sequence."
                )
            return groups
    raise KeyError("coherence_features.pt is missing meta['seq_id'] / meta['source_index']; group-aware evaluation cannot run.")


def evaluate_one_layer(X, y, groups, n_splits=5, random_state=1234):
    unique_groups = np.unique(groups)
    n_splits = min(n_splits, unique_groups.size)
    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    acc_list = []
    f1_list = []
    auc_list = []

    for train_idx, test_idx in splitter.split(X, y, groups):
        X_train = X[train_idx]
        X_test = X[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                penalty="l2",
                C=1.0,
                solver="liblinear",
                max_iter=2000,
                random_state=random_state,
            )),
        ])
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1]

        acc_list.append(accuracy_score(y_test, y_pred))
        f1_list.append(f1_score(y_test, y_pred, zero_division=0))
        auc_list.append(roc_auc_score(y_test, y_prob))

    return {
        "acc_mean": float(np.mean(acc_list)),
        "acc_std": float(np.std(acc_list)),
        "f1_mean": float(np.mean(f1_list)),
        "f1_std": float(np.std(f1_list)),
        "auc_mean": float(np.mean(auc_list)),
        "auc_std": float(np.std(auc_list)),
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


def evaluate_tokenwise_layer(layer_name, X, y, groups, n_splits=5, random_state=1234, log_every_tokens=64):
    num_tokens = X.shape[1]
    token_results = []

    for token_idx in range(num_tokens):
        token_result = evaluate_one_layer(
            X=X[:, token_idx, :],
            y=y,
            groups=groups,
            n_splits=n_splits,
            random_state=random_state,
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
                f"auc={token_result['auc_mean']:.4f}"
            )

    best_token = max(token_results, key=lambda item: item["auc_mean"])
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
            f"acc={summary_metric['acc_mean']:.4f} | "
            f"f1={summary_metric['f1_mean']:.4f} | "
            f"auc={summary_metric['auc_mean']:.4f}"
        )
    return (
        f"{layer_name:15s} | "
        f"acc={summary_metric['acc_mean']:.4f} | "
        f"f1={summary_metric['f1_mean']:.4f} | "
        f"auc={summary_metric['auc_mean']:.4f}"
    )


def build_report_text(results, y, groups, majority_acc):
    lines = []
    lines.append(f"feature_path: {FEATURE_PATH}")
    lines.append(f"result_path: {RESULT_PATH}")
    lines.append(f"num_samples: {len(y)}")
    lines.append(f"positive: {int(y.sum())}")
    lines.append(f"negative: {int(len(y) - y.sum())}")
    lines.append(f"num_groups: {len(np.unique(groups))}")
    lines.append(f"majority baseline acc: {majority_acc:.4f}")
    lines.append("random baseline auc: 0.5000")
    lines.append("")
    lines.append("===== summary sorted by auc =====")
    sorted_items = sorted(results.items(), key=lambda item: get_summary_metric(item[1])["auc_mean"], reverse=True)
    for layer_name, metric_dict in sorted_items:
        lines.append(format_metric_line(layer_name, metric_dict))
    lines.append("================================")
    return "\n".join(lines) + "\n"


def main():
    print(f"Loading feature file from: {FEATURE_PATH}")
    data = torch.load(FEATURE_PATH, map_location="cpu")

    layer_features = data["layer_features"]
    labels = data["labels"]
    meta = data.get("meta", {})
    feature_mode = data.get("feature_mode", "unknown")

    y = labels["coherence_label"].numpy().astype(np.int64)
    groups = build_groups(meta)

    print("num samples =", len(y))
    print("positive =", int(y.sum()))
    print("negative =", int(len(y) - y.sum()))
    print("num groups =", len(np.unique(groups)))
    print("feature_mode =", feature_mode)

    majority_acc = max(np.mean(y == 0), np.mean(y == 1))
    print(f"majority baseline acc = {majority_acc:.4f}")
    print("random baseline auc ≈ 0.5000")

    results = {}
    print("\n===== training coherence probes (StratifiedGroupKFold by seq_id/source_index) =====")
    for layer_name, feats in layer_features.items():
        X = feats.numpy().astype(np.float32)
        if X.ndim not in (2, 3):
            raise ValueError(f"Layer {layer_name}: expected 2D or 3D features, got {X.shape}")
        if X.shape[0] != len(y):
            raise ValueError(f"Layer {layer_name}: X.shape[0]={X.shape[0]} != len(y)={len(y)}")

        if X.ndim == 2:
            layer_result = evaluate_one_layer(
                X=X,
                y=y,
                groups=groups,
                n_splits=N_SPLITS,
                random_state=RANDOM_STATE,
            )
        else:
            print(f"{layer_name:15s} | tokenwise layer with {X.shape[1]} tokens")
            layer_result = evaluate_tokenwise_layer(
                layer_name=layer_name,
                X=X,
                y=y,
                groups=groups,
                n_splits=N_SPLITS,
                random_state=RANDOM_STATE,
            )
        results[layer_name] = layer_result

        print(format_metric_line(layer_name, layer_result))

    torch.save(results, RESULT_PATH)
    print(f"\nSaved coherence probe results to: {RESULT_PATH}")

    report_text = build_report_text(results, y, groups, majority_acc)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"Saved coherence probe report to: {REPORT_PATH}")

    print("\n===== summary sorted by auc =====")
    sorted_items = sorted(results.items(), key=lambda item: get_summary_metric(item[1])["auc_mean"], reverse=True)
    for layer_name, metric_dict in sorted_items:
        print(format_metric_line(layer_name, metric_dict))
    print("================================")


if __name__ == "__main__":
    main()
