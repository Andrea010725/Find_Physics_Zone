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


def main():
    print(f"Loading feature file from: {FEATURE_PATH}")
    data = torch.load(FEATURE_PATH, map_location="cpu")

    layer_features = data["layer_features"]
    labels = data["labels"]
    meta = data.get("meta", {})
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
            if X.ndim != 2:
                raise ValueError(f"Layer {layer_name}: expected 2D features, got {X.shape}")
            if X.shape[0] != len(y):
                raise ValueError(f"Layer {layer_name}: X.shape[0]={X.shape[0]} != len(y)={len(y)}")

            layer_result = evaluate_one_layer_regression(
                X=X,
                y=y,
                groups=groups,
                alpha=1.0,
                n_splits=N_SPLITS,
            )
            results[target_name][layer_name] = layer_result

            print(
                f"{layer_name:15s} | "
                f"mse={layer_result['mse']:.6f} | "
                f"r2={layer_result['r2']:.6f} | "
                f"pearson={layer_result['pearson']:.6f}"
            )

    torch.save(results, RESULT_PATH)
    print(f"\nSaved physics probe results to: {RESULT_PATH}")

    for target_name, layer_result_dict in results.items():
        print(f"\n===== summary sorted by r2: {target_name} =====")
        sorted_items = sorted(
            layer_result_dict.items(),
            key=lambda item: item[1]["r2"],
            reverse=True,
        )
        for layer_name, metric_dict in sorted_items:
            print(
                f"{layer_name:15s} | "
                f"mse={metric_dict['mse']:.6f} | "
                f"r2={metric_dict['r2']:.6f} | "
                f"pearson={metric_dict['pearson']:.6f}"
            )
        print("==========================================")


if __name__ == "__main__":
    main()
