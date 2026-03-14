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


def main():
    print(f"Loading feature file from: {FEATURE_PATH}")
    data = torch.load(FEATURE_PATH, map_location="cpu")

    layer_features = data["layer_features"]
    labels = data["labels"]
    meta = data.get("meta", {})

    y = labels["coherence_label"].numpy().astype(np.int64)
    groups = build_groups(meta)

    print("num samples =", len(y))
    print("positive =", int(y.sum()))
    print("negative =", int(len(y) - y.sum()))
    print("num groups =", len(np.unique(groups)))

    majority_acc = max(np.mean(y == 0), np.mean(y == 1))
    print(f"majority baseline acc = {majority_acc:.4f}")
    print("random baseline auc ≈ 0.5000")

    results = {}
    print("\n===== training coherence probes (StratifiedGroupKFold by seq_id/source_index) =====")
    for layer_name, feats in layer_features.items():
        X = feats.numpy().astype(np.float32)
        if X.ndim != 2:
            raise ValueError(f"Layer {layer_name}: expected 2D features, got {X.shape}")
        if X.shape[0] != len(y):
            raise ValueError(f"Layer {layer_name}: X.shape[0]={X.shape[0]} != len(y)={len(y)}")

        layer_result = evaluate_one_layer(
            X=X,
            y=y,
            groups=groups,
            n_splits=N_SPLITS,
            random_state=RANDOM_STATE,
        )
        results[layer_name] = layer_result

        print(
            f"{layer_name:15s} | "
            f"acc={layer_result['acc_mean']:.4f}±{layer_result['acc_std']:.4f} | "
            f"f1={layer_result['f1_mean']:.4f}±{layer_result['f1_std']:.4f} | "
            f"auc={layer_result['auc_mean']:.4f}±{layer_result['auc_std']:.4f}"
        )

    torch.save(results, RESULT_PATH)
    print(f"\nSaved coherence probe results to: {RESULT_PATH}")

    print("\n===== summary sorted by auc =====")
    sorted_items = sorted(results.items(), key=lambda item: item[1]["auc_mean"], reverse=True)
    for layer_name, metric_dict in sorted_items:
        print(
            f"{layer_name:15s} | "
            f"acc={metric_dict['acc_mean']:.4f} | "
            f"f1={metric_dict['f1_mean']:.4f} | "
            f"auc={metric_dict['auc_mean']:.4f}"
        )
    print("================================")


if __name__ == "__main__":
    main()
