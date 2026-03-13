import os
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


FEATURE_PATH = "/home/zhiwen/DrivingWorld/data/coherence_features.pt"
RESULT_PATH = "/home/zhiwen/DrivingWorld/data/coherence_probe_results.pt"

N_SPLITS = 5
RANDOM_STATE = 1234


def evaluate_one_layer(X, y, n_splits=5, random_state=1234):
    """
    对单层 feature 做 stratified k-fold logistic regression 分类，
    返回平均 acc / f1 / auc。
    """
    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state
    )

    acc_list = []
    f1_list = []
    auc_list = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train = X[train_idx]
        X_test = X[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        # 简单线性 probe
        clf = LogisticRegression(
            penalty="l2",
            C=1.0,
            solver="liblinear",
            max_iter=2000,
            random_state=random_state,
        )
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)

        acc_list.append(acc)
        f1_list.append(f1)
        auc_list.append(auc)

    return {
        "acc_mean": float(np.mean(acc_list)),
        "acc_std": float(np.std(acc_list)),
        "f1_mean": float(np.mean(f1_list)),
        "f1_std": float(np.std(f1_list)),
        "auc_mean": float(np.mean(auc_list)),
        "auc_std": float(np.std(auc_list)),
    }


def main():
    print(f"Loading feature file from: {FEATURE_PATH}")
    data = torch.load(FEATURE_PATH, map_location="cpu")

    layer_features = data["layer_features"]
    labels = data["labels"]
    y = labels["coherence_label"].numpy().astype(np.int64)

    print("num samples =", len(y))
    print("positive =", int(y.sum()))
    print("negative =", int(len(y) - y.sum()))

    results = {}

    print("\n===== training coherence probes =====")
    for layer_name, feats in layer_features.items():
        X = feats.numpy().astype(np.float32)

        print(f"Training layer: {layer_name}, feature shape = {X.shape}")
        layer_result = evaluate_one_layer(
            X=X,
            y=y,
            n_splits=N_SPLITS,
            random_state=RANDOM_STATE
        )
        results[layer_name] = layer_result

        print(
            f"  acc={layer_result['acc_mean']:.4f}±{layer_result['acc_std']:.4f}, "
            f"f1={layer_result['f1_mean']:.4f}±{layer_result['f1_std']:.4f}, "
            f"auc={layer_result['auc_mean']:.4f}±{layer_result['auc_std']:.4f}"
        )

    torch.save(results, RESULT_PATH)
    print(f"\nSaved coherence probe results to: {RESULT_PATH}")

    print("\n===== summary sorted by auc =====")
    sorted_items = sorted(results.items(), key=lambda x: x[1]["auc_mean"], reverse=True)
    for layer_name, r in sorted_items:
        print(
            f"{layer_name:15s} | "
            f"acc={r['acc_mean']:.4f} | "
            f"f1={r['f1_mean']:.4f} | "
            f"auc={r['auc_mean']:.4f}"
        )
    print("================================")


if __name__ == "__main__":
    main()
