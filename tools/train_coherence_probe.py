import os
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler 
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
        clf = Pipeline([
    		("scaler", StandardScaler()),
    		("clf", LogisticRegression(
        	penalty="l2",
        	C=1.0,
        	solver="liblinear",
        	max_iter=2000,
        	random_state=random_state,
    		))
			])

        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)

        f1 = f1_score(y_test, y_pred, zero_division=0)
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
    "acc_list": acc_list,
    "f1_list": f1_list,
    "auc_list": auc_list,
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
    
    majority_acc = max(np.mean(y == 0), np.mean(y == 1))
    print(f"majority baseline acc = {majority_acc:.4f}")
    print("random baseline auc ≈ 0.5000")

    results = {}

    print("\n===== training coherence probes =====")
    
    for layer_name, feats in layer_features.items():
    
        assert X.shape[0] == len(y), (
    		f"Layer {layer_name}: X.shape[0]={X.shape[0]} != len(y)={len(y)}"
			)
        assert X.ndim == 2, f"Layer {layer_name}: expected 2D features, got {X.shape}"
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
