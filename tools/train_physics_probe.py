import os
import torch
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score


FEATURE_PATH = "/home/zhiwen/DrivingWorld/data/physics_features.pt"
RESULT_PATH = "/home/zhiwen/DrivingWorld/data/physics_probe_results.pt"


def pearson_corr(x, y):
    x = np.asarray(x)
    y = np.asarray(y)

    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def evaluate_one_layer_regression(X, y, alpha=1.0):
    """
    对单层 feature 做 LOOCV Ridge 回归，
    返回 mse / r2 / pearson。
    """
    loo = LeaveOneOut()

    y_true_all = []
    y_pred_all = []

    for train_idx, test_idx in loo.split(X):
        X_train = X[train_idx]
        X_test = X[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        # 标准化 + Ridge
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=alpha))
        ])

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        y_true_all.append(float(y_test[0]))
        y_pred_all.append(float(y_pred[0]))

    y_true_all = np.array(y_true_all, dtypexs=np.float32)
    y_pred_all = np.array(y_pred_all, dtype=np.float32)

    mse = mean_squared_error(y_true_all, y_pred_all)
    r2 = r2_score(y_true_all, y_pred_all)
    corr = pearson_corr(y_true_all, y_pred_all)

    return {
        "mse": float(mse),
        "r2": float(r2),
        "pearson": float(corr),
        "y_true": y_true_all,
        "y_pred": y_pred_all,
    }


def main():
    print(f"Loading feature file from: {FEATURE_PATH}")
    data = torch.load(FEATURE_PATH, map_location="cpu")

    layer_features = data["layer_features"]
    labels = data["labels"]

    targets = {
        "speed": labels["speed"].numpy().astype(np.float32),
        "yaw_rate": labels["yaw_rate"].numpy().astype(np.float32),
        "heading_change": labels["heading_change"].numpy().astype(np.float32),
    }

    print("num samples =", len(targets["speed"]))
    print("target stats:")
    for name, y in targets.items():
        print(
            f"  {name}: mean={y.mean():.6f}, std={y.std():.6f}, "
            f"min={y.min():.6f}, max={y.max():.6f}"
        )

    results = {}

    print("\n===== training physics probes =====")
    for target_name, y in targets.items():
        print(f"\n--- target: {target_name} ---")
        results[target_name] = {}

        for layer_name, feats in layer_features.items():
            X = feats.numpy().astype(np.float32)

            print(f"Training layer: {layer_name}, feature shape = {X.shape}")
            layer_result = evaluate_one_layer_regression(
                X=X,
                y=y,
                alpha=1.0,
            )

            results[target_name][layer_name] = {
                "mse": layer_result["mse"],
                "r2": layer_result["r2"],
                "pearson": layer_result["pearson"],
            }

            print(
                f"  mse={layer_result['mse']:.6f}, "
                f"r2={layer_result['r2']:.6f}, "
                f"pearson={layer_result['pearson']:.6f}"
            )

    torch.save(results, RESULT_PATH)
    print(f"\nSaved physics probe results to: {RESULT_PATH}")

    for target_name in results:
        print(f"\n===== summary sorted by r2: {target_name} =====")
        sorted_items = sorted(
            results[target_name].items(),
            key=lambda x: x[1]["r2"],
            reverse=True
        )
        for layer_name, r in sorted_items:
            print(
                f"{layer_name:15s} | "
                f"mse={r['mse']:.6f} | "
                f"r2={r['r2']:.6f} | "
                f"pearson={r['pearson']:.6f}"
            )
        print("==========================================")


if __name__ == "__main__":
    main()
