import os
import argparse

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

DEFAULT_FEATURE_PATH = os.path.join(data_root, "coherence_features.pt")
DEFAULT_RESULT_PATH = os.path.join(data_root, "coherence_probe_results.pt")
DEFAULT_REPORT_PATH = os.path.join(data_root, "coherence_probe_results.txt")

N_SPLITS = 5
RANDOM_STATE = 1234


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_path", type=str, default=DEFAULT_FEATURE_PATH)
    parser.add_argument("--test_feature_path", type=str, default=None)
    parser.add_argument("--result_path", type=str, default=None)
    parser.add_argument("--report_path", type=str, default=None)
    parser.add_argument("--n_splits", type=int, default=N_SPLITS)
    parser.add_argument("--random_state", type=int, default=RANDOM_STATE)
    return parser.parse_args()


def resolve_output_paths(args):
    if args.result_path is not None and args.report_path is not None:
        return args.result_path, args.report_path

    if args.test_feature_path is None:
        result_path = args.result_path or DEFAULT_RESULT_PATH
        report_path = args.report_path or DEFAULT_REPORT_PATH
    else:
        base_dir = os.path.dirname(os.path.abspath(args.feature_path))
        result_path = args.result_path or os.path.join(base_dir, "coherence_probe_holdout_results.pt")
        report_path = args.report_path or os.path.join(base_dir, "coherence_probe_holdout_results.txt")

    for path in (result_path, report_path):
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)
    return result_path, report_path


def load_feature_bundle(path):
    data = torch.load(path, map_location="cpu")
    return data["layer_features"], data["labels"], data.get("meta", {}), data.get("feature_mode", "unknown")


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


def get_groups_for_stats(meta):
    for key in ("seq_id", "source_index"):
        if key in meta:
            groups = meta[key]
            if isinstance(groups, torch.Tensor):
                groups = groups.cpu().numpy()
            return np.asarray(groups)
    raise KeyError("coherence_features.pt is missing meta['seq_id'] / meta['source_index']; cannot read group stats.")


def get_negative_type_array(meta, expected_len):
    if "negative_type" not in meta:
        raise KeyError("coherence_features.pt is missing meta['negative_type']; cannot split results by negative type.")

    negative_type = meta["negative_type"]
    if isinstance(negative_type, torch.Tensor):
        negative_type = negative_type.cpu().numpy().tolist()

    negative_type = np.asarray(list(negative_type), dtype=object)
    if negative_type.shape[0] != expected_len:
        raise ValueError(
            f"negative_type length mismatch: {negative_type.shape[0]} vs expected {expected_len}"
        )
    return negative_type


def sanitize_features(X):
    X = np.asarray(X, dtype=np.float32)
    invalid_mask = ~np.isfinite(X)
    num_invalid = int(invalid_mask.sum())
    if num_invalid > 0:
        X = X.copy()
        X[invalid_mask] = 0.0
    return X, num_invalid


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
        X_train, _ = sanitize_features(X[train_idx])
        X_test, _ = sanitize_features(X[test_idx])
        y_train = y[train_idx]
        y_test = y[test_idx]

        clf = fit_classifier(X_train, y_train, random_state=random_state)

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


def fit_classifier(X_train, y_train, random_state=1234):
    X_train, _ = sanitize_features(X_train)
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
    return clf


def evaluate_holdout_layer(X_train, y_train, X_test, y_test, random_state=1234):
    clf = fit_classifier(X_train, y_train, random_state=random_state)
    X_test, _ = sanitize_features(X_test)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    return {
        "acc": float(accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_test, y_prob)),
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


def get_nested_summary_metric(metric_dict, key):
    summary_metric = metric_dict[key]
    if "best_token" in summary_metric:
        return summary_metric["best_token"]
    return summary_metric


def format_holdout_metric_line(layer_name, metric_dict):
    train_summary = get_nested_summary_metric(metric_dict, "train_cv")
    test_summary = get_nested_summary_metric(metric_dict, "test_holdout")
    if "token_idx" in test_summary:
        return (
            f"{layer_name:15s} | "
            f"selected_token={test_summary['token_idx']:3d} ({test_summary['token_name']}) | "
            f"train_cv_auc={train_summary['auc_mean']:.4f} | "
            f"test_auc={test_summary['auc']:.4f} | "
            f"test_acc={test_summary['acc']:.4f} | "
            f"test_f1={test_summary['f1']:.4f}"
        )
    return (
        f"{layer_name:15s} | "
        f"train_cv_auc={train_summary['auc_mean']:.4f} | "
        f"test_auc={test_summary['auc']:.4f} | "
        f"test_acc={test_summary['acc']:.4f} | "
        f"test_f1={test_summary['f1']:.4f}"
    )


def evaluate_layer_dict(layer_features, y, groups):
    results = {}
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
    return results


def build_split_specs(y, negative_type):
    split_specs = [("overall", np.ones(len(y), dtype=bool))]

    for neg_name in ("random_mismatch", "hard_mismatch"):
        mask = (negative_type == "positive") | (negative_type == neg_name)
        if np.any(mask & (y == 1)) and np.any(mask & (y == 0)):
            split_specs.append((neg_name, mask))

    return split_specs


def build_report_text(results_by_split, split_stats):
    lines = []
    lines.append(f"feature_path: {DEFAULT_FEATURE_PATH}")
    lines.append(f"result_path: {DEFAULT_RESULT_PATH}")
    for split_name, stats in split_stats.items():
        lines.append("")
        lines.append(f"===== split: {split_name} =====")
        lines.append(f"num_samples: {stats['num_samples']}")
        lines.append(f"positive: {stats['positive']}")
        lines.append(f"negative: {stats['negative']}")
        lines.append(f"num_groups: {stats['num_groups']}")
        lines.append(f"majority baseline acc: {stats['majority_acc']:.4f}")
        lines.append("random baseline auc: 0.5000")
        lines.append("")
        lines.append("===== summary sorted by auc =====")
        sorted_items = sorted(
            results_by_split[split_name].items(),
            key=lambda item: get_summary_metric(item[1])["auc_mean"],
            reverse=True,
        )
        for layer_name, metric_dict in sorted_items:
            lines.append(format_metric_line(layer_name, metric_dict))
        lines.append("================================")
    return "\n".join(lines) + "\n"


def build_holdout_report_text(results_by_split, split_stats, args, result_path):
    lines = []
    lines.append(f"train_feature_path: {args.feature_path}")
    lines.append(f"test_feature_path: {args.test_feature_path}")
    lines.append(f"result_path: {result_path}")
    lines.append(f"n_splits: {args.n_splits}")
    lines.append(f"random_state: {args.random_state}")
    for split_name, stats in split_stats.items():
        lines.append("")
        lines.append(f"===== split: {split_name} =====")
        lines.append(f"train_num_samples: {stats['train_num_samples']}")
        lines.append(f"train_positive: {stats['train_positive']}")
        lines.append(f"train_negative: {stats['train_negative']}")
        lines.append(f"train_num_groups: {stats['train_num_groups']}")
        lines.append(f"test_num_samples: {stats['test_num_samples']}")
        lines.append(f"test_positive: {stats['test_positive']}")
        lines.append(f"test_negative: {stats['test_negative']}")
        lines.append(f"test_num_groups: {stats['test_num_groups']}")
        lines.append(f"test_majority_baseline_acc: {stats['test_majority_acc']:.4f}")
        lines.append("test_random_baseline_auc: 0.5000")
        lines.append("")
        lines.append("===== summary sorted by test_auc =====")
        sorted_items = sorted(
            results_by_split[split_name].items(),
            key=lambda item: get_nested_summary_metric(item[1], "test_holdout")["auc"],
            reverse=True,
        )
        for layer_name, metric_dict in sorted_items:
            lines.append(format_holdout_metric_line(layer_name, metric_dict))
        lines.append("====================================")
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    result_path, report_path = resolve_output_paths(args)

    print(f"Loading feature file from: {args.feature_path}")
    layer_features, labels, meta, feature_mode = load_feature_bundle(args.feature_path)

    y = labels["coherence_label"].numpy().astype(np.int64)
    groups = build_groups(meta)
    negative_type = get_negative_type_array(meta, expected_len=len(y))

    if args.test_feature_path is None:
        print("num samples =", len(y))
        print("positive =", int(y.sum()))
        print("negative =", int(len(y) - y.sum()))
        print("num groups =", len(np.unique(groups)))
        print("feature_mode =", feature_mode)
        unique_negative_types, counts = np.unique(negative_type, return_counts=True)
        print("negative_type stats:")
        for name, count in zip(unique_negative_types.tolist(), counts.tolist()):
            print(f"  {name}: {count}")

        majority_acc = max(np.mean(y == 0), np.mean(y == 1))
        print(f"majority baseline acc = {majority_acc:.4f}")
        print("random baseline auc ≈ 0.5000")

        results_by_split = {}
        split_stats = {}
        split_specs = build_split_specs(y, negative_type)

        print("\n===== training coherence probes (StratifiedGroupKFold by seq_id/source_index) =====")
        for split_name, mask in split_specs:
            y_split = y[mask]
            groups_split = groups[mask]
            majority_acc_split = max(np.mean(y_split == 0), np.mean(y_split == 1))
            split_stats[split_name] = {
                "num_samples": int(mask.sum()),
                "positive": int(y_split.sum()),
                "negative": int(len(y_split) - y_split.sum()),
                "num_groups": int(np.unique(groups_split).size),
                "majority_acc": float(majority_acc_split),
            }

            print(f"\n===== split: {split_name} =====")
            print("num samples =", split_stats[split_name]["num_samples"])
            print("positive =", split_stats[split_name]["positive"])
            print("negative =", split_stats[split_name]["negative"])
            print("num groups =", split_stats[split_name]["num_groups"])
            print(f"majority baseline acc = {majority_acc_split:.4f}")

            layer_features_split = {
                layer_name: feats[mask]
                for layer_name, feats in layer_features.items()
            }
            results_by_split[split_name] = evaluate_layer_dict(
                layer_features=layer_features_split,
                y=y_split,
                groups=groups_split,
            )

        torch.save(
            {
                "results_by_split": results_by_split,
                "split_stats": split_stats,
            },
            result_path,
        )
        print(f"\nSaved coherence probe results to: {result_path}")

        report_text = build_report_text(results_by_split, split_stats)
        report_text = report_text.replace(DEFAULT_FEATURE_PATH, args.feature_path).replace(DEFAULT_RESULT_PATH, result_path)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_text)
        print(f"Saved coherence probe report to: {report_path}")

        for split_name, split_results in results_by_split.items():
            print(f"\n===== summary sorted by auc | split: {split_name} =====")
            sorted_items = sorted(
                split_results.items(),
                key=lambda item: get_summary_metric(item[1])["auc_mean"],
                reverse=True,
            )
            for layer_name, metric_dict in sorted_items:
                print(format_metric_line(layer_name, metric_dict))
            print("================================")
        return

    print(f"Loading held-out test feature file from: {args.test_feature_path}")
    test_layer_features, test_labels, test_meta, test_feature_mode = load_feature_bundle(args.test_feature_path)
    if feature_mode != test_feature_mode:
        raise ValueError(
            f"feature_mode mismatch: train={feature_mode}, test={test_feature_mode}. "
            "Train/test features must be extracted with the same mode."
        )
    if sorted(layer_features.keys()) != sorted(test_layer_features.keys()):
        raise KeyError("Train/test layer mismatch.")

    y_test_all = test_labels["coherence_label"].numpy().astype(np.int64)
    negative_type_test = get_negative_type_array(test_meta, expected_len=len(y_test_all))
    groups_test_all = get_groups_for_stats(test_meta)

    print("train_num_samples =", len(y))
    print("test_num_samples =", len(y_test_all))
    print("train_num_groups =", len(np.unique(groups)))
    print("test_num_groups =", len(np.unique(groups_test_all)))
    print("feature_mode =", feature_mode)

    results_by_split = {}
    split_stats = {}
    train_split_specs = build_split_specs(y, negative_type)
    test_split_specs = {name: mask for name, mask in build_split_specs(y_test_all, negative_type_test)}

    print("\n===== training coherence probes (train CV + held-out test) =====")
    for split_name, train_mask in train_split_specs:
        if split_name not in test_split_specs:
            continue
        test_mask = test_split_specs[split_name]
        y_train = y[train_mask]
        y_test = y_test_all[test_mask]
        groups_train = groups[train_mask]
        groups_test = groups_test_all[test_mask]

        split_stats[split_name] = {
            "train_num_samples": int(train_mask.sum()),
            "train_positive": int(y_train.sum()),
            "train_negative": int(len(y_train) - y_train.sum()),
            "train_num_groups": int(np.unique(groups_train).size),
            "test_num_samples": int(test_mask.sum()),
            "test_positive": int(y_test.sum()),
            "test_negative": int(len(y_test) - y_test.sum()),
            "test_num_groups": int(np.unique(groups_test).size),
            "test_majority_acc": float(max(np.mean(y_test == 0), np.mean(y_test == 1))),
        }

        print(f"\n===== split: {split_name} =====")
        print("train_num_samples =", split_stats[split_name]["train_num_samples"])
        print("test_num_samples =", split_stats[split_name]["test_num_samples"])
        print("train_num_groups =", split_stats[split_name]["train_num_groups"])
        print("test_num_groups =", split_stats[split_name]["test_num_groups"])

        split_results = {}
        for layer_name, feats in layer_features.items():
            X_train = feats.numpy().astype(np.float32)[train_mask]
            X_test = test_layer_features[layer_name].numpy().astype(np.float32)[test_mask]
            if X_train.ndim not in (2, 3):
                raise ValueError(f"Layer {layer_name}: expected 2D or 3D features, got {X_train.shape}")
            if X_train.ndim != X_test.ndim or X_train.shape[1:] != X_test.shape[1:]:
                raise ValueError(f"Layer {layer_name}: train/test feature shape mismatch {X_train.shape} vs {X_test.shape}")

            if X_train.ndim == 2:
                train_cv = evaluate_one_layer(
                    X=X_train,
                    y=y_train,
                    groups=groups_train,
                    n_splits=args.n_splits,
                    random_state=args.random_state,
                )
                test_holdout = evaluate_holdout_layer(
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    random_state=args.random_state,
                )
            else:
                print(f"{layer_name:15s} | tokenwise layer with {X_train.shape[1]} tokens")
                train_cv = evaluate_tokenwise_layer(
                    layer_name=layer_name,
                    X=X_train,
                    y=y_train,
                    groups=groups_train,
                    n_splits=args.n_splits,
                    random_state=args.random_state,
                )
                best_token = train_cv["best_token"]
                test_holdout = evaluate_holdout_layer(
                    X_train=X_train[:, best_token["token_idx"], :],
                    y_train=y_train,
                    X_test=X_test[:, best_token["token_idx"], :],
                    y_test=y_test,
                    random_state=args.random_state,
                )
                test_holdout["token_idx"] = int(best_token["token_idx"])
                test_holdout["token_name"] = best_token["token_name"]

            layer_result = {
                "train_cv": train_cv,
                "test_holdout": test_holdout,
            }
            split_results[layer_name] = layer_result
            print(format_holdout_metric_line(layer_name, layer_result))
        results_by_split[split_name] = split_results

    torch.save(
        {
            "results_by_split": results_by_split,
            "split_stats": split_stats,
        },
        result_path,
    )
    print(f"\nSaved coherence holdout probe results to: {result_path}")

    report_text = build_holdout_report_text(results_by_split, split_stats, args, result_path)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"Saved coherence holdout probe report to: {report_path}")

    for split_name, split_results in results_by_split.items():
        print(f"\n===== summary sorted by test_auc | split: {split_name} =====")
        sorted_items = sorted(
            split_results.items(),
            key=lambda item: get_nested_summary_metric(item[1], "test_holdout")["auc"],
            reverse=True,
        )
        for layer_name, metric_dict in sorted_items:
            print(format_holdout_metric_line(layer_name, metric_dict))
        print("====================================")


if __name__ == "__main__":
    main()
