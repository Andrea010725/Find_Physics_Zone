import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = os.path.abspath(__file__)
ROOT = "/".join(ROOT.split("/")[:-2])
WORKSPACE_ROOT = os.path.dirname(ROOT)


def resolve_repo_path(*parts):
    local_path = os.path.join(ROOT, *parts)
    workspace_path = os.path.join(WORKSPACE_ROOT, *parts)
    if os.path.exists(local_path):
        return local_path
    return workspace_path


DATA_ROOT = resolve_repo_path("data")
DEFAULT_FEATURE_PATH = os.path.join(DATA_ROOT, "physics_features.pt")
DEFAULT_CANDIDATE_PATH = os.path.join(DATA_ROOT, "target_layer_candidates.json")
DEFAULT_RESULT_JSON = os.path.join(DATA_ROOT, "depth_selective_attnres_value_results.json")
DEFAULT_RESULT_TXT = os.path.join(DATA_ROOT, "depth_selective_attnres_value_results.txt")

DEFAULT_TARGETS = [
    "future_speed",
    "future_forward_progress",
    "future_yaw_rate",
    "future_delta_yaw",
    "future_lateral_offset",
    "future_speed_delta",
    "future_acc",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_path", type=str, default=DEFAULT_FEATURE_PATH)
    parser.add_argument("--candidate_path", type=str, default=DEFAULT_CANDIDATE_PATH)
    parser.add_argument("--result_json", type=str, default=DEFAULT_RESULT_JSON)
    parser.add_argument("--result_txt", type=str, default=DEFAULT_RESULT_TXT)
    parser.add_argument("--targets", nargs="*", default=DEFAULT_TARGETS)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--entropy_lambda", type=float, default=1e-3)
    parser.add_argument("--attn_hidden_dim", type=int, default=256)
    parser.add_argument("--attn_gate_init_logit", type=float, default=-1.5)
    parser.add_argument("--value_hidden_dim", type=int, default=256)
    parser.add_argument("--value_gate_init_logit", type=float, default=-1.5)
    parser.add_argument("--value_reward_lambda", type=float, default=0.2)
    parser.add_argument("--value_reward_temperature", type=float, default=0.25)
    parser.add_argument("--ridge_alpha", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pearson_corr(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    corr = pearson_corr(y_true, y_pred)
    target_std = float(np.std(y_true))
    normalized_mae = float(mae / target_std) if target_std >= 1e-12 else 0.0
    return {
        "mse": float(mse),
        "mae": float(mae),
        "normalized_mae": normalized_mae,
        "r2": float(r2),
        "pearson": float(corr),
    }


class DepthSelectiveReadout(nn.Module):
    def __init__(self, num_layers, in_dim):
        super().__init__()
        self.layer_logits = nn.Parameter(torch.zeros(num_layers))
        self.layer_weight = nn.Parameter(torch.zeros(num_layers, in_dim))
        self.layer_bias = nn.Parameter(torch.zeros(num_layers))
        nn.init.normal_(self.layer_weight, mean=0.0, std=0.02)

    def forward(self, x):
        # x: [N, L, D]
        per_layer_pred = torch.einsum("nld,ld->nl", x, self.layer_weight) + self.layer_bias
        layer_weights = torch.softmax(self.layer_logits, dim=0)
        y_pred = torch.sum(per_layer_pred * layer_weights[None, :], dim=1)
        return y_pred, layer_weights, per_layer_pred


class AttnResidualReadout(nn.Module):
    """AttnRes-style readout: y = y_base + gate * y_attn_delta."""

    def __init__(self, num_layers, in_dim, hidden_dim=256, gate_init_logit=-1.5):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.base_head = nn.Linear(in_dim, 1)
        self.q_proj = nn.Linear(in_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(in_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(in_dim, hidden_dim, bias=False)
        self.delta_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.gate_logit = nn.Parameter(torch.tensor(float(gate_init_logit)))

    def forward(self, x):
        # x: [N, L, D], layer 0 is the best single-layer baseline.
        x_base = x[:, 0, :]  # [N, D]
        y_base = self.base_head(x_base).squeeze(-1)  # [N]

        q = self.q_proj(x_base).unsqueeze(1)  # [N, 1, H]
        k = self.k_proj(x)  # [N, L, H]
        v = self.v_proj(x)  # [N, L, H]

        scale = float(self.hidden_dim) ** -0.5
        attn_logits = torch.sum(q * k, dim=-1) * scale  # [N, L]
        attn_weights = torch.softmax(attn_logits, dim=1)
        context = torch.sum(attn_weights.unsqueeze(-1) * v, dim=1)  # [N, H]
        y_delta = self.delta_head(context).squeeze(-1)  # [N]

        gate = torch.sigmoid(self.gate_logit)
        y_pred = y_base + gate * y_delta
        return y_pred, attn_weights, gate, y_base, y_delta


class ValueFunctionReadout(nn.Module):
    """Value-guided residual readout.

    The best single layer (candidate index 0) supplies a stable base prediction.
    Each candidate layer also proposes a residual delta and a value score.
    The value function is trained to assign more mass to layers whose residual
    would have yielded lower supervised error on the current sample.
    """

    def __init__(self, num_layers, in_dim, hidden_dim=256, gate_init_logit=-1.5):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.base_head = nn.Linear(in_dim, 1)
        self.delta_head = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.value_head = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.gate_logit = nn.Parameter(torch.tensor(float(gate_init_logit)))

    def forward(self, x):
        # x: [N, L, D], layer 0 is the best single-layer baseline.
        x_base = x[:, 0, :]  # [N, D]
        y_base = self.base_head(x_base).squeeze(-1)  # [N]

        layer_delta = self.delta_head(x).squeeze(-1)  # [N, L]
        value_logits = self.value_head(x).squeeze(-1)  # [N, L]
        layer_weights = torch.softmax(value_logits, dim=1)

        y_delta = torch.sum(layer_weights * layer_delta, dim=1)  # [N]
        gate = torch.sigmoid(self.gate_logit)
        y_pred = y_base + gate * y_delta
        return y_pred, layer_weights, gate, y_base, y_delta, layer_delta, value_logits


def build_groups(meta):
    if "seq_id" not in meta:
        raise KeyError("Feature file is missing meta['seq_id'].")
    groups = meta["seq_id"]
    if isinstance(groups, torch.Tensor):
        groups = groups.cpu().numpy()
    groups = np.asarray(groups)
    unique_groups = np.unique(groups)
    if unique_groups.size < 2:
        raise ValueError("GroupKFold needs at least 2 distinct seq_id groups.")
    return groups


def standardize_per_layer(X_train, X_test):
    # X shape: [N, L, D]
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (X_train - mean) / std, (X_test - mean) / std


def evaluate_ridge_baseline(X, y, groups, n_splits, alpha):
    splitter = GroupKFold(n_splits=n_splits)
    y_true_all = []
    y_pred_all = []
    for train_idx, test_idx in splitter.split(X, y, groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=alpha)),
            ]
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_true_all.extend(y_test.tolist())
        y_pred_all.extend(y_pred.tolist())
    return regression_metrics(y_true_all, y_pred_all)


def evaluate_depth_selective(
    X_layers,
    y,
    groups,
    n_splits,
    device,
    epochs,
    lr,
    weight_decay,
    entropy_lambda,
):
    splitter = GroupKFold(n_splits=n_splits)
    y_true_all = []
    y_pred_all = []
    split_weights = []
    split_losses = []

    for split_id, (train_idx, test_idx) in enumerate(splitter.split(X_layers, y, groups)):
        X_train = X_layers[train_idx]
        X_test = X_layers[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        X_train_std, X_test_std = standardize_per_layer(X_train, X_test)
        y_mean = y_train.mean()
        y_std = y_train.std()
        if y_std < 1e-6:
            y_std = 1.0

        y_train_norm = (y_train - y_mean) / y_std

        x_train_t = torch.from_numpy(X_train_std).to(device=device, dtype=torch.float32)
        x_test_t = torch.from_numpy(X_test_std).to(device=device, dtype=torch.float32)
        y_train_t = torch.from_numpy(y_train_norm).to(device=device, dtype=torch.float32)

        num_layers = X_layers.shape[1]
        in_dim = X_layers.shape[2]
        model = DepthSelectiveReadout(num_layers=num_layers, in_dim=in_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        model.train()
        last_loss = None
        for _ in range(epochs):
            optimizer.zero_grad(set_to_none=True)
            pred_norm, layer_weights, _ = model(x_train_t)
            mse = F.mse_loss(pred_norm, y_train_t)
            entropy = -torch.sum(layer_weights * torch.log(layer_weights + 1e-8))
            loss = mse + entropy_lambda * entropy
            loss.backward()
            optimizer.step()
            last_loss = float(loss.detach().cpu())

        model.eval()
        with torch.no_grad():
            pred_norm, layer_weights, _ = model(x_test_t)
            pred = pred_norm.cpu().numpy() * y_std + y_mean

        y_true_all.extend(y_test.tolist())
        y_pred_all.extend(pred.tolist())
        split_weights.append(layer_weights.detach().cpu().numpy().tolist())
        split_losses.append(
            {
                "split_id": int(split_id),
                "final_loss": float(last_loss),
            }
        )

    metrics = regression_metrics(y_true_all, y_pred_all)
    weights_arr = np.asarray(split_weights, dtype=np.float32)  # [S, L]
    metrics["layer_weight_mean"] = weights_arr.mean(axis=0).tolist()
    metrics["layer_weight_std"] = weights_arr.std(axis=0).tolist()
    metrics["split_layer_weights"] = split_weights
    metrics["split_losses"] = split_losses
    return metrics


def evaluate_attn_residual(
    X_layers,
    y,
    groups,
    n_splits,
    device,
    epochs,
    lr,
    weight_decay,
    entropy_lambda,
    attn_hidden_dim,
    attn_gate_init_logit,
):
    splitter = GroupKFold(n_splits=n_splits)
    y_true_all = []
    y_pred_all = []
    split_attn_means = []
    split_gates = []
    split_losses = []

    for split_id, (train_idx, test_idx) in enumerate(splitter.split(X_layers, y, groups)):
        X_train = X_layers[train_idx]
        X_test = X_layers[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        X_train_std, X_test_std = standardize_per_layer(X_train, X_test)
        y_mean = y_train.mean()
        y_std = y_train.std()
        if y_std < 1e-6:
            y_std = 1.0
        y_train_norm = (y_train - y_mean) / y_std

        x_train_t = torch.from_numpy(X_train_std).to(device=device, dtype=torch.float32)
        x_test_t = torch.from_numpy(X_test_std).to(device=device, dtype=torch.float32)
        y_train_t = torch.from_numpy(y_train_norm).to(device=device, dtype=torch.float32)

        num_layers = X_layers.shape[1]
        in_dim = X_layers.shape[2]
        model = AttnResidualReadout(
            num_layers=num_layers,
            in_dim=in_dim,
            hidden_dim=attn_hidden_dim,
            gate_init_logit=attn_gate_init_logit,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        model.train()
        last_loss = None
        for _ in range(epochs):
            optimizer.zero_grad(set_to_none=True)
            pred_norm, attn_weights, gate, _, _ = model(x_train_t)
            mse = F.mse_loss(pred_norm, y_train_t)
            attn_entropy = -torch.sum(attn_weights * torch.log(attn_weights + 1e-8), dim=1).mean()
            loss = mse + entropy_lambda * attn_entropy
            loss.backward()
            optimizer.step()
            last_loss = float(loss.detach().cpu())

        model.eval()
        with torch.no_grad():
            pred_norm, attn_weights, gate, _, _ = model(x_test_t)
            pred = pred_norm.cpu().numpy() * y_std + y_mean
            attn_mean = attn_weights.mean(dim=0).cpu().numpy()
            gate_scalar = float(gate.detach().cpu())

        y_true_all.extend(y_test.tolist())
        y_pred_all.extend(pred.tolist())
        split_attn_means.append(attn_mean.tolist())
        split_gates.append(gate_scalar)
        split_losses.append(
            {
                "split_id": int(split_id),
                "final_loss": float(last_loss),
            }
        )

    metrics = regression_metrics(y_true_all, y_pred_all)
    attn_arr = np.asarray(split_attn_means, dtype=np.float32)  # [S, L]
    gate_arr = np.asarray(split_gates, dtype=np.float32)  # [S]
    metrics["layer_attention_mean"] = attn_arr.mean(axis=0).tolist()
    metrics["layer_attention_std"] = attn_arr.std(axis=0).tolist()
    metrics["gate_mean"] = float(gate_arr.mean())
    metrics["gate_std"] = float(gate_arr.std())
    metrics["split_layer_attention"] = split_attn_means
    metrics["split_gates"] = split_gates
    metrics["split_losses"] = split_losses
    return metrics


def evaluate_value_function(
    X_layers,
    y,
    groups,
    n_splits,
    device,
    epochs,
    lr,
    weight_decay,
    entropy_lambda,
    value_hidden_dim,
    value_gate_init_logit,
    value_reward_lambda,
    value_reward_temperature,
):
    splitter = GroupKFold(n_splits=n_splits)
    y_true_all = []
    y_pred_all = []
    split_weight_means = []
    split_value_means = []
    split_gates = []
    split_losses = []

    reward_temperature = max(float(value_reward_temperature), 1e-6)

    for split_id, (train_idx, test_idx) in enumerate(splitter.split(X_layers, y, groups)):
        X_train = X_layers[train_idx]
        X_test = X_layers[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        X_train_std, X_test_std = standardize_per_layer(X_train, X_test)
        y_mean = y_train.mean()
        y_std = y_train.std()
        if y_std < 1e-6:
            y_std = 1.0
        y_train_norm = (y_train - y_mean) / y_std

        x_train_t = torch.from_numpy(X_train_std).to(device=device, dtype=torch.float32)
        x_test_t = torch.from_numpy(X_test_std).to(device=device, dtype=torch.float32)
        y_train_t = torch.from_numpy(y_train_norm).to(device=device, dtype=torch.float32)

        num_layers = X_layers.shape[1]
        in_dim = X_layers.shape[2]
        model = ValueFunctionReadout(
            num_layers=num_layers,
            in_dim=in_dim,
            hidden_dim=value_hidden_dim,
            gate_init_logit=value_gate_init_logit,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        model.train()
        last_loss = None
        last_reward_loss = None
        for _ in range(epochs):
            optimizer.zero_grad(set_to_none=True)
            pred_norm, layer_weights, gate, y_base, _, layer_delta, value_logits = model(x_train_t)
            mse = F.mse_loss(pred_norm, y_train_t)

            with torch.no_grad():
                per_layer_pred = y_base.unsqueeze(1) + layer_delta
                sq_errors = torch.square(per_layer_pred - y_train_t.unsqueeze(1))
                per_layer_reward = -sq_errors

                # 使用更稳定的 Top-K 掩码
                k = max(1, num_layers // 2)
                topk_vals, _ = torch.topk(per_layer_reward, k=k, dim=1)
                min_topk = topk_vals[:, -1:]
                # 这里的 -1e4 比 -1e9 更稳定，防止 softmax 溢出
                masked_reward = torch.where(
                    per_layer_reward >= min_topk,
                    per_layer_reward,
                    torch.tensor(-1e4).to(per_layer_reward.device)
                )

                # 优化温度系数逻辑，确保稳定性
                # 这里的 reward_temperature 是外部传入的参数
                target_policy = torch.softmax(
                    masked_reward / reward_temperature,
                    dim=1,
                )

            # 统一使用传入的 reward_temperature
            value_log_probs = F.log_softmax(value_logits / reward_temperature, dim=1)
            reward_loss = F.kl_div(value_log_probs, target_policy, reduction="batchmean")
            weight_entropy = -torch.sum(
                layer_weights * torch.log(layer_weights + 1e-8),
                dim=1,
            ).mean()

            loss = mse + value_reward_lambda * reward_loss + entropy_lambda * weight_entropy
            loss.backward()
            optimizer.step()
            last_loss = float(loss.detach().cpu())
            last_reward_loss = float(reward_loss.detach().cpu())

        model.eval()
        with torch.no_grad():
            pred_norm, layer_weights, gate, _, _, _, value_logits = model(x_test_t)
            pred = pred_norm.cpu().numpy() * y_std + y_mean
            weight_mean = layer_weights.mean(dim=0).cpu().numpy()
            value_mean = value_logits.mean(dim=0).cpu().numpy()
            gate_scalar = float(gate.detach().cpu())

        y_true_all.extend(y_test.tolist())
        y_pred_all.extend(pred.tolist())
        split_weight_means.append(weight_mean.tolist())
        split_value_means.append(value_mean.tolist())
        split_gates.append(gate_scalar)
        split_losses.append(
            {
                "split_id": int(split_id),
                "final_loss": float(last_loss),
                "reward_loss": float(last_reward_loss),
            }
        )

    metrics = regression_metrics(y_true_all, y_pred_all)
    weight_arr = np.asarray(split_weight_means, dtype=np.float32)  # [S, L]
    value_arr = np.asarray(split_value_means, dtype=np.float32)  # [S, L]
    gate_arr = np.asarray(split_gates, dtype=np.float32)  # [S]
    metrics["layer_weight_mean"] = weight_arr.mean(axis=0).tolist()
    metrics["layer_weight_std"] = weight_arr.std(axis=0).tolist()
    metrics["layer_value_mean"] = value_arr.mean(axis=0).tolist()
    metrics["layer_value_std"] = value_arr.std(axis=0).tolist()
    metrics["gate_mean"] = float(gate_arr.mean())
    metrics["gate_std"] = float(gate_arr.std())
    metrics["split_layer_weights"] = split_weight_means
    metrics["split_layer_values"] = split_value_means
    metrics["split_gates"] = split_gates
    metrics["split_losses"] = split_losses
    return metrics


def select_target_layers(candidates_payload, target_name, available_layers):
    info = candidates_payload["targets"].get(target_name)
    if info is None:
        return []
    layer_names = info.get("recommended_layer_names", [])
    return [name for name in layer_names if name in available_layers]


def build_txt_report(result_payload):
    lines = []
    lines.append(f"feature_path: {result_payload['feature_path']}")
    lines.append(f"candidate_path: {result_payload['candidate_path']}")
    lines.append(f"num_samples: {result_payload['num_samples']}")
    lines.append(f"num_groups: {result_payload['num_groups']}")
    lines.append("")

    for target, item in result_payload["per_target"].items():
        lines.append(f"===== {target} =====")
        lines.append("candidates: " + ", ".join(item["candidate_layers"]))
        lines.append(
            "single_layer_ridge: "
            f"r2={item['single_layer_ridge']['r2']:.6f} | "
            f"pearson={item['single_layer_ridge']['pearson']:.6f} | "
            f"nmae={item['single_layer_ridge']['normalized_mae']:.6f}"
        )
        lines.append(
            "mean_layer_ridge:   "
            f"r2={item['mean_layer_ridge']['r2']:.6f} | "
            f"pearson={item['mean_layer_ridge']['pearson']:.6f} | "
            f"nmae={item['mean_layer_ridge']['normalized_mae']:.6f}"
        )
        lines.append(
            "depth_selective:    "
            f"r2={item['depth_selective']['r2']:.6f} | "
            f"pearson={item['depth_selective']['pearson']:.6f} | "
            f"nmae={item['depth_selective']['normalized_mae']:.6f}"
        )
        lines.append(
            "attn_residual:      "
            f"r2={item['attn_residual']['r2']:.6f} | "
            f"pearson={item['attn_residual']['pearson']:.6f} | "
            f"nmae={item['attn_residual']['normalized_mae']:.6f} | "
            f"gate_mean={item['attn_residual']['gate_mean']:.6f}"
        )
        lines.append(
            "value_function:     "
            f"r2={item['value_function']['r2']:.6f} | "
            f"pearson={item['value_function']['pearson']:.6f} | "
            f"nmae={item['value_function']['normalized_mae']:.6f} | "
            f"gate_mean={item['value_function']['gate_mean']:.6f}"
        )
        lines.append(
            "delta_depth_vs_single: "
            f"delta_r2={item['delta_depth_vs_single']['r2']:.6f} | "
            f"delta_nmae={item['delta_depth_vs_single']['normalized_mae']:.6f}"
        )
        lines.append(
            "delta_attn_vs_single:  "
            f"delta_r2={item['delta_attn_vs_single']['r2']:.6f} | "
            f"delta_nmae={item['delta_attn_vs_single']['normalized_mae']:.6f}"
        )
        lines.append(
            "delta_value_vs_single: "
            f"delta_r2={item['delta_value_vs_single']['r2']:.6f} | "
            f"delta_nmae={item['delta_value_vs_single']['normalized_mae']:.6f}"
        )
        ranked = sorted(
            zip(item["candidate_layers"], item["depth_selective"]["layer_weight_mean"]),
            key=lambda z: z[1],
            reverse=True,
        )
        lines.append(
            "depth_weights_mean: "
            + ", ".join([f"{name}:{weight:.4f}" for name, weight in ranked])
        )
        ranked_attn = sorted(
            zip(item["candidate_layers"], item["attn_residual"]["layer_attention_mean"]),
            key=lambda z: z[1],
            reverse=True,
        )
        lines.append(
            "attn_weights_mean:  "
            + ", ".join([f"{name}:{weight:.4f}" for name, weight in ranked_attn])
        )
        ranked_value = sorted(
            zip(item["candidate_layers"], item["value_function"]["layer_weight_mean"]),
            key=lambda z: z[1],
            reverse=True,
        )
        lines.append(
            "value_weights_mean: "
            + ", ".join([f"{name}:{weight:.4f}" for name, weight in ranked_value])
        )
        if item.get("note"):
            lines.append("note: " + item["note"])
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    print("loading feature file:", args.feature_path)
    feat_payload = torch.load(args.feature_path, map_location="cpu")
    layer_features = feat_payload["layer_features"]
    labels = feat_payload["labels"]
    meta = feat_payload.get("meta", {})

    print("loading candidates:", args.candidate_path)
    with open(args.candidate_path, "r", encoding="utf-8") as f:
        candidate_payload = json.load(f)

    groups = build_groups(meta)
    unique_groups = np.unique(groups)
    n_splits = min(args.n_splits, unique_groups.size)
    if n_splits < 2:
        raise ValueError(f"n_splits must be >=2, got {n_splits}.")

    print("num_samples =", len(groups))
    print("num_groups =", len(unique_groups))
    print("n_splits =", n_splits)
    print("device =", device)

    available_layers = {
        name for name, tensor in layer_features.items() if tensor.ndim == 2
    }

    per_target = {}
    for target in args.targets:
        if target not in labels:
            print(f"[skip] target not in labels: {target}")
            continue

        candidate_layers = select_target_layers(candidate_payload, target, available_layers)
        if not candidate_layers:
            print(f"[skip] no candidate layers available for target: {target}")
            continue

        y = labels[target].numpy().astype(np.float32)
        X_layers = np.stack(
            [layer_features[name].numpy().astype(np.float32) for name in candidate_layers],
            axis=1,
        )  # [N, L, D]

        print(f"\n--- target={target} ---")
        print("candidate_layers =", candidate_layers)
        print("X_layers shape =", X_layers.shape)

        single_layer_metrics = evaluate_ridge_baseline(
            X=X_layers[:, 0, :],
            y=y,
            groups=groups,
            n_splits=n_splits,
            alpha=args.ridge_alpha,
        )
        mean_layer_metrics = evaluate_ridge_baseline(
            X=X_layers.mean(axis=1),
            y=y,
            groups=groups,
            n_splits=n_splits,
            alpha=args.ridge_alpha,
        )
        depth_metrics = evaluate_depth_selective(
            X_layers=X_layers,
            y=y,
            groups=groups,
            n_splits=n_splits,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            entropy_lambda=args.entropy_lambda,
        )
        attn_metrics = evaluate_attn_residual(
            X_layers=X_layers,
            y=y,
            groups=groups,
            n_splits=n_splits,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            entropy_lambda=args.entropy_lambda,
            attn_hidden_dim=args.attn_hidden_dim,
            attn_gate_init_logit=args.attn_gate_init_logit,
        )
        value_metrics = evaluate_value_function(
            X_layers=X_layers,
            y=y,
            groups=groups,
            n_splits=n_splits,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            entropy_lambda=args.entropy_lambda,
            value_hidden_dim=args.value_hidden_dim,
            value_gate_init_logit=args.value_gate_init_logit,
            value_reward_lambda=args.value_reward_lambda,
            value_reward_temperature=args.value_reward_temperature,
        )

        note = candidate_payload["targets"].get(target, {}).get("note", "")

        per_target[target] = {
            "candidate_layers": candidate_layers,
            "single_layer_ridge": single_layer_metrics,
            "mean_layer_ridge": mean_layer_metrics,
            "depth_selective": depth_metrics,
            "attn_residual": attn_metrics,
            "value_function": value_metrics,
            "delta_depth_vs_single": {
                "r2": float(depth_metrics["r2"] - single_layer_metrics["r2"]),
                "normalized_mae": float(
                    depth_metrics["normalized_mae"] - single_layer_metrics["normalized_mae"]
                ),
            },
            "delta_attn_vs_single": {
                "r2": float(attn_metrics["r2"] - single_layer_metrics["r2"]),
                "normalized_mae": float(
                    attn_metrics["normalized_mae"] - single_layer_metrics["normalized_mae"]
                ),
            },
            "delta_value_vs_single": {
                "r2": float(value_metrics["r2"] - single_layer_metrics["r2"]),
                "normalized_mae": float(
                    value_metrics["normalized_mae"] - single_layer_metrics["normalized_mae"]
                ),
            },
            "note": note,
        }

        print(
            "single_layer_ridge:",
            f"r2={single_layer_metrics['r2']:.6f}",
            f"nmae={single_layer_metrics['normalized_mae']:.6f}",
        )
        print(
            "mean_layer_ridge:  ",
            f"r2={mean_layer_metrics['r2']:.6f}",
            f"nmae={mean_layer_metrics['normalized_mae']:.6f}",
        )
        print(
            "depth_selective:   ",
            f"r2={depth_metrics['r2']:.6f}",
            f"nmae={depth_metrics['normalized_mae']:.6f}",
        )
        print(
            "attn_residual:    ",
            f"r2={attn_metrics['r2']:.6f}",
            f"nmae={attn_metrics['normalized_mae']:.6f}",
            f"gate={attn_metrics['gate_mean']:.4f}",
        )
        print(
            "value_function:   ",
            f"r2={value_metrics['r2']:.6f}",
            f"nmae={value_metrics['normalized_mae']:.6f}",
            f"gate={value_metrics['gate_mean']:.4f}",
        )

    result_payload = {
        "feature_path": args.feature_path,
        "candidate_path": args.candidate_path,
        "num_samples": int(len(groups)),
        "num_groups": int(len(unique_groups)),
        "n_splits": int(n_splits),
        "config": {
            "device": str(device),
            "epochs": int(args.epochs),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "entropy_lambda": float(args.entropy_lambda),
            "attn_hidden_dim": int(args.attn_hidden_dim),
            "attn_gate_init_logit": float(args.attn_gate_init_logit),
            "value_hidden_dim": int(args.value_hidden_dim),
            "value_gate_init_logit": float(args.value_gate_init_logit),
            "value_reward_lambda": float(args.value_reward_lambda),
            "value_reward_temperature": float(args.value_reward_temperature),
            "ridge_alpha": float(args.ridge_alpha),
            "seed": int(args.seed),
        },
        "per_target": per_target,
    }

    os.makedirs(os.path.dirname(args.result_json), exist_ok=True)
    with open(args.result_json, "w", encoding="utf-8") as f:
        json.dump(result_payload, f, ensure_ascii=False, indent=2)

    txt_report = build_txt_report(result_payload)
    with open(args.result_txt, "w", encoding="utf-8") as f:
        f.write(txt_report)

    print("\nSaved JSON:", args.result_json)
    print("Saved TXT :", args.result_txt)


if __name__ == "__main__":
    main()
