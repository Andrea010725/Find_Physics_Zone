# AGS Experiment 运行指南

本文档描述如何运行完整的 AGS (Action-Geometric State) probe 实验流程。

## 前置条件

确保以下文件/目录存在：
- `pretrained_models/world_model.pth`
- `pretrained_models/vqvae.pt`
- `data/seq_meta/*.json` 或 `data/seq_meta.json`
- `data/ego_meta/*.json`
- 对应的图像目录

## 完整流程

### 1. 构造 physics samples（含 AGS 标签）

```bash
cd /home/yanda/Find_Physics_Zone

# 基础运行（默认参数）
python tools/build_physics_dataset.py

# 推荐参数（与原实验保持一致）
python tools/build_physics_dataset.py \
  --data_root ./data \
  --json_root ./data \
  --save_root ./data \
  --condition_frames 15 \
  --future_horizon 1 \
  --stride 1 \
  --downsample_fps 5 \
  --downsample_size 16 \
  --h 256 \
  --w 512
```

**输出**：
- `data/physics_samples.pt`

**验证**：
- 检查输出统计中是否包含新增字段：
  - `future_forward_progress`
  - `future_lateral_offset`
  - `future_speed_delta`
  - `future_acc`

---

### 2. 提取 hidden-state features

#### 2.1 all_mean 模式（推荐先跑这个）

```bash
python tools/extract_zone_features.py \
  --task physics \
  --pool_mode all_mean \
  --device cuda:0 \
  --log_every 100
```

**输出**：
- `data/physics_features.pt`

**特点**：
- 每层输出一个 pooled 1536-d 向量
- 训练速度快
- 适合快速验证 AGS 标签是否正确进入流程

#### 2.2 tokenwise 模式（可选，用于精细分析）

```bash
python tools/extract_zone_features.py \
  --task physics \
  --pool_mode tokenwise \
  --device cuda:0 \
  --log_every 100 \
  --output_path ./data/physics_features_tokenwise.pt
```

**输出**：
- `data/physics_features_tokenwise.pt`

**特点**：
- 每层保留所有 token 的特征 `[N, L, C]`
- 可以定位哪些 token（yaw / pose / image patch）最承载 AGS
- 训练时间更长

#### 2.3 切片运行（调试/快速验证）

如果只想快速验证流程，可以只跑前 1000 个样本：

```bash
python tools/extract_zone_features.py \
  --task physics \
  --pool_mode all_mean \
  --device cuda:0 \
  --max_samples 1000 \
  --output_path ./data/physics_features_debug.pt
```

---

### 3. 训练 physics probe（含 AGS group summary）

#### 3.1 基础运行

```bash
python tools/train_physics_probe.py
```

**输出**：
- `data/physics_probe_results.pt`
- `data/physics_probe_results.txt`

**内容**：
- 每个 target 的 layerwise probe 结果
- MAE / normalized MAE / R² / Pearson
- **AGS group summary**（按 mean R² 排序）

#### 3.2 如果用了 tokenwise features

修改 `train_physics_probe.py` 开头的路径：

```python
FEATURE_PATH = os.path.join(data_root, "physics_features_tokenwise.pt")
RESULT_PATH = os.path.join(data_root, "physics_probe_results_tokenwise.pt")
REPORT_PATH = os.path.join(data_root, "physics_probe_results_tokenwise.txt")
```

或者直接在脚本里临时改，然后运行：

```bash
python tools/train_physics_probe.py
```

---

## 结果解读

### 单 target 结果

在 `physics_probe_results.txt` 中，每个 target 会有一节：

```
===== summary sorted by r2: future_forward_progress =====
ar_5            | mse=0.123456 | mae=0.234567 | nmae=0.345678 | r2=0.456789 | pearson=0.567890
next_state_hidden | mse=0.234567 | mae=0.345678 | nmae=0.456789 | r2=0.345678 | pearson=0.456789
...
==========================================
```

### AGS group summary

在文件末尾会有：

```
===== AGS GROUP SUMMARY SORTED BY MEAN R2 =====
ar_5            | mean_mse=0.123456 | mean_mae=0.234567 | mean_nmae=0.345678 | mean_r2=0.456789 | mean_pearson=0.567890
next_state_hidden | mean_mse=0.234567 | mean_mae=0.345678 | mean_nmae=0.456789 | mean_r2=0.345678 | mean_pearson=0.456789
...
==========================================================
```

**关键指标**：
- `mean_r2`：AGS 组内所有 target 的平均 R²
- `mean_normalized_mae`：AGS 组内所有 target 的平均 normalized MAE

**论证目标**：
- 如果 AGS 的 best layer 集中在 `next_state_hidden / ar_*`，说明更接近 next-state / policy-facing representation
- 如果 AGS 的 mean_r2 比单个 physics target 更稳定、更集中，说明这些层编码的是"一组共同的中间状态"，而不是 isolated scalar

---

## 快速验证流程（调试用）

如果只想快速验证改动是否自洽，可以跑一个最小流程：

```bash
# 1. 构造样本（只跑前几个序列）
python tools/build_physics_dataset.py --stride 10

# 2. 提取特征（只跑前 500 样本）
python tools/extract_zone_features.py \
  --task physics \
  --pool_mode all_mean \
  --max_samples 500 \
  --output_path ./data/physics_features_debug.pt

# 3. 训练 probe（需要临时改 FEATURE_PATH）
# 在 train_physics_probe.py 里把 FEATURE_PATH 改成 physics_features_debug.pt
python tools/train_physics_probe.py
```

---

## 常见问题

### Q1: `physics_samples.pt` 里没有新字段？

检查 `build_physics_dataset.py` 是否正确修改。运行后应该能看到类似输出：

```
future_forward_progress mean=0.123456, std=0.234567, min=-1.234567, max=2.345678
future_lateral_offset mean=0.012345, std=0.123456, min=-0.987654, max=0.876543
future_speed_delta mean=0.001234, std=0.098765, min=-0.543210, max=0.456789
future_acc mean=0.006172, std=0.493827, min=-2.716050, max=2.283950
```

### Q2: `physics_features.pt` 里没有新 label？

检查 `extract_zone_features.py` 的 `PHYSICS_LABEL_KEYS` 是否包含新字段。

可以用以下命令验证：

```python
import torch
data = torch.load("data/physics_features.pt")
print("labels keys:", sorted(data["labels"].keys()))
```

应该包含：
- `future_forward_progress`
- `future_lateral_offset`
- `future_speed_delta`
- `future_acc`

### Q3: probe 训练报错 `KeyError`？

检查 `train_physics_probe.py` 的 `TARGET_KEYS` 是否包含新字段。

### Q4: 想对比旧 physics probe 和新 AGS probe？

保留旧结果文件，然后用不同文件名保存新结果：

```python
# 在 train_physics_probe.py 里改成：
RESULT_PATH = os.path.join(data_root, "physics_probe_results_ags.pt")
REPORT_PATH = os.path.join(data_root, "physics_probe_results_ags.txt")
```

---

## 论文作图建议

### 图 1: AGS vs Physics layerwise R² 对比

横轴：layer name（按 time_space → next_state → ar 顺序）
纵轴：R²
曲线：
- 旧 physics target（如 `future_speed`）
- 新 AGS target（如 `future_forward_progress`）
- AGS group mean R²

**预期**：AGS 曲线在 `next_state_hidden / ar_*` 更集中、更高。

### 图 2: AGS group stability

横轴：layer name
纵轴：mean R² ± std（跨 AGS targets）

**预期**：如果 AGS 在某层的 std 更小，说明这层对 AGS 多维变量整体可读，更像"中间状态"。

### 表 1: Best layer 分布

| Target | Best Layer | R² | MAE | nMAE |
|--------|-----------|-----|-----|------|
| future_speed | ar_5 | 0.xx | 0.xx | 0.xx |
| future_forward_progress | ar_5 | 0.xx | 0.xx | 0.xx |
| ... | ... | ... | ... | ... |
| **AGS group** | **ar_5** | **0.xx** | **0.xx** | **0.xx** |

---

## 下一步扩展

如果要进一步增强论证，可以考虑：

1. **多 horizon 实验**：
   - 改 `--future_horizon 1` 为 `2 / 3 / 5`
   - 看 AGS 在不同 horizon 下的 best layer 是否稳定

2. **跨折稳定性分析**：
   - 从 `physics_probe_results.pt` 里读出每折的结果
   - 计算每层的 R² 标准差
   - 看 AGS 是否比单 target 更稳定

3. **tokenwise 精细分析**：
   - 用 tokenwise features
   - 看 yaw / pose / image token 谁最承载 AGS
   - 如果 pose token 最好，说明 AGS 确实更接近 ego motion planning interface

---

## 联系与反馈

如果运行中遇到问题，或者需要进一步定制实验，可以：
- 检查 `README_dw.md` 中的流程说明
- 查看 `data/physics_probe_results.txt` 的完整输出
- 对比新旧结果文件的 best layer 分布
