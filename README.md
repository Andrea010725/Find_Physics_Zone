# Testing Guide

这份文档只描述当前 `Find_Physics_Zone` 代码实际实现的流程

## 1. 先明确这个目录在做什么

- 根目录 `DrivingWorld/` 是原始世界模型仓库。
- `Find_Physics_Zone/` 是在原模型之上加 probe / hidden-state / physics-coherence 分析的实验分支。
- 因此需要同时满足两类前提：
  - 原始模型前提：`pretrained_models/world_model.pth`、`pretrained_models/vqvae.pt`
  - nuPlan 数据前提：`data/seq_meta/*.json`、`data/ego_meta/*.json`、对应图像目录

当前仓库没有自带 `data/` 和 `pretrained_models/`，所以只能做静态核查，不能直接跑通端到端。

## 2. 当前代码的实流水线

1. `video_data_preprocess/create_nuplan_json.py`
2. `tools/test_dataset_read.py`
3. `tools/build_physics_dataset.py`
4. `tools/build_probe_dataset.py`
5. `tools/test_hidden_states.py`
6. `tools/extract_zone_features.py --task physics`
7. `tools/extract_zone_features.py --task coherence`
8. `tools/train_physics_probe.py`
9. `tools/train_coherence_probe.py`


### A. 数据读取

脚本：`tools/test_dataset_read.py`

检查项：

- `NuPlanTest.__getitem__()` 输出是否为
  - `imgs: [T, 3, 256, 512]`
  - `poses: [T, 2]`
  - `yaws: [T, 1]`
- 图像是否已归一化到 `[-1, 1]`
- `seq_meta` / `ego_meta` / 图像路径是否能自动解析

注意：

- `datasets/dataset_nuplan.py` 输出的是逐帧相对运动，不是绝对轨迹。
- 这一点会直接决定 physics 标签该怎么定义。

### B. Physics 样本构造

脚本：`tools/build_physics_dataset.py`

输出：

- `physics_samples.pt`

每条样本包含两类标签：

- past summary
  - `past_avg_speed`
  - `past_final_speed`
  - `past_avg_yaw_rate`
  - `past_final_yaw_rate`
  - `past_heading_change`
- future step
  - `future_delta_x`
  - `future_delta_y`
  - `future_delta_yaw`
  - `future_speed`
  - `future_yaw_rate`
  - `future_turn_class`


### C. Coherence 样本构造

脚本：`tools/build_probe_dataset.py`

输出：

- `coherence_samples.pt`

它现在做的事情是：

- 对每个 physics base sample 生成 3 个 pair
  - 正样本：`base == cond`
  - 随机负样本：`random_mismatch`
  - 难负样本：`hard_mismatch`

所以它不是“直接从原始序列同时生成 physics + coherence”，而是“先 physics，后 coherence”。

### D. Hidden-state 接口验收

脚本：`tools/test_hidden_states.py`

检查项：

- tokenizer 是否能把 `[B, 16, 3, 256, 512]` 编成 `[B, 16, 512, 32]`
- `TrainTransformers.extract_hidden_states()` 是否能返回 `hidden_states`
- world model 是否能同时返回 logits 和 hidden

- 模型内部 hidden：
  - `time_space_0` 到 `time_space_11`
  - `next_state_hidden`
  - `ar_0` 到 `ar_5`
  - 合计 19 层
- 特征提取脚本还会额外加入两个 raw motion baseline：
  - `raw_pose_yaw_past`
  - `raw_pose_yaw_full`
- 特征提取脚本还会额外加入 `tokenizer_last`
  - 所以最终 feature file 里通常是 22 个 layer key

### E. 特征提取

脚本：`tools/extract_zone_features.py`

运行方式：

```bash
python tools/extract_zone_features.py --task physics
python tools/extract_zone_features.py --task coherence
```

输出：

- `physics_features.pt`
- `coherence_features.pt`

支持两种读出：

- `--pool_mode all_mean`
- `--pool_mode tokenwise`


### F. Probe 训练

Physics：

- 脚本：`tools/train_physics_probe.py`
- 模型：`StandardScaler + Ridge`
- 切分：`GroupKFold(seq_id)`

Coherence：

- 脚本：`tools/train_coherence_probe.py`
- 模型：`StandardScaler + LogisticRegression`
- 切分：`StratifiedGroupKFold(seq_id/source_index)`


- README 里过时的运行顺序和层数说明不能继续照搬
- coherence 若要宣称“严格无泄漏”，还需要把配对逻辑改成“先分折，再采样”
