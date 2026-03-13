# A. 数据读取与样本构造相关    【和数据集有关】
# 1. datasets/dataset_nuplan.py 
是后面所有样本构造、特征提取，都必须从统一的数据入口读原始序列
作用:读取处理好的 nuPlan 数据
输出：图像序列   pose 序列     yaw 序列
输入: 依赖数据本身的： seq_meta/...json  ego_meta/...  图像目录
__getitem__() 返回：
imgs: [T, 3, 256, 512]
poses: [T, 2]
yaws: [T, 1]

# 2. tools/test_dataset_read.py （可选）
作用： 单独验证 dataset_nuplan.py 的正确性 确认数据格式  图像归一化 pose/yaw shape
输入：直接调用 NuPlanTest
输出：打印的 dataset 长度        imgs/poses/yaws shape    dtype   min/max
Nuplan的结果是：
imgs shape = [100, 3, 256, 512]
poses shape = [100, 2]
yaws shape = [100, 1]
 
# B. probe 样本构造相关 （两个测试需要的数据）   【和数据集有关】
# 3. tools/build_probe_dataset.py
作用： 从原始序列中构造两类 probe 样本 -- physics 样本 和 coherence 样本
输入： 来自 NuPlanTest 的序列 含 imgs  poses yaws
输出： 两个文件 -- 
第一个文件是physics_samples.pt
每条样本大致包含：
past_imgs
past_poses
past_yaws
future_img
future_pose
future_yaw
speed   物理量标签
yaw_rate  物理量标签
heading_change  物理量标签

第二个文件是 coherence_samples.pt
每条样本大致包含：
past_imgs
past_poses
past_yaws
future_img
future_pose_cond
future_yaw_cond
coherence_label  一致性标签

# 4. tools/build_probe_dataset.py 
作用 把 dataset_nuplan.py 读出来的“基础真实序列”，加工成两份可直接用于后续实验的数据：
physics_samples.pt：只含正样本，用于 speed / yaw_rate / heading_change probe
coherence_samples.pt：含正负样本，用于 coherence 二分类 probe
输入 ：同样来自 NuPlanTest
输出 ： physics_samples.pt    coherence_samples.pt

# C. 模型 hidden-state 提取接口相关   【和复现的repo有关】
# 5. models/ar.py
整个实验的核心就是： 对每一层 hidden 做 probe
需要拿到逐层表征
新增改动
在 GPT.forward() 里新增了：
return_hidden_states=False
return_logits=True （显存问题）
作用 ： 让模型除了正常输出 logits 之外，还能选择性返回每一层 hidden states。
GPT.forward() 现在接收：
feature_total
pose_indices_total
yaw_indices_total
drop_flag
return_hidden_states
return_logits

输出
默认情况
如果 return_hidden_states=True
额外加：  hidden_states
如果 return_logits=False
跳过 head，只返回 hidden，节省显存。

# 6. models/model.py
在 TrainTransformers 里补一个接口
新增函数 extract_hidden_states(...)
作用 ：从 wrapper 层暴露出“只提 hidden”的路径。
输入
feature_total
pose_indices_total
yaw_indices_total
drop_flag
输出
返回：
hidden_states
（现在通过 return_logits=False 跳过 logits）
外面脚本不应该直接绕很深去调 GPT.forward()。
这个函数让： 
test_hidden_states.py
extract_zone_features.py
都能稳定复用同一个入口。

# D. hidden-state sanity check 相关
# 7. tools/test_hidden_states.py （可选）
作用 验证： tokenizer
world model
hidden-state 返回接口 整条链路是否能跑通。
输入 一条真实 nuPlan 样本
Tokenizer
TrainTransformers.extract_hidden_states()
输出 打印：
输入 shape
out.keys()
hidden 层数
每层 hidden shape
正式批量特征提取前的最后一个接口验收脚本。

Muplan + DrivingWorld最终得到的结果是：
num hidden states = 18
time_space_0 ~ time_space_11
ar_0 ~ ar_5
形状分别是：
time_space: [1, 15, 515, 1536]
ar: [15, 515, 1536]

# E. 特征提取相关
# 8. tools/extract_zone_features.py
作用 把：
physics_samples.pt
coherence_samples.pt
变成：
physics_features.pt
coherence_features.pt

输入 根据 TASK 选择： physics 或 coherence 样本文件
内部做了两阶段 Stage A
用 tokenizer 把图像序列编码成：
feature_total
pose_indices_total
yaw_indices_total
drop_flag

Stage B
用 world model 提 hidden，逐层 pooling，得到固定长度特征。
输出 
physics
physics_features.pt
大致是：
layer_features[layer_name] = [N, 1536]
labels["speed"] = [N]
labels["yaw_rate"] = [N]
labels["heading_change"] = [N]

coherence
coherence_features.pt
大致是：
layer_features[layer_name] = [N, 1536]
labels["coherence_label"] = [N]

把“模型内部 hidden”变成“后续 probe 可直接训练的数据文件”
后面训练每一层 probe 时就不需要重复跑：
tokenizer
world mode
hidden 提取


# F. probe 训练相关
# 9. tools/train_coherence_probe.py
作用 对 coherence_features.pt 逐层训练二分类 probe。
输入 coherence_features.pt
模型
LogisticRegression
StratifiedKFold

输出
每层的 acc / f1 / auc
保存到 coherence_probe_results.pt

# 10. tools/train_physics_probes.py
作用
对 physics_features.pt 逐层训练回归 probe。
输入
physics_features.pt

模型
Ridge Regression
LeaveOneOut CV

输出
每层对 3 个目标的：
mse
r2
pearson
保存到 physics_probe_results.pt


# ------------标准流程-------------- 以Nuplan + DrivingWorld为例
# 第 1 步：验证数据读取
运行：
python test_dataset_read.py
确认 dataset_nuplan.py 没问题。
正确输出特征
dataset length = 27
imgs shape = [100, 3, 256, 512]
poses shape = [100, 2]
yaws shape = [100, 1]

# 第 2 步：构造 probe 样本
运行：
python build_probe_dataset.py
生成：
physics_samples.pt
coherence_samples.pt
输出
physics
27 条样本
含 3 个 physics 标签
coherence
54 条样本
27 正 + 27 负

# 第 3 步：验证 hidden-state 接口
运行：
python test_hidden_states.py
验证： 
tokenizer → feature
model → hidden_states
shape 是否符合预期
最终确结果
out keys 包含 hidden_states
num hidden states = 18
time_space 和 ar 两段 shape 正常

# 第 4 步：提 physics 特征
在 extract_zone_features.py 里：
TASK = "physics"
运行：
python extract_zone_features.py
输出
physics_features.pt
summary
18 层，每层 (27, 1536)
标签：
speed (27,)
yaw_rate (27,)
heading_change (27,)

# 第 5 步：提 coherence 特征
把 TASK 改成：
TASK = "coherence"
再运行：
python extract_zone_features.py
输出
coherence_features.pt
summary
18 层，每层 (54, 1536)
标签：
coherence_label (54,)

# 第 6 步：跑 coherence probe
运行：
python train_coherence_probe.py
目的
逐层二分类，找 coherence zone。

# 第 7 步：跑 physics probes
运行：
python train_physics_probes.py
逐层回归，看 physics 信息在哪些层最可读。


physics 和 coherence 放在一起，现在的最主要结论是什么 （不准 数据太少了 没有统计1学意义）

coherence 的主信号
主要出现在：
ar_0 ~ ar_3
最佳大致是 ar_2
也就是：
future condition consistency / coherence 信息，主要在 AR 段变得可读。

physics 的主信号
主要出现在：
time_space_0 ~ time_space_3
尤其是 heading_change
也就是：
方向变化 / 运动状态摘要，更早在 time-space 段可读。

合在一起的结构性解释

现在看到的不是“所有东西都在同一层最强”，而是：
time-space 早层
更像在编码：
运动状态
几何趋势
方向变化
AR 中后层
更像在编码：
future 条件与结果的一致性
coherence / matching

分层分工现象

