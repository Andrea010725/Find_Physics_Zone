A. 数据读取与样本构造相关   
1. datasets/dataset_nuplan.py 
是后面所有样本构造、特征提取，都必须从统一的数据入口读原始序列
作用:读取处理好的 nuPlan 数据
输出：图像序列   pose 序列     yaw 序列
输入: 依赖数据本身的： seq_meta/...json  ego_meta/...  图像目录
__getitem__() 返回：
imgs: [T, 3, 256, 512]
poses: [T, 2]
yaws: [T, 1]

2. tools/test_dataset_read.py （可选）
作用： 单独验证 dataset_nuplan.py 的正确性 确认数据格式  图像归一化 pose/yaw shape
输入：直接调用 NuPlanTest
输出：打印的 dataset 长度        imgs/poses/yaws shape    dtype   min/max
Nuplan的结果是：
imgs shape = [100, 3, 256, 512]
poses shape = [100, 2]
yaws shape = [100, 1]

B. probe 样本构造相关
3. tools/build_probe_dataset.py

这个是很关键的脚本。

作用

从原始序列中构造两类 probe 样本：

physics 样本

coherence 样本

输入

来自 NuPlanTest 的序列：

imgs

poses

yaws

输出

两个文件：

physics_samples.pt

每条样本大致包含：

past_imgs

past_poses

past_yaws

future_img

future_pose

future_yaw

speed

yaw_rate

heading_change

coherence_samples.pt

每条样本大致包含：

past_imgs

past_poses

past_yaws

future_img

future_pose_cond

future_yaw_cond

coherence_label

为什么需要它

因为你后面做 probe，不是直接拿原始序列训练，而是要先定义：

物理量标签

一致性标签

这个脚本就是把“原始数据”变成“研究问题对应的监督样本”。

4. tools/build_physics_dataset.py

你中间其实先写过一个 physics-only 版本，后来再扩成了统一的 build_probe_dataset.py。

作用

最早先单独验证 physics 标签能不能算出来。

输入

同样来自 NuPlanTest

输出

最开始只保存 physics_samples.pt

为什么它存在过

它是你最早验证：

speed / yaw_rate / heading_change 标签定义是否成立
的原型脚本。

后来功能被 build_probe_dataset.py 吸收了。

C. 模型 hidden-state 提取接口相关
5. models/ar.py

这是你改动最核心的模型文件。

你做了什么修改

在 GPT.forward() 里新增了：

return_hidden_states=False

后来又新增了 return_logits=True

作用

让模型除了正常输出 logits 之外，还能选择性返回每一层 hidden states。

输入

GPT.forward() 现在接收：

feature_total

pose_indices_total

yaw_indices_total

drop_flag

return_hidden_states

return_logits

输出
默认情况

还是：

yaw_logits

pose_x_logits

pose_y_logits

img_logits

如果 return_hidden_states=True

额外加：

hidden_states

如果 return_logits=False

跳过 head，只返回 hidden，节省显存。

为什么需要它

因为你整个实验的核心就是：

对每一层 hidden 做 probe

没有这个修改，你就拿不到逐层表征。

6. models/model.py

你在 TrainTransformers 里补了一个接口。

新增函数

extract_hidden_states(...)

作用

从 wrapper 层暴露出“只提 hidden”的路径。

输入

feature_total

pose_indices_total

yaw_indices_total

drop_flag

输出

返回：

hidden_states
（现在通过 return_logits=False 跳过 logits）

为什么需要它

因为外面脚本不应该直接绕很深去调 GPT.forward()。
这个函数让：

test_hidden_states.py

extract_zone_features.py

都能稳定复用同一个入口。

D. hidden-state sanity check 相关
7. tools/test_hidden_states.py
作用

验证：

tokenizer

world model

hidden-state 返回接口

整条链路是否能跑通。

输入

一条真实 nuPlan 样本

Tokenizer

TrainTransformers.extract_hidden_states()

输出

打印：

输入 shape

out.keys()

hidden 层数

每层 hidden shape

为什么需要它

它是正式批量特征提取前的最后一个接口验收脚本。

你最终得到的正确结果

num hidden states = 18

time_space_0 ~ time_space_11

ar_0 ~ ar_5

形状分别是：

time_space: [1, 15, 515, 1536]

ar: [15, 515, 1536]

这个结果是整个实验正式开始前的关键里程碑。

E. 特征提取相关
8. tools/extract_zone_features.py

这是正式实验最重要的桥梁脚本。

作用

把：

physics_samples.pt

coherence_samples.pt

变成：

physics_features.pt

coherence_features.pt

输入

根据 TASK 选择：

physics 或 coherence 样本文件

内部做了两阶段
Stage A

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

为什么需要它

这是把“模型内部 hidden”变成“后续 probe 可直接训练的数据文件”的关键一步。

没有它，你后面训练每一层 probe 时都得重复跑：

tokenizer

world model

hidden 提取

太慢也太乱。

F. probe 训练相关
9. tools/train_coherence_probe.py
作用

对 coherence_features.pt 逐层训练二分类 probe。

输入

coherence_features.pt

模型

LogisticRegression

StratifiedKFold

输出

每层的 acc / f1 / auc

保存到 coherence_probe_results.pt

为什么需要它

它是你“找 zone”的主实验脚本。

10. tools/train_physics_probes.py
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

为什么需要它

它是你“解释 zone 里到底出现了什么运动学量”的主实验脚本。

三、现在按运行顺序，从头到尾梳理一遍

下面是你现在已经跑过的标准流程。

第 1 步：验证数据读取

运行：

python test_dataset_read.py
目的

确认 dataset_nuplan.py 没问题。

正确输出特征

dataset length = 27

imgs shape = [100, 3, 256, 512]

poses shape = [100, 2]

yaws shape = [100, 1]

第 2 步：构造 probe 样本

运行：

python build_probe_dataset.py
目的

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

第 3 步：验证 hidden-state 接口

运行：

python test_hidden_states.py
目的

验证：

tokenizer → feature

model → hidden_states

shape 是否符合预期

最终正确结果

out keys 包含 hidden_states

num hidden states = 18

time_space 和 ar 两段 shape 正常

第 4 步：提 physics 特征

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

第 5 步：提 coherence 特征

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

第 6 步：跑 coherence probe

运行：

python train_coherence_probe.py
目的

逐层二分类，找 coherence zone。

第 7 步：跑 physics probes

运行：

python train_physics_probes.py
目的

逐层回归，看 physics 信息在哪些层最可读。

四、第一次结果是什么，后面又改了什么

这部分很重要，因为你后面写研究记录时就会用到。

A. 第一次 coherence 结果（原始 pooling）
原始 time_space pooling

你最开始对 time_space 做的是：

最后一个时间步

全 515 个 token 平均

第一次 coherence 结果

time_space_* 基本全灭：

acc ≈ 0.038

auc ≈ 0.007

ar_* 有稳定信号：

最好是 ar_2

ar_0 ~ ar_5 大致在 0.60 ~ 0.67 AUC

初步解释

当时我们怀疑：

也许 time_space 不是没信息

只是全 token 平均把信号抹掉了

B. 为了检验这个怀疑，你改了 pooling
改动内容

只改 extract_zone_features.py 里的 pool_hidden_state()：

对于 time_space_*：

不再用全 token mean

改成只取最后时间步的前 3 个 token（yaw / pose_x / pose_y）平均

对于 ar_*：

保持不变

为什么这么改

因为 coherence 负样本就是通过改：

future pose_cond

future yaw_cond

构造出来的。
所以如果 time_space 层有 coherence 信息，最可能先体现在这些结构化 token 上。

C. 改完后重新跑了两步
1. 重提 coherence 特征

重新运行：

python extract_zone_features.py

TASK = "coherence"

2. 重新跑 coherence probe

运行：

python train_coherence_probe.py
D. 改完后的 coherence 结果

结果基本没变：

time_space_* 依旧接近 0

ar_* 依旧稳定在 0.60~0.67

最佳层仍然是：

ar_2

其次 ar_0 / ar_1 / ar_3

这个结果意味着什么

这非常关键：

time_space 的差结果，不是因为“全 token 平均把 pose/yaw token 信号抹掉了”。

因为你已经试过只读前 3 个 token，结果还是没救回来。

所以现在可以更有把握地说：

在你当前任务定义和当前 readout family 下，coherence 的可读信号主要出现在 AR 段。

五、第一次 physics 结果是什么

physics 这条线没有做 readout 对照实验，你现在是第一版结果。

physics 结果概况
1. speed

几乎所有层都：

R² < 0

这说明：

当前 feature 对 speed 的预测不如直接预测均值。

所以：

speed 当前不适合作为主结论

2. yaw_rate

有一点弱信号，最好大致在：

time_space_0

time_space_9

其中 time_space_0 最好，R² ≈ 0.16

这说明：

模型里可能有一点瞬时转向变化率信息，但还不强。

3. heading_change

这是目前最强的 physics 量，尤其在 time_space 早层：

time_space_0: R² ≈ 0.74

time_space_1: R² ≈ 0.64

很多 time_space 层都显著 > 0

这说明：

方向变化相关的运动学趋势，在 time_space 早中层非常可读。

六、physics 和 coherence 放在一起，现在的最主要结论是什么

这是你目前最值得记住的一段。

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

你现在看到的不是“所有东西都在同一层最强”，而是：

time-space 早层

更像在编码：

运动状态

几何趋势

方向变化

AR 中后层

更像在编码：

future 条件与结果的一致性

coherence / matching

这个分层分工现象，是你目前最有价值的中间结论。

七、现在这些结果能不能说明设计合理

我的判断是：

可以说明：整体设计是合理的

因为你已经看到了：

coherence 非随机

physics 非全灭

层间差异清楚

改 pooling 后还能做出有诊断力的对照

这说明：

你的实验 pipeline 不是假的，不是随机碰运气。

但也说明：不同标签的可信度不同
比较稳的主线

coherence

yaw_rate

heading_change

当前不适合当主结果

speed

对 heading_change 要额外谨慎

因为它很强，但也可能部分来自：

输入中已有的 past yaw 变化趋势

所以它虽然有用，但更像：

time-space 早层保留方向变化信息的证据
而不是单独证明“深层 physics emergence”。

八、你现在迁移到完整数据前，应该怎么理解自己已经完成了什么

你现在已经完成的是：

1. 整条技术链路打通

包括：

数据读取

样本构造

hidden 提取

特征保存

逐层 probe 训练

2. 小规模验证已经有非随机结果

特别是：

coherence 在 AR 段

heading_change 在 time-space 早层

3. 做过至少一次有效对照实验

也就是 time_space pooling 对照。

这很重要，因为它说明你不是只跑了一组结果就停，而是已经开始做“结果诊断”。

九、如果你现在要上完整数据，最合理的顺序是什么

我建议你上完整数据时，优先级这样排：

第一优先

coherence
因为它是 zone 主任务，而且你已经看到稳定信号。

第二优先

yaw_rate
因为它比 heading_change 更不 trivial，更适合作为辅助 physics 信号。

第三优先

heading_change
保留，但解释时要明确：

它可能部分来源于输入中已有的方向变化信息

第四优先

speed
作为探索项，不要先拿它当主结论。

十、你现在最值得补的一个小检查是什么

在上完整数据前，我仍然建议你后面补一个：

raw pose/yaw baseline

尤其针对：

heading_change

yaw_rate

作用是判断：

这些标签是不是从原始输入就能很容易读出

模型表征到底比 raw input 强多少

但这个 baseline 不是你上完整数据的硬前提。
如果你现在想先上完整数据，也是可以的。

十一、最简版“文件-作用-输入输出-运行顺序”总表
文件与作用
datasets/dataset_nuplan.py

读原始 nuPlan 序列
输入：json + 图像
输出：imgs / poses / yaws

tools/test_dataset_read.py

检查数据读取
输入：dataset
输出：shape / dtype / minmax

tools/build_probe_dataset.py

构造 physics / coherence 样本
输入：原始序列
输出：

physics_samples.pt

coherence_samples.pt

models/ar.py

支持逐层 hidden 提取
输入：模型前向
输出：hidden_states，可选跳过 logits

models/model.py

暴露 extract_hidden_states()
输入：feature_total / pose_indices / yaw_indices
输出：hidden_states

tools/test_hidden_states.py

验证 hidden 提取链
输入：1 条样本
输出：18 层 hidden shape

tools/extract_zone_features.py

提逐层特征文件
输入：

physics_samples.pt

coherence_samples.pt
输出：

physics_features.pt

coherence_features.pt

tools/train_coherence_probe.py

逐层 coherence 分类
输入：coherence_features.pt
输出：每层 acc/f1/auc

tools/train_physics_probes.py

逐层 physics 回归
输入：physics_features.pt
输出：每层 mse/r2/pearson

运行顺序
python test_dataset_read.py
python build_probe_dataset.py
python test_hidden_states.py

# 提 physics 特征
# TASK = "physics"
python extract_zone_features.py

# 提 coherence 特征
# TASK = "coherence"
python extract_zone_features.py

python train_coherence_probe.py
python train_physics_probes.py
