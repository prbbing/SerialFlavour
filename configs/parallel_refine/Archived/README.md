# 旧训练方案归档

此目录集中保存已停用的实验配置、专用组件和训练入口，供查阅历史设计与运行记录。
这些实验不再用于新训练或当前结果比较。

## 目录结构

```text
Archived/
├── data/                  旧实验专用及已停用的数据组件
├── experiments/           旧实验 JSON、扩展配置和原有说明
├── parallel/              旧实验专用的 Transformer 组件
├── refiner/               旧实验专用的 refiner 组件
└── scripts/
    └── production/        旧 Experiment 1 EX 生产训练入口
```

归档保留配置原有的分类与实验子目录结构，相对组件路径已调整到移动后的实际位置。
旧的 `local/scripts` 训练入口已删除，不在本归档中。

## 归档范围

`experiments/` 共包含 120 个实验 JSON：

| 实验系列 | 配置数 | 原实验内容 |
|---|---:|---|
| `experiment1/` | 42 | 固定 1M 训练预算，扫描模型大小与 A/B 分配，含扩展 |
| `experiment1_ex/` | 42 | 总数据 1M–6M，各扫描 7 种 A/B 分配 |
| `experiment2/` | 25 | 固定 B-train，扫描 A-train 规模，含扩展 |
| `experiment2_ex/` | 5 | 加宽 upstream 任务头，扫描 backbone 与 A-train 规模 |
| 基础及过渡配置 | 6 | `default*.json`、`a1m_6layers.json`、`a500k_4layers.json`、`a500k_6layers.json` |

`data/`、`parallel/`、`refiner/` 合计保留 25 个专用或停用组件。
旧生产入口位于 `scripts/production/run_experiment1_ex_full.sh`。
已有结果、checkpoint 和缓存目录没有随此次配置归档移动。

## 训练方案分界与历史复现

2026-10-05 的两次更新构成此次新旧方案分类的分界：

- `ed39f64`：upstream 与 downstream 加入验证集驱动的 `ReduceLROnPlateau` 学习率衰减。
- `83740c3`：下游 tabular DNN 与 graph-DNN 的 jet CE 对齐 upstream 的
  b/c/light 类别权重 `[2, 2, 1]`，训练、验证、checkpoint 选择、early stopping
  和学习率调度使用一致的加权 CE。

归档包含上述更新前引入的实验，以及引入 LR 衰减、但早于下游加权 CE 修订的
过渡配置 `default_lr_decay.json`。

**此归档不是旧训练代码或旧运行环境的快照。** 组件内容曾随共享配置更新，
部分实验仍引用归档外的共享组件；训练入口也调用当前仓库代码。因此，直接运行
归档配置不能保证重现更新前的训练行为。复现历史结果应使用对应的代码提交、
运行时 resolved config、manifest、数据索引和 checkpoint，并核对 loss 权重与调度器设置。

## 当前使用的实验与共享入口

当前实验位于归档外：

- [Experiment 1 EX V2](../experiments/experiment1_ex_v2/)：总数据量与 A/B 分配扫描。
- [Experiment 2 EX V2](../experiments/experiment2_ex_v2/)：加宽 upstream 任务头的比较。
- [Experiment 3](../experiments/experiment3/)：origin/pair 辅助任务 loss 权重扫描。

`scripts/production/run_experiment1_ex_recipe_queue.py` 与
`scripts/production/run_experiment1_ex_seed.py` 虽保留旧实验名称，但仍被当前生产入口复用，
因此留在仓库原位置。通用训练代码与新实验使用的共享组件也保留在归档外。

## 2026-11-03：当前 downstream 输入配方调整

当前共享组件 `../refiner/dnn_default.json` 重新启用 origin 和 vertex-pair
输入对比，并新增重复 global jet representation 的参数量对照。名称对应如下：

| 原名称或本次临时名称 | 当前名称 | 输入与对比目的 |
|---|---|---|
| `F1O` | `F2_origin` | global jet representation + 冻结 track-origin 概率的 attention 汇总 |
| `F1V` | `F2_vertex` | global jet representation + 冻结 pair 概率加权的邻居 embedding 汇总 |
| `F1p`（本次临时名称） | `F1_double` | 同一个 global jet representation 拼接两次，作为 `F2_vertex` 的相同参数量对照 |

当前默认启用 9 个配方：`F1_embed`、`F1_double`、`F2_origin`、`F2_vertex`、
`F3_embed_aux`、`F4_all`、`FG2`、`FG2s`、`FG4`。
在 global representation 为 48 维、origin 为 8 类、MLP 隐藏层为
`128 → 64 → 32` 时，`F1_embed`、`F2_origin` 的参数量分别为 16,707、17,731；
`F1_double` 和 `F2_vertex` 均为 22,851。`F2_vertex` 不包含
`F3_embed_aux` 中的 pair mean/max/sum 标量汇总。

`F1_double` 不增加输入信息；重复输入的首层权重可合并，因此相对
`F1_embed` 不扩大可表示的函数类别，但可能改变优化和正则化效果。
上述调整用于当前实验，尚未由本次修改启动训练。

归档配置保留原有名称和内容，不随此次改名更新；本节仅记录它们与当前方案的对应关系。
当前代码的配方注册与选择器使用新名称，因此含旧名称的归档配置不能直接用于当前训练入口。
查阅或复现历史结果时，应以历史代码版本及当次运行产物为准，不将新名称套用于旧结果目录。
