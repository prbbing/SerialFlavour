# Experiment 3：upstream 辅助任务 loss 权重扫描

固定使用 122,077 参数的 Parallel Transformer、3,000,000 jets 总数据量和
训练池内 A/B = 75/25 的分配，扫描 track-origin 与 track-pair vertexing
两个辅助任务的 loss 系数。Jet head 的系数始终为 1：

```text
L = 1 * L_jet + lambda_origin * L_origin + lambda_pair * L_pair
lambda_origin, lambda_pair in {0.2, 0.5, 1.0, 1.5}
```

两个系数独立取值，构成完整的 4 × 4 网格，共 16 个实验 JSON。
文件名格式为
`experiment3_p122k_n3m_a075_b025_origin<weight>_pair<weight>.json`，
其中 `0p2`、`0p5`、`1p0`、`1p5` 分别表示 0.2、0.5、1.0、1.5。
每个配置通过 `overrides.parallel.loss_weights` 显式设置三个系数，
并在 `experiment.markers.variables` 中记录对应的 `lambda_*` 值。

## 数据划分

沿用 Experiment 1 EX 的总数据 70/10/20 协议；75/25 是 70% 训练池内的
upstream A-train / downstream B-train 比例。

| 数据子集 | Jets |
|---|---:|
| A-train（训练池的 75%） | 1,575,000 |
| B-train（训练池的 25%） | 525,000 |
| A-val / B-val（同一共享验证集） | 300,000 |
| Y-test | 600,000 |
| 总计（共享验证集计一次） | 3,000,000 |

16 个配置共用 `data_seed: 42` 和同一个 split 目录：
`/data/yuyang/SerialFlavour/parallel_refine/cache/splits/experiment3_n3m_a075_b025`。
`shared_validation: true` 使 A-val 与 B-val 共用验证数据；其余子集沿用
事件互斥协议，归一化仅由 A-train 拟合。实际数据索引需在数据准备阶段生成。

## 固定设置与输出

组件引用为 `../../data/experiment1_ex_n3m.json`、
`../../parallel/experiment1_p122k.json` 和 `../../refiner/dnn_default.json`。
除实验标识、split 目录、A/B 样本数和上述 loss 系数外，其余设置继承
现有 3M、75/25 配置，包括模型结构、类别权重、优化器、学习率调度、
upstream seeds 1–5 和 downstream seeds 1–5 的 Cartesian 配对。
下游 recipes 沿用 `F1_embed`、`F3_embed_aux`、`F4_all`、`FG2`、`FG2s`、`FG4`。

每组权重使用独立的 `experiment.name`，因此 checkpoint、冻结特征缓存
和结果按实验标识隔离。结果目录为
`/data/yuyang/SerialFlavour/results/parallel_refine/<experiment.name>/`。
所有配置使用同一 `comparison_group`：
`experiment3_p122k_n3m_a075_b025_loss_weights`。

## GPU 集群训练入口

入口为 `scripts/production/run_experiment3_full.sh`，复用现有的
`scripts/production/run_experiment1_ex_recipe_queue.py` 和
`scripts/production/run_experiment1_ex_seed.py`。部署时需同时包含这三个脚本、
训练/评估入口和配置所引用的组件。

在 Linux GPU 节点激活训练环境后，从仓库根目录运行：

```bash
# 检查配置和完整命令，不生成日志或数据，不启动训练。
GPU_POOL="0 1 2 3" bash scripts/production/run_experiment3_full.sh --dry-run

# 先只准备数据；正式运行也会自动执行此阶段并复用已有缓存。
PREPARE_WORKERS=8 bash scripts/production/run_experiment3_full.sh --prepare-only

# 单节点示例：使用已分配给本次实验的四张 GPU。
GPU_POOL="0 1 2 3" bash scripts/production/run_experiment3_full.sh

# 后台运行示例。
mkdir -p logs/parallel_refine/experiment3
nohup env GPU_POOL="0 1 2 3" PYTHON=python \
  bash scripts/production/run_experiment3_full.sh \
  > logs/parallel_refine/experiment3/launcher.log 2>&1 &
```

16 个配置各训练 5 个 upstream seeds，共 80 个 Transformer；每个配置的
6 个 downstream recipes 各训练 5 × 5 个 seed 配对，共 2,400 个 refiner。
队列按 `Parallel → B cache → recipe → Y cache → evaluation → prune`
依赖调度，GPU 空闲后立即接收下一项任务。每张 GPU 同时接收一个调度单元；
每个 recipe 单元沿用现有 production worker 的行为，在同一张 GPU 上
并发训练 5 个 downstream seeds。`REFINE_GPUS` 默认是 GPU 数减一（最小为一），
在仍有 upstream 排队时限制下游并发；upstream 队列清空后会利用其余 GPU。

`GPU_POOL` 使用空格分隔的实际数字设备 ID；未设置时优先使用数字形式的
`CUDA_VISIBLE_DEVICES`，否则默认 `0 1 2 3`。脚本检查可见 CUDA 设备数量，
GPU UUID/MIG 标识不受现有数字 ID worker 支持。GPU 应由作业系统或用户提前
分配；本脚本不检查其他作业的占用，也不申请集群资源。

多节点通过配置分片运行独立训练任务。各节点需要访问相同的数据、split、
processed cache 和结果路径，并使用支持跨节点 `flock` 的共享文件系统。
所有节点必须使用一致的 `NODE_COUNT` 和代码/配置版本、不同的 `NODE_RANK`。
例如在四个节点上分别运行（`NODE_RANK` 对应为 0、1、2、3）：

```bash
NODE_COUNT=4 NODE_RANK=0 GPU_POOL="0 1 2 3" \
  bash scripts/production/run_experiment3_full.sh
```

分片规则为固定的 origin/pair 网格序号对 `NODE_COUNT` 取模，每个节点获得
互不重叠的完整配置。四节点时每节点为 4 个配置、20 个 upstream 和 600 个
downstream 训练。共享数据准备加锁且顺序执行，防止多个节点同时生成同一
split 或 processed cache；训练阶段无需等待其他节点完成准备。

默认日志和状态目录为 `logs/parallel_refine/experiment3/full/node<RANK>/`：
`stage1/` 保存数据准备日志，`scheduler.log` 保存派发记录，
`failed_jobs.log` 保存失败记录，`units/<experiment>/seed<N>/` 保存各阶段日志，
`state/` 和 `aggregate/` 保存完成标记。同一节点分片通过共享 split 目录中的
`.experiment3.node<RANK>.run.lock` 防止重复启动。

重跑同一命令会复用数据和已完成 checkpoint，跳过具有完成标记的 seed；
未完成的训练阶段从头重训，不恢复 epoch、优化器或随机数状态。
`RETRIES=1` 表示每阶段失败后再尝试一次，重试时由现有 worker 清理该 seed 的
不完整产物。重试耗尽后丢弃该 seed 的后续任务，其他任务继续，最终退出码非零。
仅在某个配置的全部 5 个 upstream seeds 流程成功后进行跨 seed 聚合。
评估成功后会清理该 seed 的 frozen/graph B/Y 缓存，保留 checkpoint、
processed cache 和评估结果。

可通过 `PYTHON`、`PREPARE_WORKERS`、`RETRIES`、`PATIENCE`、`REFINE_GPUS`、
`LOG_BASE`、`MIN_FREE_GB`、`SPACE_PATH` 设置运行参数；完整说明见 `--help`。
默认检查 `/data` 至少有 200 GiB 空闲，该阈值是启动门槛，不是整个实验所需
空间的测量值。

本地验证仅检查脚本语法、配置和调度命令；未进行实际数据准备、GPU 训练
或 Y-test 评估。
