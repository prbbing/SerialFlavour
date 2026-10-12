# Experiment 1 EX V2：3M 数据的 A/B 训练分配扫描

固定使用 122,077 参数的 Parallel Transformer 和总计 3,000,000 jets，
扫描 upstream A-train / downstream B-train 的训练池分配比例。
Upstream loss 系数在每个配置中显式固定为：

```text
L = 1.0 * L_jet + 0.5 * L_origin + 1.5 * L_pair
```

## 数据与配置矩阵

沿用总数据 70/10/20 协议：训练池 2,100,000、验证集 300,000、
Y-test 600,000。下表比例仅作用于训练池。
每个配置启用 `shared_validation: true`，其 A-val 和 B-val 使用同一份
300,000 jets 验证数据，汇总数据量时只计一次；其他子集沿用事件互斥协议。
归一化统计仅由该配置的 A-train 拟合。

| A/B 比例 | A-train jets | B-train jets | 文件名后缀 |
|---|---:|---:|---|
| 100/0 | 2,100,000 | 0 | `a100_b000` |
| 95/5 | 1,995,000 | 105,000 | `a095_b005` |
| 90/10 | 1,890,000 | 210,000 | `a090_b010` |
| 85/15 | 1,785,000 | 315,000 | `a085_b015` |
| 80/20 | 1,680,000 | 420,000 | `a080_b020` |
| 75/25 | 1,575,000 | 525,000 | `a075_b025` |
| 70/30 | 1,470,000 | 630,000 | `a070_b030` |
| 65/35 | 1,365,000 | 735,000 | `a065_b035` |
| 60/40 | 1,260,000 | 840,000 | `a060_b040` |
| 55/45 | 1,155,000 | 945,000 | `a055_b045` |
| 50/50 | 1,050,000 | 1,050,000 | `a050_b050` |

共 11 个实验 JSON，文件名格式为
`experiment1_ex_v2_p122k_n3m_a<AAA>_b<BBB>.json`。
请求中的 `85/150` 按 `85/15` 处理；已补充 `75/25`。
`100/0` 的 `b_train=0` 且 `dnn_enabled=false`，作为 Transformer-only 基线；
其余配置均为 Transformer 加下游 refiner。

## v2 相对于之前实验的变动

### 配置矩阵与实验身份

以下对比以仓库当前的 `../experiment1_ex/` 配置为准：

| 项目 | 原 Experiment 1 EX | Experiment 1 EX V2 |
|---|---|---|
| 总数据量 | 1M–6M，每种数据量扫描 7 个比例，共 42 组 | 固定 3M，扫描 11 个比例 |
| A/B 比例 | 100/0、95/5、90/10、85/15、80/20、75/25、70/30 | 保留原 7 个比例，新增 65/35、60/40、55/45、50/50 |
| Upstream loss 权重 | 从共享 Parallel 组件继承 1、0.5、1.5 | 每个实验 JSON 显式固定 1、0.5、1.5，并记录到 markers |
| 实验与输出身份 | `experiment1_ex_p122k_...` | 独立的 `experiment1_ex_v2_p122k_...` 及 comparison group |
| Split 目录 | `experiment1_ex_n3m_a..._b...` | 独立的 `experiment1_ex_v2_n3m_a..._b...` |

Loss 权重的数值没有改变；v2 将其显式写入配置，便于检查实验条件。
122,077 参数的模型结构、总数据 70/10/20 协议、A/B 共享验证、
`data_seed=42`、A-train 归一化、类别权重和 5 × 5 seed 配对保持一致。
下游 recipes 沿用当前 `dnn_default.json`，没有在 v2 中另行修改。
已有比例在去除实验身份与 split 目录差异后，当前解析出的训练设置一致。
独立 split 目录不代表索引内容一定不同；与历史划分是否一致，需核对
实际生成的 `index_sha256`。

### 当前训练流程相对于早期版本的修订

V2 配置继承当前代码中已经合入的两项训练流程修订：

- `ed39f64` 引入验证集驱动的 `ReduceLROnPlateau` 学习率衰减。
  Upstream 监控 A-val 加权 jet CE，patience 为 8；tabular DNN 和
  graph-DNN 监控 B-val 加权 jet CE，patience 为 4。
  两阶段初始 LR 均为 `1e-3`，衰减系数 0.5，最低 LR 为 `1e-5`，
  relative threshold 为 `1e-4`，cooldown 为 0。
- `83740c3` 将下游 tabular DNN 和 graph-DNN 的 jet CE 与 upstream
  的类别权重对齐，当前 b/c/light 权重为 `[2, 2, 1]`。
  同一加权 CE 用于下游训练、B-val checkpoint 选择、early stopping 和
  学习率调度；下游监督仍只有 jet 标签，不增加 origin/pair 标签监督。

这些修订已经作用于共享组件和训练入口，因此当前重新运行原
`experiment1_ex` 配置也会使用它们；它们并非只对 v2 生效。
与早期已完成结果比较时，应核对该次运行保存的 resolved config、
loss/class weights、scheduler 信息及代码版本。本次未审计历史运行产物，
不据旧实验名推断某个历史结果是否使用了上述修订。

## 继承设置与实验标识

所有配置引用现有组件：

- `../../data/experiment1_ex_n3m.json`
- `../../parallel/experiment1_p122k.json`
- `../../refiner/dnn_default.json`

通过 `overrides.data` 指定各配置的 A/B 样本数及独立 split 目录，
通过 `overrides.parallel.loss_weights` 固定三个 loss 系数。
其余设置沿用组件，包括 `data_seed=42`、模型结构、类别权重、优化器、
学习率调度、upstream seeds 1–5 和 downstream seeds 1–5 的 Cartesian 配对。
非零 B-train 配置继承 recipes：
`F1_embed`、`F3_embed_aux`、`F4_all`、`FG2`、`FG2s`、`FG4`。

每个配置使用独立的 v2 实验名和 split 目录：

```text
结果：/data/yuyang/SerialFlavour/results/parallel_refine/<experiment.name>/
划分：/data/yuyang/SerialFlavour/parallel_refine/cache/splits/experiment1_ex_v2_n3m_a<AAA>_b<BBB>/
```

共同的 `comparison_group` 为
`experiment1_ex_v2_p122k_n3m_training_allocation`。
Markers 记录模型参数量、总数据量、训练池大小、A/B 比例和样本数、
下游启用状态及三个 loss 系数。
不同 A/B 比例对应不同的 split 请求；各自的数据索引和 processed cache
需在准备阶段生成并根据缓存标识复用，不假设不同配置使用同一份 A-train。

## GPU 生产训练脚本

入口为 `scripts/production/run_experiment1_ex_v2_full.sh`，格式沿用
Experiment 3 的 production 脚本，`all_configs` 显式列出 11 个配置，
按本 README 表格中从 100/0 到 50/50 的顺序排列。
复用 `scripts/production/run_experiment1_ex_recipe_queue.py` 和
`run_experiment1_ex_seed.py`，部署时需要这两个 worker 及其训练/评估依赖。

在 Linux GPU 节点激活训练环境后，从仓库根目录运行：

```bash
# 校验全部配置并显示本节点命令，不写日志、生成数据或启动训练。
GPU_POOL="0 1 2 3" bash scripts/production/run_experiment1_ex_v2_full.sh --dry-run

# 只准备数据。
PREPARE_WORKERS=8 bash scripts/production/run_experiment1_ex_v2_full.sh --prepare-only

# 运行完整训练、缓存生成、Y-test 评估和适用的跨 seed 聚合。
GPU_POOL="0 1 2 3" bash scripts/production/run_experiment1_ex_v2_full.sh

# 后台运行。
mkdir -p logs/parallel_refine/experiment1_ex_v2
nohup env GPU_POOL="0 1 2 3" PYTHON=python \
  bash scripts/production/run_experiment1_ex_v2_full.sh \
  > logs/parallel_refine/experiment1_ex_v2/launcher.log 2>&1 &
```

共 55 个 upstream 训练；10 个非零 B-train 配置各有
6 recipes × 5 upstream seeds × 5 downstream seeds，共 1,500 个 downstream
训练。`100/0` 仅准备 A-train、A-val 和 Y-test 的 processed cache，
跳过 B 缓存、下游训练和 DNN 对比结果生成。

数据准备按本节点配置顺序串行执行；之后队列按
`Parallel → B cache → recipe → Y cache → evaluation → prune` 依赖派发。
训练并发进行，配置之间可以交错执行，不等待前一配置全部结束。
每张 GPU 同时接收一个队列单元；每个 recipe 单元沿用现有 worker，
在同一 GPU 并发运行 5 个 downstream seeds。
`REFINE_GPUS` 默认为 GPU 数减一（最小为一），在 upstream 仍排队时限制
下游并发，upstream 队列清空后可使用其余 GPU。

多节点通过 `NODE_COUNT`（1–11）和 `NODE_RANK`（从 0 开始）按
`all_configs` 下标取模分片，各节点使用同一 `NODE_COUNT` 和不同 rank，
并访问相同的数据、processed cache、split 和结果路径。
这是独立任务并行，不是 DDP；GPU 由用户或作业系统提前分配。
例如四节点中的第一个节点运行：

```bash
NODE_COUNT=4 NODE_RANK=0 GPU_POOL="0 1 2 3" \
  bash scripts/production/run_experiment1_ex_v2_full.sh
```

共享 processed-cache 目录的 `.locks/experiment1_ex_v2/` 中保存
数据准备锁和各节点分片启动锁，文件系统必须支持跨节点 `flock`。
各 A/B 比例使用自己的 split 和 A-train 归一化；prepared cache 仅按
实际数据标识命中复用，不能假定 11 组配置共用同一份预处理数据。

默认日志和状态目录为
`logs/parallel_refine/experiment1_ex_v2/full/node<RANK>/`，包括 `stage1/`、
`scheduler.log`、`failed_jobs.log`、`units/`、`state/` 和 `aggregate/`。
重跑跳过已完成 seed，复用已完成 checkpoint；不完整训练阶段从头重训，
不恢复 epoch 或优化器状态。默认每阶段额外重试一次，耗尽后返回非零退出码。
每个 seed 评估成功后清理其 frozen/graph B/Y 缓存，保留 processed cache、
归一化统计、checkpoint、结果和日志。非零 B 配置在全部 5 个 upstream seeds
流程完成后生成 DNN 对比的跨 seed 聚合。

`GPU_POOL` 为不同数字设备 ID 的空格分隔列表；未设置时优先使用数字形式的
`CUDA_VISIBLE_DEVICES`，否则默认 `0 1 2 3`。
`PYTHON`、`PREPARE_WORKERS`、`RETRIES`、`PATIENCE`、`REFINE_GPUS`、
`LOG_BASE`、`MIN_FREE_GB`、`SPACE_PATH` 的含义见脚本 `--help`。
默认在准备和队列启动前检查 `/data` 至少有 200 GiB 空闲；这是启动门槛，
不是整个 11 组实验的空间估计。

本地仅验证配置、Bash 语法和 dry-run 调度命令，未准备真实数据、
训练模型或进行 Y-test 评估。
