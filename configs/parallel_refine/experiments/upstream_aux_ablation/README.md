# 3M、100/0 上游辅助任务消融

比较辅助监督对上游 jet 分类的作用。新增三个配置，与现有
`../experiment1_ex_v2/experiment1_ex_v2_p122k_n3m_a100_b000.json`
联合构成四组比较：

| 配置文件后缀 | jet 权重 | origin 权重 | vertex（pair）权重 |
|---|---:|---:|---:|
| 现有 EX V2 基线 `a100_b000` | 1 | 0.5 | 1.5 |
| `jet_origin` | 1 | 0.5 | 0 |
| `jet_vertex` | 1 | 0 | 1.5 |
| `jet_only` | 1 | 0 | 0 |

新配置文件名前缀均为 `upstream_aux_ablation_p122k_n3m_a100_b000_`。
单辅助任务沿用基线中该任务的 loss 权重，不重新缩放总损失。

## 固定条件

- 3,000,000 jets：A-train 2,100,000，B-train 0，共享验证集 300,000，Y-test 600,000。
- 直接复用基线 split 目录
  `/data/yuyang/SerialFlavour/parallel_refine/cache/splits/experiment1_ex_v2_n3m_a100_b000`，
  保持 data seed 42、采样、A-train 归一化和事件互斥规则一致。
- 复用 `experiment1_p122k.json`：同一 122,077 参数架构、初始化 seeds 1–5、
  batch size 512、100 epochs、AdamW、学习率调度和类别权重。
- 使用相同的 A-val jet CE 选择 `best_jet.pt`，学习率调度也监控 jet CE。
- B-train 为零；不训练任何 downstream refiner。公共 refiner 组件仍被引用以满足配置格式。
- 保留所有 task heads。权重为零时，现有 `parallel_losses` 跳过对应损失计算，
  对应 head 不接收损失梯度；forward 仍计算这些 head 的输出。
  因此这是辅助监督消融，不是删除 head 的架构或训练开销比较。

每个新配置有独立实验名和结果目录，新增共 15 次上游训练。基线有 5 次上游训练；
若已有运行的数据索引、归一化、训练设置和代码版本与当前条件一致，可复用它们。
运行前核对基线和新配置产物中的 split/index hashes；共享路径本身不是历史产物一致的证据。

## 运行

完整入口为 `scripts/production/run_upstream_aux_ablation_full.sh`，默认调度新增
三组配置、每组 seeds 1–5，共 15 次上游训练。现有完整辅助任务基线不在队列中。
流程为数据准备（含 processed cache）→ 上游训练 → Y-test 特征缓存 →
逐 seed 评估 → 清理该 seed 的冻结缓存；保留 checkpoint、processed cache 和评估结果。
现有队列对 B-train=0 跳过下游专用的跨 seed 聚合，未生成四组比较汇总。

```bash
# 校验三个配置并打印命令；不写日志或启动训练。
GPU_POOL="0 1 2 3" bash scripts/production/run_upstream_aux_ablation_full.sh --dry-run

# 仅准备数据及共享 processed cache。
PREPARE_WORKERS=8 bash scripts/production/run_upstream_aux_ablation_full.sh --prepare-only

# 完整运行，包括 Y-test 评估。
GPU_POOL="0 1 2 3" bash scripts/production/run_upstream_aux_ablation_full.sh

# 后台运行。
mkdir -p logs/parallel_refine/upstream_aux_ablation
nohup env GPU_POOL="0 1 2 3" PYTHON=python \
  bash scripts/production/run_upstream_aux_ablation_full.sh \
  > logs/parallel_refine/upstream_aux_ablation/launcher.log 2>&1 &
```

该入口面向 Linux GPU 节点，复用现有 recipe queue 和 seed worker。
默认 `RETRIES=1`、`PREPARE_WORKERS=8`、`MIN_FREE_GB=200`、
`SPACE_PATH=/data`。使用已分配的 GPU；每张 GPU 同时执行一个阶段。
支持 `NODE_COUNT=1..3` 和 `NODE_RANK` 按配置分片，多节点需共享数据及结果目录，
并使用支持 `flock` 的文件系统。重跑复用已完成阶段；失败时返回非零状态。

以下是仅训练、不执行 Y-test 评估的手动方式。

在 Linux 训练环境、仓库根目录执行。先准备共用数据：

```bash
config=configs/parallel_refine/experiments/upstream_aux_ablation/upstream_aux_ablation_p122k_n3m_a100_b000_jet_only.json
python scripts/prepare_data.py --config "$config"
```

用现有生产 worker 运行单个 seed；`--stage parallel` 仅训练上游，不读取 Y-test：

```bash
python scripts/production/run_experiment1_ex_seed.py \
  --config "$config" --seed 1 --gpus 0 --stage parallel --dry-run

python scripts/production/run_experiment1_ex_seed.py \
  --config "$config" --seed 1 --gpus 0 --stage parallel
```

替换配置后缀并运行 seeds 1–5。训练入口默认 early-stopping patience 为 100，
在当前最多 100 epochs 的设置下不会提前停止；消融脚本不另行设置该参数。
若要执行包括 Y-test 评估的完整生命周期，使用 `--stage full`；应先在验证集上
确定指标和比较方案，再统一执行测试集评估。

## 结果解释

预先固定 jet 分类的主要指标（例如指定 signal efficiency 下的背景 rejection），
同时报告其他 flavour 的变化。以相同 seed 的指标差值比较四组，报告各 seed、
均值及波动，不从辅助任务的存在预设提升结论。

`jet_origin` 和 `jet_vertex` 分别相对 `jet_only` 衡量单个辅助任务的效果；
完整基线相对两个单辅助任务组衡量加入另一个辅助任务后的增量效果。
结果限定于当前权重、数据规模和模型设置；没有提升也可能涉及任务权重的选择，
不能直接推广为辅助任务普遍无效。
