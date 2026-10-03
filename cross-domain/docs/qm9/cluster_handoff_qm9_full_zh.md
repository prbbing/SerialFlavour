# 集群运行速查：QM9 gap + 电荷/键级全量

本文只讲集群上的操作步骤，设计与字段解释见 `qm9_experiment_zh.md`。目标实验：QM9 主任务 gap，辅助逐原子 Mulliken 电荷与全对键级，冻结后读出对照 R0–R4。

配置：`cross-domain/experiments/qm9/config/qm9_gap_charge_bond_full_100k.json`（100k 分子，5×5 seed，共享验证）。日志：`logs/qm9/qm9_gap_charge_bond_full/`；产物：`results/qm9/qm9_gap_charge_bond_full/`。

## 1. 集群准备

1. 修改配置中的 `data_root` 为集群 QM9 数据目录（当前是 WSL 路径），按需改 `output_root`/`log_root`/`runtime.threads`。
   - **不要改** `experiment` 名称；代码改动使 identity 变化时必须换新 experiment 名。
2. 环境：集群 `gn2_study_cross`（或等价）含 torch(CUDA)、numpy、rdkit（需 `rdkit.Chem.rdDetermineBonds`）。用 `PYTHON`/`CONDA_ENV` 指向该环境。
3. 原始数据放在 `data_root/raw/`（`dsgdb9nsd.xyz.tar.bz2`、`uncharacterized.txt`）；缺失时 `prepare` 自动下载并校验 MD5。

## 2. 最小启动检查（强烈建议先做）

先只跑 `prepare` 与一个 seed，确认真实数据上无误，再起整个 pool（pool 自动跳过已完成项）：

```bash
cd <仓库根>
python cross-domain/scripts/run_unit.py --config cross-domain/experiments/qm9/config/qm9_gap_charge_bond_full_100k.json --unit prepare
# 检查 results/qm9/qm9_gap_charge_bond_full/data/preparation_manifest.json 与 logs/.../units/prepare.json
python cross-domain/scripts/run_seed.py --config cross-domain/experiments/qm9/config/qm9_gap_charge_bond_full_100k.json --variant multi_task --seed 1 --gpu 0
# 检查 refiners/ 与 logs/.../seeds/multi_task__seed1.json
```

`prepare` 需对约 13 万候选做键感知，预计数分钟。日志：`logs/qm9/qm9_gap_charge_bond_full/prepare.log`、`seed/multi_task__seed1.log`。

## 3. 正式运行

```bash
CONDA_ENV=gn2_study_cross GPU_POOL="0 1 2 3" RETRIES=1 bash cross-domain/experiments/qm9/scripts/run_full.sh
```

或：

```bash
python cross-domain/scripts/run_pool.py \
  --config cross-domain/experiments/qm9/config/qm9_gap_charge_bond_full_100k.json \
  --gpus 0 1 2 3 --retries 1
```

- 每个 seed 单元（`variant × seed`，共 10 个）独占一卡；`GPU_POOL` 决定并发数。
- 不要为同一 experiment 同时运行两个 pool 进程。
- 全量配置共 223 个单元（prepare 1 + upstream 10 + cache 10 + refine 200 + evaluate 1 + analyze 1）。

## 4. 续跑

重跑同一条命令即可：`prepare` marker 与已完成 seed marker 会跳过；单 seed 内已存在的 `best.pt`/缓存也跳过。marker 位置：`logs/.../units/`、`logs/.../seeds/`；调度日志 `logs/.../scheduler.log`。

## 5. 产物

`results/qm9/qm9_gap_charge_bond_full/`：

- `src/data/preparation_manifest.json`：来源、许可、SHA256、划分计数、类别计数、无效候选。
- `upstream/<variant>/seed<k>/`、`cache/<variant>/seed<k>/`（冻结特征，不缓存辅助真值）。
- `refiners/<variant>/seed<u>/<recipe>/seed<d>/`。
- `evaluation/{metrics.json,metrics.csv,auxiliary_metrics.csv,*_predictions.csv}`。
- `src/analysis/{summary.json,summary.md,methods.csv,paired_deltas.csv}`。

判读方式（方法命名、对比含义、指标方向）见 `qm9_experiment_zh.md` 与 `qm9_smoke_test_results_zh.md`。

## 6. 故障处理

- **`prepare` 报 "only N valid unique candidates"**：有效唯一分子不足 100k → 降低 `y_test` 或 `a_train`，换新 experiment 名重跑；不要删除并复用同名目录。
- **CUDA 报 `use_deterministic_algorithms` 不支持**：把 `runtime.deterministic` 设为 `false` 并记录。
- **OOM**：减小 `upstream.batch_size`/`refiner.batch_size` 或数据规模。
- **rdkit 缺 `rdDetermineBonds`**：升级 rdkit（本机 2025.09.3）。
- **identity 变化报错**：换新 experiment 名。
- **某 seed 失败**：pool 重试 `RETRIES` 次后跳过继续；因 `evaluate` 需要全部 refiner，失败 seed 会导致聚合缺项。修复后重跑 pool（自动续跑）。
- **磁盘**：原子 embedding 缓存最占空间（每 `(variant, seed)` 数百 MB，总计低十 GB 量级）。

## 7. 已知边界与不要做

- 共享验证（10k 同时用于上游/下游选择）、单节点多卡（非批处理）、smoke 规模上游模型（63k/88k 参数）+ 56k 训练数据；R1/R4 与 R3 容量未严格匹配。
- **验证状态**：离线 pytest 14 项通过；真实数据端到端 smoke 尚未完成，故必须先执行第 2 节最小启动检查。
- 不要修改外部 `src/`、`scripts/`、`configs/`（Jet tagging）。
- 不要删除运行中的 processed 缓存或 results 目录；不要在 identity 改变后复用旧 experiment 名。
