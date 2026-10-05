# MP＋CrystalTransformer 集群运行结果（v2）

更新：2026-10-05。本文记录 `mp_crystran_cluster_20k_v2` 在本集群上的**实际完整运行**：数据准备、15 次上游训练、15 次缓存、225 次冻结读出、最终 Y 评估与分层分析，共 258 个 work unit。与 [集群运行说明](cluster_run_zh.md) 的静态计划相比，本轮为真实 GPU 多 seed 执行证据；历史 CPU smoke 仍只对应各自的小规模配置。

## 1. 运行身份与产物位置

| 项目 | 值 |
|---|---|
| experiment | `mp_crystran_cluster_20k_v2` |
| identity | `5b054c56403abb7022663c06f18593b4a2b1b4d218d6c86556dcb3848585a33c` |
| config SHA256 | `8edbc8a2c7fe8e7b678f048d154fee3dd4958bb49841813583cf696021a60316` |
| code SHA256 | `ecbd155b422720c3fe7e7c6da99b791f8ad34cfbd9b7cbc9b36df1a4912b06ce` |
| 结果目录 | `/data/yuyang/SerialFlavour/results/sci_mp/sci_mp_crystran/mp_crystran_cluster_20k_v2`（约 2.2 GB） |
| 日志目录 | `/data/yuyang/SerialFlavour/logs/sci_mp/sci_mp_crystran/mp_crystran_cluster_20k_v2` |
| 数据目录 | `/home/yuyang/SerialFlavour/cross-domain/local_data/sci_mp_crystran` |
| 完成标记 | `cluster_complete.json`（`status: complete`，258 units） |

关键产物：`run_manifest.json`、`resolved_config.json`、`download_manifest.json`、`prepared.json`（约 237 MiB）、`split_manifest.json`、`target_normalization.json`、`evaluation.json`、`predictions.npz`、`summary.json`、`results_zh.md`。

## 2. 环境与硬件

- 操作系统：Linux 6.8.0，glibc 2.39；4× NVIDIA A10（23 GB，驱动 565.57，CUDA 12.7）。
- conda 环境：`gn2_study_cross`（`/data/yuyang/miniconda3/envs/gn2_study_cross`）。
- Python 3.11.16；PyTorch **2.2.1+cu121**；NumPy **1.26.4**。
- `cuda_available: true`（记录于 `run_manifest.json`）。
- 调度：`GPU_POOL="0 1 2 3"`，每张 GPU 同时一个 `(variant, upstream_seed)` worker。

**依赖注意事项**：`requirements.txt` 原先未约束 NumPy，安装后会被升级到 2.x，导致环境内 torch 2.2.1 报 `Numpy is not available`。本轮已回退并写入 `numpy>=1.25,<2`（pymatgen 允许 `>=1.25`，torch 2.2.1 要求 `<2`）。复现时请保留该约束。

**数据获取注意事项**：本集群到 Figshare/S3 的单连接吞吐约 14 KB/s，直接 `download()` 不可行。已用 64 线程分块下载完成并校验 SHA256。`prepare` 内部再次校验同一文件（`184,077,884` bytes，SHA256 `1f3de2dc…0500`）。

## 3. 数据准备

由 `prepare` unit 合并 download 校验与 prepare：

| 指标 | 值 |
|---|---:|
| 扫描记录 | 83,989 |
| 候选池（2×） | 40,000 |
| 结构去重后候选 | 38,935 |
| 等价结构剔除 | 1,065 |
| 原子数过滤剔除 | 7,857 |
| 最终材料 | 20,000 |

划分（材料级、锁定 seed 2026）：`a_train=11,200`、`a_val=1,000`、`b_train=2,800`、`b_val=1,000`、`y_test=4,000`。A_train 上拟合的标准化：Eg `mean=1.069463 eV, std=1.495801`；Ef `mean=-1.410180 eV/atom, std=1.264681`。

## 4. 结果

先在每个上游 seed 内平均 5 个下游 seed，再在 5 个上游 seed 间报告均值与样本 SD（`ddof=1`）。Y 集 MAE（eV）：

| 上游 | 读出 | Y MAE mean | upstream SD | RMSE mean | R² mean |
|---|---|---:|---:|---:|---:|
| single_task | native | 0.580174 | 0.012731 | 0.983500 | 0.591272 |
| single_task | embedding | 0.580936 | 0.013041 | 0.977049 | 0.596581 |
| single_task | embedding_capacity | 0.581756 | 0.013139 | 0.976245 | 0.597234 |
| mt_main_only | native | 0.583045 | 0.003370 | 0.993312 | 0.583372 |
| mt_main_only | embedding | 0.584771 | 0.002065 | 0.983449 | 0.591588 |
| mt_main_only | embedding_capacity | 0.583803 | 0.002388 | 0.982053 | 0.592752 |
| multi_task | native | **0.538222** | 0.008495 | 0.953544 | 0.616033 |
| multi_task | embedding | 0.539733 | 0.010975 | 0.926069 | 0.637757 |
| multi_task | embedding_capacity | 0.538423 | 0.011422 | 0.925690 | 0.638060 |
| multi_task | embedding_aux | 0.539977 | 0.012988 | 0.927634 | 0.636506 |
| multi_task | embedding_aux_shuffle | 0.538932 | 0.012716 | 0.925828 | 0.637930 |
| multi_task | embedding_hidden | 0.540170 | 0.012073 | 0.927884 | 0.636318 |

配对 MAE 差值（前者减后者，负值表示改善）：

| 对比 | mean Δ (eV) | upstream SD | 改善的上游 seeds |
|---|---:|---:|---:|
| multi_task − matched main-only（native） | −0.044823 | 0.009516 | 5/5 |
| multi_task − single_task（native） | −0.041953 | 0.010223 | 5/5 |
| multi_task embedding − native | +0.001512 | 0.003447 | 1/5 |
| multi_task aux − embedding_capacity | +0.001554 | 0.002367 | 2/5 |
| multi_task aux − embedding | +0.000244 | 0.002235 | 2/5 |
| multi_task aux − shuffle | +0.001045 | 0.000747 | 0/5 |
| multi_task aux − hidden | −0.000193 | 0.001677 | 3/5 |

## 5. 解读与限制

- **上游多任务本身有效**：`multi_task` 的 native Y MAE 显著低于 `mt_main_only`（同架构、仅主损失）和 `single_task`，差值约 0.042–0.045 eV，且在 5/5 上游 seed 上方向一致。这是本轮最主要、最稳定的观察。
- **冻结读出没有带来 MAE 增益**：`multi_task` 上各读出相对 native 的 MAE 差异量级仅约 0.0002–0.0016 eV，方向在 seed 间不一致（改善 seed 数 0–3/5）。按配对定义，`aux` 相对 `shuffle` 甚至在 5/5 上更差。当前证据不支持“辅助预测读出在冻结条件下提供增量”。
- **RMSE/R² 与 MAE 方向相反**：各 embedding 读出的 RMSE（约 0.926–0.928）与 R²（约 0.636–0.638）优于 native（0.9535 / 0.6160），但 MAE 略差，符合残差从 native 起点校准时的指标分歧，不能据此判定读出更优。
- **统计口径**：25 个下游组合不是 25 个独立上游重复；本表只报告上游 seed 间的均值与样本 SD，不做显著性检验或置信区间，也不自动选择主/辅任务。
- **协议边界**是材料级随机划分，未做组成不重叠的泛化协议；架构为本项目缩小规模 v2（宽 64/4 层/4 heads/FFN 128，读出 128/64/32），不是作者默认规模的精确复现；未核验对称性/几何不变性，也未使用预训练权重。
- 全部结论限定在这一锁定划分、20000 材料、5×5 seed 设置下的描述性统计。

## 6. 运行成本与稳健性

| 阶段 | unit 数 | 合计时间 | 单 unit 最大 |
|---|---:|---:|---:|
| prepare（校验＋去重） | 1 | 21.3 min | 21.3 min |
| upstream | 15 | 124.3 min | 17.7 min |
| cache | 15 | 2.6 min | 0.2 min |
| refine | 225 | 34.7 min | 0.2 min |
| evaluate | 1 | 0.8 min | 0.8 min |
| analyze | 1 | ~0 min | ~0 min |

- 258 units 全部 `status: complete`，0 失败，0 重试；`cluster_complete.json` 已写出。
- 上游均早停于 69–221 epoch（上限 500），说明 early stopping 正常触发。
- 从 prepare 结束到 analyze 完成墙钟约 **57.5 分钟**；含 prepare 的完整流程约 **79 分钟**（4× A10）。聚合 unit 时间为 3.06 h。
- 训练中出现的 pymatgen 稀有气体电负性 `NaN` 警告为已知良性提示，不影响结果。

## 7. 复现命令

```bash
cd /home/yuyang/SerialFlavour
PYTHON=/data/yuyang/miniconda3/envs/gn2_study_cross/bin/python \
GPU_POOL="0 1 2 3" RETRIES=1 \
  bash cross-domain/experiments/sci_mp_crystran/scripts/run_cluster.sh
```

同一命令重启会逐文件校验并跳过已完成单元（工作单元级续跑，非 optimizer/epoch 状态恢复）。完整矩阵、参数量与 seed 设计的静态计划见 [cluster_run_zh.md](cluster_run_zh.md) 与 [cluster_plan_static.json](cluster_plan_static.json)；自动生成的原始结果表见结果目录 `results_zh.md` 与 `summary.json`。
