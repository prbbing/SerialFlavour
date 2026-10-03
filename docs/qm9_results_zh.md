# QM9 本地 smoke 结果

本文合并两份本地 smoke 记录：最初的 μ 闭环（验证通用 pipeline，设计已被取代）与当前的 gap + 逐原子电荷/键级闭环。设计与配置见 `qm9_experiment_zh.md`。两者都是单种子工程验证，不是机制结论。

环境统一为 WSL conda `gn2_study_cross`：Python 3.11.6、torch 2.5.1、numpy 2.4.6、rdkit 2025.09.3；CPU、FP32、4 线程。

## 1. 初始 μ 闭环（设计已由 gap 取代）

目的：验证通用 pipeline（download/prepare/train/cache/refine/evaluate/analyze）与 A/B/Y 冻结协议。

- 任务：主 μ（Debye），辅助 α/R²/Cv（分子级标量，旧设计）。
- 配置：`config/qm9_smoke.json`；划分 A 2048 / A-val 256 / B-train 1024 / B-val 256 / Y 512。
- 代码：当时的 `MT-embed-aux` 把辅助标量预测与 `g` 拼接。

| 方法 | μ MAE (Debye) | RMSE | R² | 上游/下游参数 |
|---|---:|---:|---:|---:|
| ST-native | 0.800650 | 1.042055 | 0.427 | 62977 / 0 |
| ST-embed | 0.752008 | 0.986221 | 0.487 | 62977 / 6465 |
| MT-native | 0.798538 | 1.045139 | 0.424 | 81796 / 0 |
| MT-embed | 0.776343 | 1.013424 | 0.458 | 81796 / 6465 |
| MT-embed-aux | 0.777088 | 1.014868 | 0.457 | 81796 / 6465 |

辅助原单位 MAE：α 2.489297 Bohr³、R² 115.927415 Bohr²、Cv 1.324538 cal/(mol K)。配对增量：`readout_gain` +0.022195、`auxiliary_readout_gain` −0.000745、`auxiliary_supervision_gain` −0.024335 Debye。阶段耗时约：prepare 19.6s、train 49.3s、cache 4.2s、refine 4.3s、evaluate 1.5s。

结论：闭环可运行；但该设计的辅助是分子级标量且由 `g` 计算，`MT-embed-aux` 无增量，与理论一致。已被 gap + 局部辅助设计取代。

## 2. gap + 逐原子电荷/键级 smoke

- 配置：`config/qm9_gap_charge_bond.json`；主 gap（eV）、辅助逐原子 Mulliken 电荷(T3) + 全对键级(T5)。
- 划分 A 2048 / A-val 256 / B-train 1024 / B-val 256 / Y 512（非共享验证）。
- 候选 8192，其中 516 个在 SMILES/键感知阶段失败被排除，无重复；类别计数 `[无键 574681, 单 68137, 双 2943, 三 1168, 芳香 3963]`。
- 上游 ST 62,977 / MT 87,943 参数。

Y-test（512 分子，gap MAE eV）：

| 方法 | MAE | RMSE | R² | 下游参数 |
|---|---:|---:|---:|---:|
| ST-native | 0.544754 | 0.7000 | 0.713 | 0 |
| ST-R0 | 0.530961 | 0.6839 | 0.726 | 6,913 |
| ST-R3 | 0.495593 | 0.6508 | 0.752 | 14,593 |
| MT-native | 0.504927 | 0.6536 | 0.750 | 0 |
| MT-R0 | 0.463258 | 0.6175 | 0.777 | 6,913 |
| MT-R1 | 0.836909 | 1.0286 | 0.381 | 40,129 |
| MT-R2 | 0.453818 | 0.6110 | 0.782 | 6,913 |
| MT-R3 | **0.410426** | 0.5572 | 0.818 | 14,593 |
| MT-R4 | 0.438076 | 0.5856 | 0.799 | 39,489 |
| MT-R4-shuffle | 0.429705 | 0.5763 | 0.806 | 39,489 |

辅助诊断（Y-test）：逐原子电荷 MAE 0.657(e)；键存在 accuracy 0.705；真实键上键级 accuracy 0.908（9,608 对）。

配对增量（正值表示前者 MAE 更低）：

| 增量 | 定义 | eV |
|---|---|---:|
| `pool_to_local_gain` | MT-R0 − MT-R3 | +0.0528 |
| `predicted_local_gain` | MT-R3 − MT-R4 | −0.0277 |
| `auxiliary_readout_gain` | MT-R0 − MT-R2 | +0.0094 |
| `auxiliary_supervision_gain` | ST-R0 − MT-R0 | +0.0677 |

观察：多任务上游（MT）在相同读出下优于单任务（ST）；H-only（R3）优于 g-only（R0）；`R2` 相对 `R0` 仅很小增量；`R4` 与 `R4-shuffle` 接近且都未超过 `R3`，本规模下预测键图未带来超出 H-only 的增量；`R1` 最弱。

阶段耗时约：prepare 21.0s、train 50.2s、cache 5.6s、refine 121.7s、evaluate 3.2s、analyze 0.3s。产物在 `cross-domain/results/qm9/qm9_gap_charge_bond/`。

**重要**：以上 gap 数字来自集群重构前的代码（set/GNN 仍为全批、无 dropout/早停）。此后实现了分子分块 mini-batch、dropout、早停与共享验证，需要重跑本地 smoke 复核；集群全量结果以 `results/qm9/qm9_gap_charge_bond_full/` 为准。

## 3. 局限

- 单种子、短训练，仅证明闭环可运行；`sample_sd` 未估计，不判断统计显著性。
- 容量未严格匹配（R1/R4 约 4 万 vs R3 1.46 万）；R4 与 shuffle 接近，预测键图增量不成立。
- 键存在与几何高度相关，不能作为机制证据；机制应看真实键上键级与 `R4−R3`。
- 候选失败分子需在正式运行前核查分布。
- 不涉及骨架/构象外推，也不声称辅助任务含不可恢复的独有信息。
