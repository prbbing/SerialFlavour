# QM9 gap + 电荷/键级：refine-v2 协议 20k / 50k / 100k 结果（中文）

本文件记录修正后的 post-refinement 协议（v2）在三个规模上的完整运行与评估。配置见
`cross-domain/config/qm9_gap_charge_bond_refine_v2_20k.json`、`..._50k.json`、`..._100k.json`；
集群副本为对应的 `*_cluster.json`（仅 `data_root`/`output_root`/`log_root` 不同）。设计与字段解释见
`qm9_experiment_zh.md`，历史（旧协议）结果见 `qm9_gap_charge_bond_full_results_zh.md`。

运行时间线：2026-10-03 11:09 启动 20k，2026-10-04 11:25 完成 100k 的 analyze，总墙钟约 **24 小时**
（单节点 4×A10，跨实验全局队列调度）。三档均 **10/10 seed 成功、无失败**，各产出 `evaluation/metrics.csv`
**410 行 / 18 个方法**，并生成完整 `analysis/`。

---

## 1. 协议要点（相对历史版本的修正）

- **独立验证**：`shared_validation=false`，A-val 与 B-val 分离（20k：1000/1000；50k：2500/2500；100k：5000/5000）。
- **更长上游训练**：`upstream.epochs=500`（历史为 100），早停 patience=20。
- **LR 衰减**：`ReduceLROnPlateau`（上游 patience=8、下游 patience=4，factor=0.5，`min_lr_ratio=0.01`），
  由 `src/pipeline/fit.py` 实现；早停计数不因衰减重置。
- **native 初始化读出**：`R0-native`/`R2-native` 以 native 头初始化并吸收 A→B 标准化，训练前验证复现 native 物理预测。
- **同容量图消融**：`R3-graph`/`R4-nocharge`/`R4-uniform`/`R4-existence`/`R4`/`R4-shuffle` 使用相同节点输入/层数/容量；
  `R4-shuffle` 在分子内部打乱 pair 概率行。
- **预测量口径**：电荷反标准化为 e 后缓存；图配方消费 4 个「有键」类别的 soft 概率。

划分（每个规模总分子数 = 各项之和）：

| 规模 | a_train | a_val | b_train | b_val | y_test |
|---|---:|---:|---:|---:|---:|
| 20k | 11,200 | 1,000 | 2,800 | 1,000 | 4,000 |
| 50k | 28,000 | 2,500 | 7,000 | 2,500 | 10,000 |
| 100k | 56,000 | 5,000 | 14,000 | 5,000 | 20,000 |

---

## 2. 上游训练与 LR 衰减

5 个上游 seed 的均值（全部 5/5 触发早停）：

| 规模 | variant | epochs_run 范围（均值） | best a_val MAE (eV) | LR 衰减次数 | 训练耗时/seed（秒） |
|---|---|---:|---:|---:|---:|
| 20k | single_task | 152–164 (158) | 0.16515 | 4–7 | ~650 |
| 20k | multi_task | 152–213 (179) | 0.16436 | 6–7 | ~688 |
| 50k | single_task | 139–152 (143) | 0.11100 | 4–6 | ~1,277 |
| 50k | multi_task | 165–283 (196) | 0.11207 | 6–7 | ~2,944 |
| 100k | single_task | 139–167 (151) | 0.08643 | 5–7 | ~3,709 |
| 100k | multi_task | 219–302 (252) | 0.08682 | 7–7 | ~7,036 |

要点：LR 衰减使上游在约 140–300 轮收敛（历史 100 轮时 best 常落在最后几轮、属未收敛）；`multi_task`
需要明显更多的轮数才收敛（100k：252 vs ST 151），说明辅助头增加了优化难度。

---

## 3. 主任务结果（y_test MAE，单位 eV）

方法按规模聚合：先在每个上游 seed 内对 5 个下游 seed 求平均，再对 5 个上游 seed 求均值。

| 方法 | 20k | 50k | 100k |
|---|---:|---:|---:|
| ST-native | 0.1638 | 0.1138 | **0.0874** |
| ST-R0 | 0.2063 | 0.1516 | 0.1191 |
| ST-R0-native | 0.1636 | 0.1132 | 0.0874 |
| ST-R3 | 0.2105 | 0.1538 | 0.1208 |
| ST-R3-graph | 0.2133 | 0.1553 | 0.1219 |
| MT-native | 0.1632 | 0.1150 | 0.0879 |
| MT-R0 | 0.2112 | 0.1584 | 0.1268 |
| MT-R0-native | 0.1622 | 0.1146 | 0.0880 |
| MT-R1 | 0.3088 | 0.2809 | 0.2434 |
| MT-R2 | 0.2073 | 0.1564 | 0.1264 |
| MT-R2-native | 0.1621 | 0.1145 | 0.0880 |
| MT-R3 | 0.2205 | 0.1659 | 0.1323 |
| MT-R3-graph | 0.2208 | 0.1707 | 0.1324 |
| MT-R4 | 0.2137 | 0.1605 | 0.1274 |
| MT-R4-nocharge | 0.2122 | 0.1607 | 0.1278 |
| MT-R4-existence | 0.2192 | 0.1678 | 0.1321 |
| MT-R4-uniform | 0.2214 | 0.1704 | 0.1329 |
| MT-R4-shuffle | 0.2216 | 0.1710 | 0.1326 |

- native 头与 **native 初始化读出**（ST/MT 的 `R0-native`/`R2-native`）几乎相同，是本协议下最强的一组（100k 上 ST-R0-native 0.0874 = ST-native）。
- 随机初始化读出（`R0`/`R3`/`R4*`）明显更差；`MT-R1`（无 g、无边的弱消融）最差。
- 三个规模的主任务绝对精度随数据单调改善（native 0.164 → 0.114 → 0.087）。

---

## 4. 配对增量（跨 25 个 seed-pair 的均值；正值 = 第二个方法 MAE 更低）

| 对比（first − second） | 含义 | 20k | 50k | 100k |
|---|---|---:|---:|---:|
| MT-R0 − MT-R3 | 全局 g vs 局部 H（`pool_to_local_gain`） | −0.00926 | −0.00753 | −0.00551 |
| MT-R3 − MT-R4 | 预测局部结构在 H 之外（`predicted_local_gain`） | +0.00680 | +0.00545 | +0.00492 |
| MT-R0 − MT-R2 | 压缩 g 的可读性（`auxiliary_readout_gain`） | +0.00399 | +0.00193 | +0.00045 |
| ST-R0 − MT-R0 | 辅助监督对随机读出的影响（`auxiliary_supervision_gain`） | −0.00490 | −0.00679 | −0.00776 |
| ST-native − MT-native | 辅助监督对 native 的影响（`native_auxiliary_supervision_gain`） | +0.00060 | −0.00125 | −0.00051 |
| MT-native − MT-R0 | native vs 随机 g 读出（`random_readout_gain`） | −0.04808 | −0.04333 | −0.03892 |
| ST-native − ST-R0-native | ST native 初始化读出损失（`st_native_initialized_readout_gain`） | +0.00018 | +0.00059 | +0.00001 |
| MT-native − MT-R0-native | MT native 初始化读出损失（`mt_native_initialized_readout_gain`） | +0.00097 | +0.00045 | −0.00007 |
| MT-R0-native − MT-R2-native | native 初始化下加摘要（`native_initialized_auxiliary_readout_gain`） | +0.00012 | +0.00013 | −0.00000 |

**同容量图消融（仅 MT）**：

| 对比 | 隔离因素 | 20k | 50k | 100k |
|---|---|---:|---:|---:|
| MT-R3-graph − MT-R4-nocharge | 预测完整键级 vs 均匀图（`predicted_graph_gain_given_H`） | +0.00859 | +0.00992 | +0.00461 |
| MT-R4-nocharge − MT-R4 | 电荷增量（`charge_gain_given_graph`） | −0.00149 | +0.00028 | +0.00039 |
| MT-R4-uniform − MT-R4-existence | 连接性（`connectivity_gain`） | +0.00219 | +0.00265 | +0.00078 |
| MT-R4-existence − MT-R4 | 键级（`bond_order_gain`） | +0.00546 | +0.00732 | +0.00474 |
| MT-R4-shuffle − MT-R4 | 关系位置（`graph_assignment_gain`） | +0.00786 | +0.01059 | +0.00522 |

---

## 5. 结论

1. **全局 g 读出一致优于局部 H 读出**：`pool_to_local_gain`（R0 vs R3）三档均为负，说明在完整局部读出之外并未带来收益；差距随规模略有缩小（−0.0093 → −0.0055）。
2. **预测的局部/关系结构确有增量，但随规模衰减**：`R4 > R3`（+0.0068/+0.0055/+0.0049）；同容量图消融中「键级」（+0.0055/+0.0073/+0.0047）与「关系位置/shuffle」（+0.0079/+0.0106/+0.0052）稳定为正且最大；「连接性」弱正并随规模衰减；「电荷」基本中性。方向支持"把预测的局部关系组织进读出是有效的"，但收益在大数据下变小。
3. **辅助监督（多任务）在随机读出层面持续有害、在 native 层面基本无差异**：`ST-R0 − MT-R0` 三档为负且随规模增大（−0.0049 → −0.0078）；而 `ST-native − MT-native` 接近 0（±0.001）。说明历史 v1 中观察到的"辅助监督拖累主任务"主要体现在**随机初始化读出**的可优化性/样本效率，而非表示本身；在 LR 衰减 + 充分训练后，native 层的负迁移基本消失。
4. **native 头可被下游读出无损吸收**：`R0-native`/`R2-native` 与 native 差异 ≤0.001（100k 上 ≈0），并显著优于随机初始化读出。因此判断"辅助预测是否可用"不能只看随机读出，而应看 native 初始化与同容量消融。
5. **弱消融 R1 崩溃**（0.309/0.281/0.243）：去掉全局 g 与图结构后预测大幅变差，表明性能主要由 g 与预测图承载。

---

## 6. 辅助任务诊断（multi_task 上游头，y_test 均值）

| 指标 | 20k | 50k | 100k |
|---|---:|---:|---:|
| 逐原子 Mulliken 电荷 MAE (e) | 0.00914 | 0.00710 | 0.00600 |
| 键存在 accuracy | 0.99070 | 0.99405 | 0.99539 |
| 键存在 AUC | 0.99884 | 0.99943 | 0.99970 |
| 真实键上键级 accuracy | 0.99662 | 0.99736 | 0.99812 |
| 真实键对数 | 74,071 | 185,624 | 372,036 |

辅助预测质量随规模提升；键存在接近饱和（几何强相关），机制解释应优先看真实键上的键级与图消融。

---

## 7. 必须声明的局限

- **单节点 4×A10、手工全局队列调度**（`global_pool.py`），非批处理；机器为共享节点，CPU 争用使运行时间受外部负载影响。
- **读出来源单一冻结上游**：结果只反映"读出如何组织"，不代表表示能力上限；R1/R3/R4 与 R3 容量不严格匹配（R1 为弱消融）。
- **统计口径**：每档 5 个上游 seed × 5 个下游 seed；下游 seed 先在上游 seed 内平均，样本 SD/标准误只覆盖初始化变化，不覆盖分子抽样、划分或调参不确定性。
- **非 SOTA**：上游为 SchNet smoke 规模（ST 62,977 / MT 87,943 参数），仅作机制对照。
- **LR 衰减与 500 轮**是本轮相对历史的关键变化；与历史结果不可直接逐项比较，须换新 experiment 名。

---

## 8. 产物与环境

结果根目录：`/data/yuyang/SerialFlavour/results/qm9/`：
- `qm9_gap_charge_bond_refine_v2_20k/`（identity `ea5571e0a878c755…`）
- `qm9_gap_charge_bond_refine_v2_50k/`（identity `291f50aff777f52d…`）
- `qm9_gap_charge_bond_refine_v2/`（100k，identity `97cf00479fd8adc0…`）

每个实验含：`run_manifest.json`、`data/preparation_manifest.json`、`upstream/`、`cache/`、`refiners/`、
`evaluation/{metrics.csv,metrics.json,auxiliary_metrics.csv,*_predictions.csv}`、`analysis/{summary.md,summary.json,methods.csv,paired_deltas.csv,per_upstream_deltas.csv}`。

日志：`/data/yuyang/SerialFlavour/logs/qm9/<experiment>/`。环境：`gn2_study_cross`
（Python 3.11.16、torch 2.2.1+cu121、numpy 1.24.2、rdkit 2025.09.3）。

---

## 9. 复现命令

```bash
PY=/data/yuyang/miniconda3/envs/gn2_study_cross/bin/python
cd /home/yuyang/SerialFlavour
$PY -m pytest cross-domain/tests -q          # 28 项通过

# 单实验（顺序）：每个实验各 423 单元
$PY cross-domain/scripts/run_pool.py \
  --config cross-domain/config/qm9_gap_charge_bond_refine_v2_20k_cluster.json \
  --gpus 0 1 2 3 --retries 1 --python $PY

# 三档跨实验全局调度（本次实际使用，保持 GPU 不空闲）：
#   /data/yuyang/tmp/opencode/global_pool.py
#   依次 prepare 20k/50k/100k，按空闲 GPU 派发 (variant, seed)，
#   每档全部 seed 完成后自动 evaluate + analyze。
```

判读方式与对比含义见 `qm9_experiment_zh.md`。
