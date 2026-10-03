# QM9 HOMO−LUMO gap + 电荷/键级：full 规模运行结果（中文）

本文件记录 `qm9_gap_charge_bond_full` 配置在集群上的完整运行与最终评估结果。运行配置见
`cross-domain/config/qm9_test.json`（由 `qm9_gap_charge_bond_full.json` 复制，仅改
`data_root` 与 `output_root`）。数据与代码的 identity 为
`7eab8a5f0ee06082999409491609dcc26026d6127401bd7aa7613e9289ddefd5`。

---

## 1. 一句话结论

**实验没有支持「局部辅助监督 + 局部读出优于全局 embedding + 标量辅助预测」这一假设。**
在 5 个上游种子上：全局池化读出 `R0(g)` 一致优于局部读出 `R3(H)`；预测的局部/关系结构 `R4`
在 `R3` 之上没有带来任何增量；多任务（局部辅助监督）相对单任务反而略微**损害**了主任务。
上游自带头（native）始终是最强基线。

---

## 2. 运行概况

| 项目 | 值 |
|---|---|
| 状态 | **全部完成，10/10 seed 成功，无失败** |
| 环境 | conda `gn2_study_cross`（clone 自 `gn2_env` + rdkit） |
| Python / torch / numpy / rdkit | 3.11.16 / 2.2.1+cu121 / 1.24.2 / 2025.09.3 |
| GPU | 4× NVIDIA A10；实际使用 GPU0/1/2（GPU3 被他用占用，已避开） |
| `prepare` | 180.4 s |
| single_task 各 seed | 53.0–57.0 min |
| multi_task 各 seed | 156.3–167.4 min |
| `evaluate` / `analyze` | 260.3 s / 0.1 s |
| 结果目录 | `/data/yuyang/SerialFlavour/results/qm9_test/qm9/qm9_gap_charge_bond_full/` |
| 结果体积 | 2.6 GB（其中冻结缓存 2.3 GB） |
| 单元数 | 223（prepare 1 + upstream 10 + cache 10 + refine 200 + evaluate 1 + analyze 1） |
| 评估表行数 | `metrics.csv` 210 行（10 个方法） |

运行方式：`prepare` 后按 seed 并行（每个 seed 独占一卡），随后 `evaluate` → `analyze`。
后台以 `nohup` 启动，进程已脱离会话（PPID=1），窗口关闭不影响运行。

---

## 3. 数据与划分

| 项目 | 值 |
|---|---|
| 原始分子 | 133,885 |
| 官方排除（uncharacterized） | 3,054 |
| 合格分子 | 130,831 |
| 候选（candidate_multiplier=2，封顶全部合格） | 130,831 |
| 最终唯一分子子集 | 100,000 |
| 无效候选 | 8,739 |
| 去重丢弃 | 81 |
| 原子数范围 | 3 – 29 |
| 去重口径 | relaxed-geometry 的唯一 canonical isomeric 重原子 SMILES |

划分（**共享验证** `shared_validation=true`）：

| split | 分子数 | 作用 |
|---|---:|---|
| a_train | 56,000 | 上游训练 |
| b_train | 14,000 | 下游训练（仅主任务标签） |
| a_val = b_val（共享） | 10,000 | 上游/下游 checkpoint 选择 |
| y_test | 20,000 | 锁定评估 |

全对键级真值类别计数（`i<j`）：无键 13,966,417；单键 1,665,278；双键 70,997；三键 28,623；芳香 95,831。

---

## 4. 模型与读出

**上游 SchNet**（无预训练）：

| variant | 参数量 | 任务头 | 选择指标（best epoch） |
|---|---:|---|---:|
| single_task | 62,977 | gap | a_val MAE 0.09749（e93/100） |
| multi_task | 87,943 | gap + 局部电荷 + 全对键级 | a_val MAE 0.10375（e97/100） |

**下游读出**（容量未严格匹配）：

| recipe | 类型 | 输入 | 参数量（MT seed1） |
|---|---|---:|---:|
| R0 | 表格 MLP | 全局 g | 6,913 |
| R2 | 表格 MLP | g + 分子级汇总（电荷统计+键类占比） | 6,913 |
| R1 | 集合/GNN | 原子类型嵌入 + 预测电荷（无 g、无边） | 40,129 |
| R3 | 集合/GNN | 逐原子 H（无 g、无边） | 14,593 |
| R4 | 集合/GNN | H + 预测电荷 + 预测键概率（GNN 消息） | 39,489 |
| R4-shuffle | 集合/GNN | 同 R4，但键概率随机打乱（对照） | ≈39,489 |

single_task 上游只枚举 `R0/R3`；multi_task 枚举全部 recipes。

---

## 5. 主任务结果（y_test 20,000，MAE 单位 eV）

按方法聚合：先在每个上游 seed 内对 5 个下游 seed 求平均，再对 5 个上游 seed 求均值/样本 SD。

| 方法 | MAE 均值 | 样本 SD | 上游种子数 |
|---|---:|---:|---:|
| **ST-native** | **0.098634** | 0.001397 | 5 |
| ST-R0 | 0.119503 | 0.001338 | 5 |
| ST-R3 | 0.123116 | 0.003186 | 5 |
| **MT-native** | **0.103823** | 0.002319 | 5 |
| MT-R0 | 0.127056 | 0.001320 | 5 |
| MT-R2 | 0.126578 | 0.000466 | 5 |
| MT-R3 | 0.133507 | 0.003372 | 5 |
| MT-R4 | 0.134290 | 0.002935 | 5 |
| MT-R4-shuffle | 0.135384 | 0.003244 | 5 |
| MT-R1 | 0.340686 | 0.010030 | 5 |

> 参见 `analysis/methods.csv`、`analysis/summary.md`。

---

## 6. 关键对比解读（配对增量）

约定：`summary.md` 中**正值表示第一个方法 MAE 更低（更好）**。

| 对比（first − second） | 含义 | 均值 | 样本 SD | 结论 |
|---|---|---:|---:|---|
| MT-R0 − MT-R3 | 全局池化 vs 局部池化（`pool_to_local_gain_mae`） | **−0.006450** | 0.003238 | R3 比 R0 **差**，局部读出无优势 |
| MT-R3 − MT-R4 | 预测局部/关系结构的增量（`predicted_local_gain_mae`） | −0.000783 | 0.003113 | 在噪声内，**无增量** |
| MT-R0 − MT-R2 | 压缩 g 的可读性（`auxiliary_readout_gain_mae`） | +0.000478 | 0.001192 | 可忽略的微弱增益 |
| ST-R0 − MT-R0 | 辅助监督是否改变表示（`auxiliary_supervision_gain_mae`） | **−0.007553** | 0.002203 | 多任务辅助监督**损害**下游读出 |

补充（来自 `methods.csv`，同一读出下 ST vs MT）：

- `R0`：ST 0.1195 vs MT 0.1271 → 多任务更差（Δ≈0.0076）。
- `R3`：ST 0.1231 vs MT 0.1335 → 多任务更差（Δ≈0.0104）。
- native：ST 0.0986 vs MT 0.1038 → 多任务更差（Δ≈0.0052）。

**方向性总结（与假设相反）：**

1. **全局 g 优于局部 H**：`R0` 在 ST/MT 下都优于 `R3`。
2. **预测的局部结构没有价值**：`R4 ≈ R3`（差异在 SD 内）；`R4-shuffle` 也只比 `R4` 差 0.001，说明打乱键概率几乎无损。
3. **辅助监督有害**：同一读出下 ST 全面优于 MT；`ST-R0` 比 `MT-R0` 低 0.0076。
4. **上游 native 头最强**：任何下游读出都不如直接预测头。
5. **去掉全局 g 与边会崩溃**：`R1`（仅原子类型嵌入+电荷，无 g、无边）MAE 0.34，是其余方法的约 2.7 倍，说明性能主要由全局 g 与（预测）图结构承载。

---

## 7. 辅助任务诊断（multi_task 上游头，y_test）

对 5 个上游种子的均值：

| 指标 | 均值 | 说明 |
|---|---:|---|
| 逐原子 Mulliken 电荷 MAE | 0.6531 e | 电荷尺度 ~[-1,1]，误差偏大 |
| 键存在 accuracy | 0.9940 | 存在/不存在判定很准 |
| 真实键上键级 accuracy | 0.9991 | 但对不存在键的类别不平衡需谨慎 |
| 真实键对数 | 372,036 | — |

> 键存在与几何高度相关，机制解释应看真实键上的键级与 `R4−R3`；后者本轮无增量。

---

## 8. 必须声明的局限

1. **共享验证**：10k 验证集同时用于上游与下游选择，两者选择不独立；`y_test` 仍锁定。
2. **上游是 smoke 规模**（63k/88k 参数）+ 56k 训练分子，不对标 SchNet 报告数值（QM9 gap 强模型为 meV 级）。
3. **容量未严格匹配**：`R4`（≈39k）远大于 `R3`（≈15k），`R1/R4` 在 14k b_train 上有过拟合风险（已用 dropout 0.2 + weight_decay 5e-4 缓解）。
4. **单节点多卡、手工调度**，非批处理；无全局 `stage_state`。
5. 本实验为**读出机制对照**，不据此判断统计显著性；方向性结论在 5 个上游种子上一致，但样本量有限。
6. `identity` 已随 `data_root/output_root` 变化；结果目录为该 identity 的首次运行，未覆盖历史证据。

---

## 9. 产物清单

根目录：`/data/yuyang/SerialFlavour/results/qm9_test/qm9/qm9_gap_charge_bond_full/`

- `run_manifest.json`：环境与 identity。
- `data/preparation_manifest.json`：数据来源、许可、SHA256、划分、类别计数、无效候选。
- `upstream/<variant>/seed<k>/{best.pt,history.json,csv,training_manifest.json}`。
- `cache/<variant>/seed<k>/{b_train,b_val,y_test}.{npz,json}`（冻结特征）。
- `refiners/<variant>/seed<u>/<recipe>/seed<d>/...`。
- `evaluation/metrics.{json,csv}`、`auxiliary_metrics.csv`、`*_predictions.csv`。
- `analysis/summary.{json,md}`、`methods.csv`、`paired_deltas.csv`。

日志：`/home/yuyang/SerialFlavour/logs/qm9/qm9_gap_charge_bond_full/`（保持默认）。

---

## 10. 复现命令

```bash
cd /home/yuyang/SerialFlavour
PY=/data/yuyang/miniconda3/envs/gn2_study_cross/bin/python

# 1) 数据准备（已预置 raw 文件；download 仅做 MD5 校验）
$PY cross-domain/pipeline/run_unit.py --config cross-domain/config/qm9_test.json --unit prepare

# 2) 10 个 seed（每个 seed 独占一卡）；或直接用 pool：
$PY cross-domain/pipeline/run_pool.py \
  --config cross-domain/config/qm9_test.json --gpus 0 1 2 --retries 1 \
  --python $PY

# 3) 评估与聚合
$PY cross-domain/pipeline/run_unit.py --config cross-domain/config/qm9_test.json --unit evaluate
$PY cross-domain/pipeline/run_unit.py --config cross-domain/config/qm9_test.json --unit analyze
```

> 官方便捷入口：`PYTHON=$PY CONFIG=cross-domain/config/qm9_test.json GPU_POOL="0 1 2" \
> bash cross-domain/scripts/run_qm9_full.sh`（本机 GPU3 被他用占用，故仅用 0/1/2）。
