# QM9 HOMO−LUMO gap + 电荷/键级：full 规模运行结果（中文）

本文件记录 `qm9_gap_charge_bond_full` 配置在集群上的完整运行与最终评估结果。运行配置见
`cross-domain/config/qm9_test.json`（由 `qm9_gap_charge_bond_full.json` 复制，仅改
`data_root` 与 `output_root`）。历史运行的 config/code identity 为
`7eab8a5f0ee06082999409491609dcc26026d6127401bd7aa7613e9289ddefd5`。


**版本与证据边界（2026-10-03 补充）**：本文的数值属于修正前的历史 full 运行，
不是新协议的结果。本地已核对 `cross-domain/results/qm9_gap_charge_bond_full/analysis/`
中的 `methods.csv`、`paired_deltas.csv`、`summary.json`、`summary.md`；四项配对均值和
样本 SD 由全部 25 行配对结果复算一致。运行完成状态、环境、训练耗时和辅助指标来自
原运行记录；本次未连接集群核验原始 checkpoint、训练曲线或 210 份逐分子预测。
历史分析 JSON/CSV 保持原样，本文纠正其中的文字与解释。

当前代码已修正电荷单位、无序 pair 对称性及键级读出，并新增 native 初始化与同容量图消融。
新配置为 `cross-domain/config/qm9_gap_charge_bond_refine_v2.json`，实验名为
`qm9_gap_charge_bond_refine_v2`；尚未运行 100k 正式实验，不能把以下旧数值当作其性能。

---

## 1. 主要结论

**在当前 QM9 gap、等权电荷/键级辅助损失、固定划分和训练预算下，冻结后重新训练的读出
没有超过 native；多任务配置也没有改善主任务。** 十个方法的聚合 MAE 中 ST-native 最低。
MT-R0 相对 ST-R0 的劣化在 25 个配对运行中均出现；每个上游 seed 内平均后，MT-R3
在 5/5 个上游 seed 中差于 MT-R0。R4 和 R2 则没有稳定收益，不能写成“所有种子方向一致”。

这些结果不支持当前实现的 post-refinement 收益，但还不能证明局部辅助信息没有价值：
历史 R4 只使用键存在概率，R0/R3/R4 容量和读出结构不同，且原电荷 MAE 的单位处理有误。
负结果应限定为当前任务、辅助配置与读出训练条件，不能推广为辅助监督普遍有害。

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

## 4. 历史模型与读出的实际实现

上游是无预训练的 TinySchNet，`g = Σ_i h_i`。下列验证 MAE/epoch 按原运行记录保留，
不作为五个上游 seed 的聚合指标；具体 checkpoint 仍需原始 training manifest 核对。

| variant | 参数量 | 任务头 | 原记录的验证指标 |
|---|---:|---|---|
| single_task | 62,977 | gap | a_val MAE 0.09749（e93/100） |
| multi_task | 87,943 | gap + 局部电荷 + 全对键级 | a_val MAE 0.10375（e97/100） |

历史下游读出的输入和参数量如下；不能用当前 v2 实现重新定义这些旧结果。

| recipe | 类型 | 实际输入/操作 | 参数量（原记录 MT seed1） |
|---|---|---|---:|
| R0 | 表格 MLP | g + 10 个零槽 | 6,913 |
| R2 | 表格 MLP | g + 标准化电荷预测的五项统计 + argmax 键类占比 | 6,913 |
| R1 | GNN | 原子类型嵌入 + 电荷预测 + 预测连接权重；无 H/g，**有边** | 40,129 |
| R3 | 集合读出 | 逐原子 H，经节点 MLP 后求和；无显式 g/边 | 14,593 |
| R4 | GNN | H + 电荷预测；消息权重为 `1 − p(no_bond)` | 39,489 |
| R4-shuffle | GNN | 同 R4，但在整个 split 的 pair 行之间打乱概率 | 39,489 |

single_task 只运行 R0/R3；multi_task 运行全部六个 recipes。R0/R2 参数量相同，
但 R3/R4 未容量匹配。历史 R4 虽缓存五类概率，消息传递实际只使用其键存在总概率；
单/双/三/芳香键并未作为不同关系被读出。

原始辅助头在标准化电荷空间训练，历史缓存没有将其还原为 e；这不自动使 gap MAE 无效，
但不能将其物理电荷统计或电荷诊断按 e 解释。R1 是“原子类型 + 辅助预测 + 图”的诊断，
其较差表现不能归因于移除了边。

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

## 6. 配对统计与结论边界

### 6.1 统一增量方向

代码和历史 CSV 实际定义为 `Δ = MAE(first) − MAE(second)`：
**正值表示第二个方法 MAE 更低，负值表示第一个方法更好**。
历史 `summary.md` 的“正值表示第一个方法更好”写反；数值及原 CSV 不改变。

| first − second | 均值（eV） | 上游样本 SD | second 改善的上游 seed 数 | 支持的解释 |
|---|---:|---:|---:|---|
| MT-R0 − MT-R3 | −0.006450 | 0.003238 | 0/5 | 当前 R3 读出弱于 R0 |
| MT-R3 − MT-R4 | −0.000783 | 0.003113 | 1/5 | 未观察到稳定的预测辅助读出增益 |
| MT-R0 − MT-R2 | +0.000478 | 0.001192 | 3/5 | 平均微弱改善，种子间方向不一致 |
| ST-R0 − MT-R0 | −0.007553 | 0.002203 | 0/5 | 当前等权多任务配置下 R0 稳定劣化 |

逐上游 seed 内先平均五个下游 seed 后，增量为：

| upstream seed | R0−R3（meV） | R3−R4（meV） | R0−R2（meV） | ST-R0−MT-R0（meV） |
|---|---:|---:|---:|---:|
| 1 | −6.089 | −0.551 | −0.137 | −5.757 |
| 2 | −2.274 | −1.065 | +0.905 | −9.053 |
| 3 | −5.839 | −4.531 | −0.945 | −4.853 |
| 4 | −6.708 | −1.832 | +2.231 | −10.090 |
| 5 | −11.343 | +4.063 | +0.335 | −8.012 |

MT-R0−MT-R3 在原始 25 个配对中为负 22 次、正 3 次；R3−R4 为正 13 次、负 12 次。
因此，下游单次运行波动不能被五个上游均值的 SD 完全描述，也不能把 25 个配对当作
25 次独立上游训练。

### 6.2 不确定性的范围

将五个上游 seed 均值作为统计单位，粗略 t 区间为
`mean ± t(0.975, df=4) × SD / sqrt(5)`：上述四项依次为
`[−10.472, −2.429]`、`[−4.649, +3.082]`、`[−1.002, +1.958]`、
`[−10.288, −4.818]` meV。这是固定数据/划分下初始化均值的近似区间，依赖小样本 t 假设，
不覆盖测试分子抽样、split 或调参不确定性，也不是经过多重比较校正的机制检验。
区间跨零不证明等效；“差异小于 SD”同样不是零效应判据。

### 6.3 首先需要解释 native 到新读出的下降

ST-R0 比 ST-native 的 MAE 高 0.020869 eV（约 21.2%）；MT-R0 比 MT-native 高
0.023233 eV（约 22.4%）。MT-native 相对 ST-native 的均值劣化为 0.005189 eV（约 5.3%）。
这些是方法均值差；缺少逐 seed native 指标，不能声称每个 native seed 都有同样排序。

native 头随编码器在 56k A-train 上联合训练；新读出随机初始化，仅使用 14k B-train，
并采用 dropout 0.2、weight_decay 5e-4 和较短早停耐心。现有证据不足以区分读出
欠拟合、优化困难、过拟合或泛化差异，需要 native 初始化对照及训练曲线。

`g = Σ_i h_i`，所以 H 已包含计算 g 的信息。R3 的非线性节点变换后求和与原 sum pooling
不同；R0 优于 R3 是当前读出可用性的结果，不能推出 H 信息更少，也不能直接估计 pooling
损失了多少信息。辅助预测是冻结上游表示的确定性函数；性能变化不能直接解释为新增
Shannon 信息。

历史 R4-shuffle 的均值仅比 R4 差 0.001094 eV，但本地配对 CSV 没有这个 contrast，
尚不能复算其配对 SD/区间。旧 shuffle 还混合了不同分子的概率分布，故均值接近并不能证明
图关系完全无用。R1 的 MAE 0.340686 表明其诊断输入/读出不足以匹配 H/g 路线，
不能证明性能主要由预测图承载。

---

## 7. 历史辅助诊断与单位错误

以下数值按原记录保留，尚未用集群原始缓存独立复算：

| 指标 | 原记录均值 | 当前可用性/口径 |
|---|---:|---|
| 电荷误差 | 0.6531（原文误标为 e） | **不能作为物理单位电荷 MAE；待反标准化后重算** |
| 键存在 accuracy | 0.9940 | 由五类 argmax 后折叠为有键/无键；不是 p(bond)≥0.5 的二分类指标 |
| 真实键上键级 accuracy | 0.9991 | 条件于真值有键；需各键类混淆矩阵检查少数类 |
| 真实键对数 | 372,036 | 同一 Y-test 的 pair 数，不是五个 seed 相加 |

训练拟合 `z_q = (q − μ_A) / σ_A`。历史缓存保存 `hat_z_q`，诊断却计算
`mean(abs(hat_z_q − q))`。正确物理预测应为 `hat_q = μ_A + σ_A × hat_z_q`。
因此 0.6531 不能用来判断电荷头学习质量，也不能据此解释负迁移。

诊断可先从历史 checkpoint 的 `charge_mean/charge_std` 与旧预测缓存恢复，不必为重算
这个指标重训上游；但应在隔离的诊断目录保存，并绑定原始 checkpoint/cache 哈希。
更改标准化电荷特征、键级消息和模型后，完整 v2 主任务结果必须另行运行。

新代码在缓存前统一还原为 e，manifest 强制记录并校验单位；增加键存在 AUC、precision、
recall、F1、TP/FP/FN/TN，以及真实键上的五类混淆矩阵。旧 argmax 折叠指标以
`bond_multiclass_existence_accuracy` 单独保留；新 `bond_existence_accuracy` 使用
`1−p(no_bond)≥0.5`。辅助指标高不自动意味着 gap 读出收益。

---

## 8. 必须声明的局限

1. 历史 10k 验证集同时用于上游和下游选择；选择不独立。Y 未进入代码中的训练/选择，
   但本次后续实验设计已由历史 Y 结果启发，不能把重复使用同一 Y 的改进当作完全独立的确认性证据。
2. 只有一个固定采样与 random split（seed=2026），五个 seed 改变初始化。canonical SMILES
   去重不等于 scaffold 外推；不能声称跨骨架或跨构象泛化。
3. 8,739 个无效候选约占官方排除后候选的 6.68%；需原始 preparation manifest 检查失败原因及
   分布，最终结论限定于经过当前过滤与去重的子集。
4. 上游为约 63k/88k 参数、56k 训练分子的 TinySchNet 实现；不直接对标文献模型。
   原记录最佳 epoch 接近 100 的预算上限，也不能凭此证明已收敛。
5. R3/R4 容量不匹配；旧 R4 不消费完整键级；R1 实际有图；电荷诊断单位错误。
   正则化的存在本身不能证明过拟合已解决，也不能证明下降必然来自过拟合。
6. 只测了当前等权辅助配置，未隔离电荷与键级的上游监督贡献，不能将观察到的负迁移推广到所有权重。
7. 旧 smoke 曾出现 MT/R3 改善，但当时为单种子，并使用不同的 batching、dropout、早停和验证规则。
   与 full 的反转不能单独归因于数据规模，更不能作为 scaling law。
8. 本地只有四个分析汇总文件；运行状态、耗时及辅助指标仍需原始集群产物核验。
   单节点单元调度没有全局 stage_state，并不等同于缺少单元运行标记。
9. 当前代码修改会改变 identity。不得用修改后的代码覆盖原实验名或将新指标混入旧表。

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

## 10. 历史复现命令

以下命令只记录原运行。`qm9_test.json` 未随本地分析文件提供；复现原数值须恢复匹配
历史 identity 的源码与配置。当前修正代码应使用第 11 节的新实验名，不应直接续跑旧目录。

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


---

## 11. 修正后的 post-refinement 协议（尚无正式性能结果）

配置：`cross-domain/config/qm9_gap_charge_bond_refine_v2.json`。A-train 56k、A-val 5k、
B-train 14k、B-val 5k、Y-test 20k，总数仍为 100k，A-val/B-val 不共享。
固定总量下 A/B 边界变化，须核对新 split 哈希；这不是对原历史模型的无改动复评。

全部上游参数保持冻结，B 只使用主 gap 真值。新增 `R0-native/R2-native` 使用 native
激活与宽度，具有相同的 10 个附加槽和参数量；R0 槽为零，R2 为电荷/键概率摘要。
初始化将 B 特征标准化与 A→B 标签标准化吸收到首/末线性层中，R2 的新槽权重初始化为零。
训练前必须复现缓存中的 native 物理预测。B-val 在 epoch 0 和训练 checkpoint 中选择，
Y 不用于 fallback；不能保证选出的模型在 Y 一定改善。两个 native 初始化配方使用相同
无 dropout、lr=3e-4、weight_decay=1e-5 设置，保留随机初始化 R0/R2 作参照。

五类键概率中四个有键关系分别聚合；不再把它们全部压成一个存在权重。无序 pair 头使用
`[h_i+h_j, |h_i−h_j|, g]`，消除端点顺序捷径。R2 摘要使用平均 soft 概率，避免 argmax
丢失预测置信度。以下六个 H 图配方具有相同 65 维节点输入、typed GNN、层数和参数量
（默认宽度下 64,065）；无电荷臂以零槽补齐：

| recipe | 节点 | 边概率 | 隔离的因素 |
|---|---|---|---|
| R3-graph | H + 零电荷槽 | 全对均匀连接/均匀键类 | 同容量、无显式辅助的图架构对照；ST/MT 均可运行 |
| R4-nocharge | H + 零电荷槽 | 预测完整键级概率 | 与 R3-graph 比较预测关系；与 R4 比较电荷增量 |
| R4-uniform | H + 预测电荷 | 全对均匀连接/均匀键类 | 固定电荷的图结构基线 |
| R4-existence | H + 预测电荷 | 保留 p(no_bond)，四种键类均分有键概率 | 与 uniform 比较连接；与 R4 比较键级 |
| R4 | H + 预测电荷 | 预测完整键级概率 | 完整辅助读出 |
| R4-shuffle | H + 预测电荷 | 每个分子内部打乱 pair 概率行 | 保留每个分子的类别分布，检验关系位置 |

保留 H-only 集合读出 R3 与辅助/原子类型诊断 R1，但它们不属于上述同容量图归因比较。
新配置共 423 个单元：prepare 1、upstream 10、cache 10、refine 400、evaluate/analyze 各 1。
完整评估应为 410 行（10 个 native + 400 个 refiner）。分析拒绝缺项、重复与非有限指标，
输出 `per_upstream_deltas.csv`，按真实 seed 数记录 SD/标准误，不再硬编码“单种子”。

验证：21 项协议与离线微型闭环 CPU 测试通过，覆盖物理电荷还原、native 输出复现、
epoch 0 验证选择、冻结 checkpoint 不变、键级敏感性、pair 对称性、图容量与 shuffle
分布、完整配对网格。这些测试不代替真实 QM9 100k 训练，也不证明 v2 有性能增益。

```bash
# 在确认集群路径后，使用新的实验配置；当前未执行该正式运行。
python cross-domain/pipeline/run_pool.py \
  --config cross-domain/config/qm9_gap_charge_bond_refine_v2.json --gpus 0 1 2
```

后续优先核查 native 初始化读出与 matched graph 消融，再通过仅电荷、仅键级及损失权重
对照判断训练期负迁移来源。现有代码允许配置辅助损失权重，但本次没有执行权重扫描，
也没有预设某一种辅助任务必然有效。
