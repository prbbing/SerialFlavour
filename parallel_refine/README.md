# Parallel Refine：从多任务潜空间中重新组织可用信息

> 状态：基础训练、冻结特征缓存、DNN/BDT 和锁定 Y 评估代码已实现；尚未运行真实训练，因此没有可汇报的物理结果。

## 1. 技术摘要

此前的 extra-input 实验说明：Parallel 模型的共享表示和辅助任务输出中，确实保留了可被后续模型恢复的信息；但把这些信息作为额外输入单独读出，并没有达到“辅助任务直接进入联合 loss”时的效果。因此，**单独训练辅助任务以替代联合多任务训练**这条路线目前不成立。

`parallel_refine` 不再假设辅助任务会引入新信息，而是研究一个更窄、也更可检验的问题：

> 在冻结的 GN2-like Parallel 模型中，jet、track-origin、track-pair 三个任务已经形成了不同的信息读出。能否通过重新校准、汇总和组合这些读出，使下游 DNN 或 BDT 更容易利用共享潜空间中的已有信息，并在严格隔离的 Y 集上改善 jet flavour tagging？

这里的辅助任务有两个可能作用：

1. 在上游训练阶段约束共享 encoder，使其学习与重味物理有关的表示；
2. 在下游阶段把潜空间中的信息整理成更容易读出的结构，减轻最终 jet 分类器的表示学习压力。

第二点是本项目的新重点。它不等价于增加新的 detector/truth 输入，也不能预设一定会成功。若最终没有稳定的 Y 增益，潜空间可读性、任务竞争、下游样本效率和 pair-head 显存优化仍应形成可复现的负结果，而不是只保留零散尝试。

## 2. 当前模型中“embedding”和“task readout”的准确含义

当前 [`parallel_origin_vertex_jet.py`](../src/models/parallel_origin_vertex_jet.py) 并不存在三个完全独立的 task embedding。模型首先产生共享的逐 track 表示

\[
H\in\mathbb{R}^{K\times D},
\]

随后由三个 head 读出：

- `jet head`：attention pooling 后的 jet-level embedding \(z\in\mathbb{R}^{D}\)，以及 3 类 jet logits/probabilities；
- `origin head`：每条有效 track 的 8 类 origin logits/probabilities；
- `pair head`：每一对有效 track 的 same-vertex logit/probability 矩阵。

因此，本项目第一阶段所说的“组合三个任务的信息”，默认是组合 **共享表示、pooling 结果和三个 head 的预测**。若后续为每个 head 新增非平凡的中间层，并把其中间激活称为 task-specific embedding，必须单独版本化 feature schema，不能与当前共享 embedding 混称。

## 3. 核心假设与可证伪问题

### H1：共享潜空间中存在原始 jet head 没有充分利用的信息

若冻结 encoder 后，一个在独立 B 集上训练的下游模型能稳定超过原始 jet head，说明上游表示包含额外的、可用于 jet flavour 的信息，原始线性 jet head并非该表示上的最佳读出。

### H2：辅助预测能够改善信息组织，而不仅是增加维度

在相同下游模型、相同 B 数据、相同选择规则下，`embedding + auxiliary predictions` 应优于 `embedding only`。仅优于原始 jet head不足以证明辅助输出有用，因为增益也可能完全来自更强的下游分类器。

### H3：原始辅助 head 不是冻结表示上的最优读出

多任务 loss 的固定系数会改变共享 encoder 的优化方向，各任务梯度也可能竞争。冻结 encoder 后重新训练线性或小型非线性 origin/pair probe，可能得到更好的辅助预测；但只有这些改进进一步转化为 jet 指标增益时，才算对主任务有直接价值。

### H4：更好的组织形式可以降低下游读出所需的数据量或模型容量

即使最终性能相同，若结构化辅助输出能让更小的 DNN、BDT 或更少的 B 数据达到同等性能，也说明辅助任务改善了信息的可访问性和样本效率。

对应的零假设是：原始 jet head 已经近似饱和可用信息；或者 auxiliary readout 与 embedding 高度冗余，任何下游增益都只来自模型容量、额外 B 监督或选择偏差。

## 4. A / B / Y 数据边界

| 数据部分 | 唯一用途 | 允许发生的操作 | 禁止事项 |
| --- | --- | --- | --- |
| A-train / A-val | 训练并选择带双辅助任务的 Parallel 模型 | 优化多任务 loss；按预先指定的 A-val 规则选择 checkpoint | 不训练下游 DNN/BDT；不根据 B/Y 结果反选 checkpoint |
| B-fit / B-select | 训练、校准和选择下游 readout | 拟合 feature normalizer、auxiliary probe、DNN/BDT；选择输入 recipe 和超参数 | 不更新上游 encoder；不查看 Y 来调参 |
| Y-test | 最终锁定评估 | 对冻结的完整 pipeline 做一次统一评估 | 不训练、不早停、不选择 feature、checkpoint 或阈值 |

必须满足：

- A、B、Y 在 event 层面互斥，而不仅是 jet 行号互斥；
- 每次运行保存数据源、split manifest/hash、实际 event/jet 数、各 flavour 分母、随机种子和 cache identity；
- 默认上游规模为 A-train 的 1M jets，但必须记录实际保留的 jets/events，不能把请求值当作最终分母；
- Y 的任何结果都不能回流到模型、特征、working point 或 checkpoint 的选择中；
- Y 上的预测数组与 event id 应保存，以支持配对差值和 event-level bootstrap。

### B 上重训辅助读出时的额外约束

如果 auxiliary probe 在 B 上用 origin/pair truth 训练，随后它的预测又作为 jet DNN/BDT 的输入，不能直接把 probe 对自身训练样本的 in-sample 预测交给下游模型。首选以下两种方案之一：

1. 把 B 明确分成 `B-probe-fit`、`B-refiner-fit`、`B-select`；
2. 在 B 上做 K-fold cross-fitting，生成 out-of-fold auxiliary predictions，再训练 jet refiner。

最终部署到 Y 时，probe 可在完整允许的 B-fit 数据上重训一次，然后冻结。该协议避免下游训练特征与 Y 推理特征出现由过拟合造成的分布差异。

## 5. 第一阶段必须完成的对照矩阵

### 5.1 固定上游与读出基线

| ID | 上游模型 | 下游读出 | 回答的问题 |
| --- | --- | --- | --- |
| `P-direct` | 双辅助任务 Parallel 模型 | 原始 jet head | 当前联合多任务模型的原始结果 |
| `P-embed` | 同一个 Parallel checkpoint | 冻结 embedding + 默认 DNN | 更强读出本身能带来多少增益 |

本轮不训练 Jet-only 或将辅助 loss 系数置零的上游模型。`P-direct` 与所有下游 recipe 必须来自同一个 Parallel checkpoint；因此本轮结论只回答“如何进一步利用已经由双辅助任务训练出的表示”，不回答“辅助监督本身相对无辅助训练是否改善表示”。

### 5.2 信息组合与 DNN / BDT 公平比较

| Feature recipe | 固定长度内容 | DNN | XGBoost | 主要比较 |
| --- | --- | :---: | :---: | --- |
| `F1_embed` | attention-pooled track embedding（默认 32 维） | ✓ | ✓ | embedding-only 基线 |
| `F2_jet_aux` | 3 类 jet probability + 8 维 pooled origin + 35 维 pooled pair relation | ✓ | ✓ | 辅助输出能否在原始 jet prediction 上继续修正 |
| `F3_embed_aux` | 32 维 pooled embedding + 43 维 structured auxiliary，不含 jet probability | ✓ | ✓ | 辅助输出对 embedding 的增量价值 |
| `F4_all` | jet probability + pooled embedding + structured auxiliary | ✓ | ✓ | 全部部署时可得读出的 stacking 上界 |

原始 Parallel 结果直接使用冻结 checkpoint 的 jet head probability 评估，不再为其额外训练 DNN/XGBoost，因此不设置 `F0`。缓存中的 jet-head 部分也只保留 3 类 probability，不保存 logits、margin、entropy 或 discriminant 等确定性变换。

DNN 与 BDT 的主比较必须使用同一份固定长度 feature table、相同的 B-fit/B-select 划分和相同的 Y 样本。不能用 raw sequence DNN 对比 structured-pooled BDT 后直接把差异解释为模型类别差异。

默认 DNN 为：

```text
input -> 128 -> 64 -> 32 -> 3-class output
```

激活、归一化、dropout、优化器、早停指标和参数量将在实现配置中显式记录。BDT 的树数、深度、学习率、subsample、正则化和 early stopping 轮数同样必须进入 effective config；当前阶段不预设最优 BDT 参数。

## 6. 不依赖 truth 的信息组织候选

固定长度 recipe 不再同时堆叠大量 sum/mean/std/max/quantile/entropy。第一版先在 track 层组织信息，再使用冻结 Parallel 已学到的 attention 做一次统一池化。

### 6.1 每条 track 的结构化向量

对每条有效 track $i$ 构造：

\[
u_i = [h_i,\; o_i,\; m_i,\; \bar p_i,\; p_i^{\max},\; p_i^{\sum}],
\]

其中 $h_i$ 是 track embedding，$o_i$ 是 8 类 origin probability。对有效非对角 pair，$m_i$ 是与 $h_i$ 同维的 pair-weighted neighbour embedding：

\[
m_i = \frac{\sum_{j\ne i}p_{ij}h_j}{\sum_{j\ne i}p_{ij}+\epsilon}.
\]

另外保留该 track 对其他 track 的 match probability mean、max、sum。weighted embedding 使用归一化加权平均，使表示尺度不随 track 数量增长；单独的 sum 保留连接强度和 multiplicity 信息。只有一条有效 track 或权重和为零时，所有 pair relation 特征确定为零。

### 6.2 冻结 attention pooling

使用 A 上训练好的 Parallel pooling weight $a_i$ 得到：

\[
g = \sum_i a_i u_i.
\]

默认 `d_model=32` 时，pool 后由 32 维自身 embedding、8 维 origin probability、32 维 pair-weighted embedding 和 3 个 pair probability statistics 组成，共 75 维。`aux` 部分是后 43 维；加入 3 类 jet probability 后 `F4_all` 总计 78 维。由于使用相同的冻结 attention，数学上“先拼接再池化”等价于分别池化后拼接，不需要将逐 track 张量写入磁盘。

所有 DNN 归一化参数只能在 B-fit 上拟合。padding、无有效 pair、只有单条有效 track 等边界情况必须有确定值和显式 mask，不能让 placeholder 数值被误解为真实预测。

## 7. 辅助任务读出能否进一步优化

在上游 encoder 固定后，按以下顺序研究 auxiliary head：

1. **原始 head**：直接使用 A 上联合训练得到的 origin/pair logits；
2. **线性 probe**：在冻结的逐 track embedding 上重新拟合 origin 分类，在冻结的 pair 表示上重新拟合 pair 分类；
3. **小型非线性 probe**：仅在线性 probe 已显示信息可读但欠拟合时加入；
4. **结构化聚合器**：对 `[track embedding, origin probability]` 使用 DeepSets/attention pooling，或对 pair matrix 使用轻量图聚合；
5. **联合下游优化**：只更新 probe 和 jet refiner，不更新 A 上训练的 encoder。

需要同时报告两层结果：

- probe 自身的 origin Macro-F1、逐类效率和 pair AUC/交叉熵；
- probe 输出加入 jet refiner 后的 Y jet 指标变化。

第一层改善不自动等价于第二层改善。若 auxiliary metric 上升但 jet tagging 不变，应解释为“辅助标签更可读，但对主任务没有新增可用信息”，而不能称为直接性能增益。

## 8. 评估指标与判定规则

### 8.1 Jet flavour 主指标

- multiclass cross-entropy：首要优化/选择指标，能看到概率质量变化；
- Accuracy 与 per-class efficiency；
- one-vs-rest AUC；
- 固定 signal-efficiency working point 下的 `b/c`、`b/light`、`c/b`、`c/light` rejection；
- 必要时报告 calibration error，但不替代 tagging 指标。

不同指标必须分开解释。Accuracy 提升不能代替 rejection 结论，零 background pass 对应的是有限样本下界，不能作为精确的无穷大倍率参与排序。

### 8.2 原始 Parallel 辅助头诊断

- track origin：在有效 truth track 上报告 confusion matrix（计数与按 truth 行归一化）、Accuracy、cross-entropy、Macro-F1 和逐类 precision/recall/F1；
- track pair：把 `match` 作为正类、`other` 作为负类，报告两类 score 分布、BCE、AUC、分位数和阈值通过率，并按 jet flavour 拆分直方图；
- pair 分母必须与训练的 `pair_vertex_loss` 保持一致，并显式记录是否包含对角 self-pair；若使用固定-bin 流式统计，AUC/分位数必须标记为近似值。

这些指标用于判断原始辅助头是否学到了相应结构，不代替 jet flavour 的主指标，也不能据此在 Y 上选择模型。

### 8.3 统计与复现

- pilot 可用单 seed 排除明显无效方案；进入结论的配置至少使用 3 个预先固定的训练 seed；
- 比较使用相同 A/B/Y split 的 paired delta；
- seed 间标准差描述训练波动，event-level bootstrap 描述固定模型在有限 Y 样本上的评估不确定性，两者不能混称；
- 所有方案使用同一 checkpoint 规则。主分析默认比较 A-val jet loss 选择的 `best_jet.pt`；`best_total.pt` 只能作为预先声明的敏感性分析，不能根据 Y 择优；
- 结果表必须同时给出绝对指标、相对基线差值、实际分母和 seed 覆盖。

### 8.4 何时认为路线“成功”

只有在以下条件同时满足时，才称为直接增益：

1. recipe、超参数和 checkpoint 全部只由 A/B 决定；
2. 相比 `P-direct` 和 `P-embed`，锁定 Y 上的 cross-entropy 或预注册的物理主指标出现一致的 paired improvement；
3. 增益不是由个别 seed、类别比例、无穷 rejection 或 BDT/DNN 不公平输入造成；
4. deployable 方案不使用 truth-derived oracle 输入。

如果只证明非线性 probe 优于线性 probe，应称为“潜空间信息可读性结果”；如果只在 truth/oracle feature 上提高，应称为“诊断上界”，不能报告为可部署性能。

## 9. Scaling law 应放在哪里

本项目是两阶段系统，上游表示和下游读出具有不同的数据与容量瓶颈。因此不应只画一条“总参数量—性能”曲线，而应将 Y loss 写成条件关系：

\[
L_Y = f(N_{P}, D_A, E_A, N_R, D_B, \mathcal{F}),
\]

其中：

- \(N_P\)：Parallel/GN2-like 上游参数量；
- \(D_A\)：A 中不重复的训练 events/jets 数；
- \(E_A\)：A 的重复曝光量（epochs、optimizer steps 或 seen jets）；
- \(N_R\)：下游 DNN 容量，或 BDT 的等效复杂度；
- \(D_B\)：B-fit 中不重复的训练样本数；
- \(\mathcal{F}\)：信息组织 recipe。

### 推荐顺序

1. **机制筛选**：固定默认 55k Parallel、A=1M、默认 DNN，在 B-select 上筛选 `F1–F4`、DNN/XGBoost 和少量 probe 方案；
2. **上游 scaling**：固定胜出的下游 recipe/capacity，扫描 \(N_P\) 与 \(D_A\)；
3. **重复性 scaling**：在固定 \(D_A\) 下改变训练 steps/epochs，区分“更多独立数据”与“重复看相同数据”；
4. **下游 scaling**：对选定上游 checkpoint 扫描 \(N_R\) 与 \(D_B\)，观察辅助组织是否降低读出容量或数据需求；
5. **交互验证**：只对前几步显示出的关键交互补小型二维网格，避免一开始做完整笛卡尔积。

至少有 4 个近似对数间隔的数据点、多个 seed 且趋势稳定时，才拟合 power-law exponent 或 irreducible-loss 项；点数不足时只称为 learning curve/capacity scan。所有模型还应报告训练 FLOPs 或 seen jets、峰值显存和 wall time，避免把更大训练预算误写成纯参数 scaling。

建议的首轮范围将在代码实现前固定到配置中；下列仅是候选，不是已执行实验：

| 轴 | 候选点 |
| --- | --- |
| A unique data | 100k、300k、1M；若资源允许再增加一点 |
| Parallel size | 约 0.5×、1×、2×、4× 参数；保持 GN2-like encoder 形式 |
| A exposure | 固定 epochs 与固定 seen-jets/steps 两套对照 |
| B data | 至少 3–4 个子样本点直到完整 B-fit |
| DNN capacity | 小于默认、默认 `128-64-32`、大于默认的对数级容量 |

## 10. Parallel 训练的显存工程问题

当前 pair head 会显式构造两个 `(B,K,K,D)` view 并 reshape 后送入 `Bilinear`；dense pair target 也按 `(N,K,K)` 保存。`K=40` 时，pair 路径通常是显存和缓存的主要压力之一。

工程优化按“先保持数学等价，再考虑改变目标”分层：

### 数学等价优化

- 将 bilinear 计算改写为投影加 batched matrix multiplication/einsum，避免物化 `(B,K,K,D)` 中间量；
- pair 维度分块计算 loss，减少峰值 activation；
- 从 truth vertex index 在 batch 内惰性构造 pair target，避免把完整 dense target 常驻缓存；
- 评估 AMP/bfloat16、gradient checkpointing 和更合适的 effective batch accumulation。

### 可能改变训练目标的优化

- 只训练上三角有效 pairs；
- hard-negative 或随机 pair sampling；
- 低秩/对称 pair score；
- 稀疏图或候选 vertex 分组。

第二类方案必须单独命名，不能与原始 full-pair loss 当作纯工程等价版本。每项优化都应验证 logits/loss/梯度的容差、有效 pair 分母、峰值 CUDA memory、吞吐量和最终 A-val/Y 指标。

## 11. 潜空间研究支线

若主 stacking 路线增益有限，可用同一套冻结 checkpoint 和 split 研究：

- layer-wise linear/nonlinear probes：区分“信息存在”与“信息容易读出”；
- probe learning curve：比较不同任务信息的样本效率；
- CKA/SVCCA 等 representation similarity：比较不同 Parallel seed、层或模型规模；
- task-gradient cosine、范数和 Pareto front：检查 origin/pair/jet 的竞争与协同；
- feature permutation/ablation：确认增益来自 origin、pair-weighted embedding 还是 pair probability statistics；
- calibration 与 conditional performance：检查 origin/pair 读出是否只在特定 flavour、track multiplicity 或拓扑上有用。

互信息估计对 estimator 和维度非常敏感，除非有明确的估计器验证与负对照，否则不把单个 mutual-information 数值作为主要证据。

## 12. 实施阶段与停止规则

### Phase 0：基础设施

- 固化 event-disjoint A/B/Y split 和 manifests；
- 为冻结 checkpoint 生成带 schema/version 的 B/Y feature cache；
- 实现 DNN/BDT 共用的固定长度 feature table；
- 实现预测、指标和 resource-usage artifacts。

### Phase 1：最小机制矩阵

- 完成 `P-direct`、`P-embed`，不增加无辅助任务的上游模型；
- 用相同 `F1–F4` 对比默认 DNN 与 XGBoost；
- 先单 seed pilot，再对 B-select 胜出项补多 seed；
- 只有锁定方案进入 Y。

### Phase 2：优化辅助读出

- 比较原始 head、线性 probe、小型非线性 probe；
- 使用 B split 或 OOF 协议生成下游训练特征；
- 同时报告 auxiliary metric 和 jet metric。

### Phase 3：信息组织模型

- 只在冻结 attention 的 structured pooling 已显示增量价值后尝试重新训练 attention 或更深的图消息传递；
- 与参数量相近的 embedding-only refiner 对照；
- 检查 permutation、mask 和 track-count 稳健性。

### Phase 4：Scaling

- 只对 Phase 1–3 中胜出的少量方案扫描上游数据/容量和下游数据/容量；
- 先做 learning curves，再决定是否有资格拟合 scaling law。

### 停止或转向条件

若 `F3/F4` 在 B-select 和多 seed 上均不能稳定优于 `P-embed`，则停止扩大 stacking 超参数搜索；这意味着辅助输出没有证明超出 embedding 的增量价值。后续工作转向潜空间诊断和 pair-memory 工程，而不是继续用 Y 反复筛选。

## 13. 预期目录与 artifacts

```text
parallel_refine/
├── README.md                    # 本研究契约
├── configs/                     # 上游、feature recipe、DNN/BDT、scaling 配置
├── src/                         # 冻结特征抽取、组织、训练与评估代码
├── tests/                       # split、mask、schema、等价性与泄漏回归测试
└── results/                     # manifests、metrics、predictions、plots/reports
```

每个可比较运行至少保存：

- `effective_config.json`；
- `split_manifest.json` 与 checkpoint identity/hash；
- `feature_manifest.json`（字段、shape、dtype、normalizer、mask、schema version）；
- `metrics.json`；
- `test_predictions.npz`（含 event id、truth 和 class probabilities）；
- `resource_usage.json`（参数量、seen jets/steps、峰值显存、训练时间）；
- 训练历史和 B-select 的模型选择记录。

当前实现已经提供从 split manifest 到 Y artifacts 的入口，见下一节。真实运行仍应先用缩小的数据配置完成端到端 smoke test，再提交 1M 默认训练。

代码依赖边界已经收敛：`parallel_refine` 不再从 `local/experiments` 导入任何实现。它只从正式 `src` 复用项目级配置默认值与随机种子、HDF5 块读取工具、Parallel 模型工厂、origin class weight 和 pair loss；A/B/Y 五路事件划分、实验专用 processed cache、训练循环和 checkpoint 适配器位于 `parallel_refine/src`。这样本实验不依赖另一份 local experiment snapshot，也不需要修改正式 `src` 的通用 train/val/test 数据契约。

## 14. 当前默认与待固定事项

已确定的默认：

- 上游保持 GN2-like Parallel 架构；
- 基准上游约 55k 参数、A-train 约 1M jets；
- 默认下游 DNN 为 `128-64-32-output`；
- A 训练上游，B 训练/选择下游，Y 只做锁定评估；
- 首轮使用带双辅助任务的 Parallel checkpoint，比较原始 jet head、embedding-only 读出、DNN/XGBoost 和辅助组合读出。

正式规模训练前仍需确认或预注册：

- A/B/Y 的实际 event 列表、B-fit/B-select 比例和 Y 锁定策略；
- 第一轮 feature schema 和 BDT 搜索预算；
- 预注册的主 working points、成功阈值和 seed 列表；
- scaling 各轴的最终点位与总 compute budget；
- pair-memory 优化是先做数学等价改写，还是与第一轮物理实验并行推进。

## 15. 当前实现与运行方式

### 15.1 组件配置与轻量 experiment config

配置分为三个可复用组件和一个轻量实验入口：

```text
configs/
├── data/          # 数据源、A/B/Y 划分和预处理
├── parallel/      # Parallel 架构、训练参数和模型 seed
├── refiner/       # frozen cache、DNN 和 XGBoost
└── experiments/   # 运行入口：实验身份、输出路径和组件路径
```

默认运行配置是 [`configs/experiments/default.json`](configs/experiments/default.json)。experiment 文件不再复制 data、Parallel、cache、DNN 和 XGBoost 的具体字段，只保留实验名称、结果根目录及三个组件路径：

```json
{
  "config_kind": "experiment",
  "experiment": {
    "name": "parallel_refine_default",
    "output_root": "/data/yuyang/SerialFlavour/results/parallel_refine"
  },
  "components": {
    "data": "../data/data_1m_b500k.json",
    "parallel": "../parallel/gn2_4layers.json",
    "refiner": "../refiner/dnn_xgboost_default.json"
  }
}
```

组件相对路径以 experiment 文件所在目录为基准。所有训练、缓存和评估命令仍然只允许传入 experiment config；loader 在内存中解析三个组件并执行完整 schema 校验。它同时计算 experiment 文件 SHA256、每个组件 SHA256 和解析后配置 SHA256，因此任何组件内容变化都会形成新的实验身份。实际输出目录固定为 `<experiment.output_root>/<experiment.name>/`，也不需要单独的 Evaluation config。

组件文件不能直接传给 `--config`，runtime config 也禁止使用 `base_config`。创建新实验入口时可使用：

```bash
python -m parallel_refine.tools.compose_experiment \
  --experiment-name parallel_refine_default \
  --output-root /data/yuyang/SerialFlavour/results/parallel_refine \
  --data-config parallel_refine/configs/data/data_1m_b500k.json \
  --parallel-config parallel_refine/configs/parallel/gn2_4layers.json \
  --refiner-config parallel_refine/configs/refiner/dnn_xgboost_default.json \
  --output parallel_refine/configs/experiments/default.json \
  --force
```

composer 只写入相对组件路径，不再展开或复制组件内容。`--force` 只在明确更新同名 experiment 入口时使用；创建新名称时不需要。少量确实只属于单个实验的差异可以放在可选 `overrides` 中，但不应把整个组件复制进去。生成后，实验中的所有命令只使用 `configs/experiments/default.json`。A/B/Y 大小在 data 组件中定义：

```json
"sizes": {
  "a_train": 1000000,
  "a_val": 100000,
  "b_train": 500000,
  "b_val": 100000,
  "y_test": 200000
}
```

Parallel 组件中的 `parallel.seeds` 只定义模型 seed 和相对输出名，不描述机器或 GPU：

```json
"seeds": [
  {"seed": 1, "output_name": "parallel_seed1"},
  {"seed": 2, "output_name": "parallel_seed2"},
  {"seed": 3, "output_name": "parallel_seed3"},
  {"seed": 4, "output_name": "parallel_seed4"},
  {"seed": 5, "output_name": "parallel_seed5"}
]
```

默认使用五个模型 seed（1–5）。`parallel.training.gpu_ids` 和 `refiners.dnn.gpu_ids` 固定为 `[-1]`；具体物理 GPU 由运行脚本开头的 `SEEDS`、`GPU_IDS` 和 `CUDA_VISIBLE_DEVICES` 指定。这样同一 experiment config 可以在不同机器上运行而不改变实验身份。

当前协议固定 `Parallel seed = DNN seed = BDT random_state`。如果以后需要独立扫描下游 seed，应显式升级配置 schema，而不是静默形成笛卡尔积。

### 15.2 生成 A/B/Y split

```bash
python -m parallel_refine.training.prepare_data \
  --config parallel_refine/configs/experiments/default.json
```

如需同时构建 processed caches：

```bash
python -m parallel_refine.training.prepare_data \
  --config parallel_refine/configs/experiments/default.json \
  --build-processed-caches
```

可以重复使用 `--processed-split` 只构建指定 split，例如在多 seed 训练前先串行构建公共 A/B cache，避免多个训练进程竞争写入：

```bash
python -m parallel_refine.training.prepare_data \
  --config parallel_refine/configs/experiments/default.json \
  --build-processed-caches \
  --processed-split a_train --processed-split a_val \
  --processed-split b_train --processed-split b_val
```

split 由 `eventNumber` 划分，manifest 记录实际 jet/event 数、类别分母、索引 hash、event hash 和所有 split 间 overlap。

### 15.3 在一个配置中训练多个 Parallel seed

```bash
python -m parallel_refine.training.train_parallel \
  --config parallel_refine/configs/experiments/default.json \
  --skip-complete
```

也可以只运行一个或多个指定 seed：

```bash
python -m parallel_refine.training.train_parallel \
  --config parallel_refine/configs/experiments/default.json \
  --seed 1 --seed 3
```

每个 seed 写入 `<experiment.output_root>/<experiment.name>/parallel/<output_name>/`。脚本拒绝覆盖非空的部分运行目录，`--skip-complete` 只跳过已经存在配置 checkpoint 的运行。

### 15.4 生成供 DNN/BDT 共用的 B/Y 缓存

```bash
python -m parallel_refine.training.generate_cache \
  --config parallel_refine/configs/experiments/default.json
```

默认生成 `b_train`、`b_val` 和 `y_test`。每个缓存绑定：

- Parallel seed 和配置中的 `output_name`；
- checkpoint SHA256；
- split-index SHA256 与 source-index SHA256；
- `top_k`、track fields、dtype 和 feature schema version。

缓存只从冻结 Parallel 的 reco 输入、共享 embedding 和三个 head 的预测构造，不读取 truth feature。它一次保存 `F4_all` 的完整 78 维 structured-pooled table，DNN/XGBoost 通过列选择得到 `F1–F4`，因此同一 recipe 的 DNN/XGBoost 输入逐元素一致。

pair head 在缓存推理中使用与 `nn.Bilinear` 等价的 `(H W)Hᵀ`，避免构造两个 `(B,K,K,D)` 中间量。对每条 track 计算归一化 pair-weighted neighbour embedding 和 mean/max/sum match probability，再用冻结 attention 池化；完整 pair matrix 和逐 track structured tensor 都不写入磁盘。

### 15.5 训练相同 seed 的 DNN 和 BDT

训练配置中启用的全部 recipe：

```bash
python -m parallel_refine.training.train_dnn \
  --config parallel_refine/configs/experiments/default.json \
  --skip-complete

python -m parallel_refine.training.train_bdt \
  --config parallel_refine/configs/experiments/default.json \
  --skip-complete
```

只训练指定 seed/recipe：

```bash
python -m parallel_refine.training.train_dnn \
  --config parallel_refine/configs/experiments/default.json \
  --seed 2 --recipe F3_embed_aux

python -m parallel_refine.training.train_bdt \
  --config parallel_refine/configs/experiments/default.json \
  --seed 2 --recipe F3_embed_aux
```

DNN 为带 B-train-only standardization 的 `128-64-32-output` MLP，按 B-val cross-entropy 保存 `best_dnn.pt`。BDT 使用 XGBoost `XGBClassifier`，固定 `multi:softprob`、`mlogloss` 和 histogram tree method，并用 B-val early stopping 选择最佳 boosting iteration。Y 不参与训练或选择。

公共缓存和指标代码不依赖 XGBoost；只有运行 BDT 训练或评估 XGBoost checkpoint 时才要求当前 SerialFlavour 环境安装 `xgboost>=2.0`。训练仍使用 `XGBClassifier` 的 sklearn 接口与 early stopping，但保存和评估使用其底层原生 `Booster` 的 `best_bdt.json`；这避开部分 XGBoost/sklearn 组合在 wrapper `save_model` 上缺少 `_estimator_type` 的兼容性错误。评估通过 `Booster.predict(DMatrix)` 执行，避免 CPU cache 与 CUDA booster 的 inplace-predict 设备回退警告。评估时显式使用 manifest 记录的最佳 boosting iteration。默认配置使用 GPU `device="cuda"`；CPU 运行时将其改为 `"cpu"`。

### 15.6 锁定后在 Y 上评估

```bash
python -m parallel_refine.training.evaluate \
  --config parallel_refine/configs/experiments/default.json \
  --model direct_dnn
```

`direct_dnn` 是默认模式，同时评估 direct Parallel 和所有选中的 DNN recipe。可以用 `--seed`、`--recipe` 和 `--model direct|dnn|direct_dnn` 缩小范围。`bdt|all` 只保留用于复现早期 BDT 研究，不属于默认流程。评估会保存 `metrics.json`、`test_predictions.npz` 和 `evaluation_manifest.json`。零 background pass 的 rejection 写成 `null` 加显式标志，不作为精确无穷值排序。

当 `--model direct|direct_dnn|all` 时，还会对原始 Parallel checkpoint 在同一 Y split 上做一次 live inference，评估两个辅助头。这样做是因为冻结的 jet-level feature cache 只保存 truth-free 汇总，不能从中精确恢复逐 track origin confusion matrix 或完整 pair score 分布。运行前会核对 live loader 与冻结缓存的 `source_index` 完全一致。

- track origin：只统计 `origin >= 0` 的真实 track，写出 accuracy、cross-entropy、Macro-F1、逐类 precision/recall/F1、原始计数矩阵和按 truth 行归一化的 confusion matrix；
- track pair：沿用 `src.losses.pair_vertex_loss` 与 `src.plotting.plot_pair_vertexing` 的分母，统计所有有效有序 pair，包含对角 self-pair；`match` 为 `truth_pair=1`，`other` 为 `truth_pair=0`；
- pair score 不整表落盘，而是以 100 个固定 bins 流式累计总体和各 jet flavour 的直方图，避免额外保存巨大的 `(N,K,K)` score 数组。pair AUC 和分位数因此明确标为 histogram approximation，BCE、均值、方差和计数仍按逐 pair 精确累计。

辅助评估文件位于 direct Parallel 的 `parallel/auxiliary_tasks/`：

- `origin_confusion_matrix.png` 与 `origin_confusion_matrix.csv`；
- `pair_score_comparison.png`：总体 match/other、按 jet flavour 的 match/other，以及 ROC 三联图；
- `pair_score_histogram.csv`：总体和各 jet flavour 的固定-bin 计数；
- `auxiliary_metrics.json`：完整定义、分母、指标、config/checkpoint/cache 身份。

### 15.7 多 GPU 完整运行脚本

[`run_experiments.sh`](run_experiments.sh) 使用脚本开头的 seed/GPU 映射，以简单的 `for`、后台 `&` 和 `wait` 按 seed 并行执行：

```bash
bash parallel_refine/run_experiments.sh
```

如果希望直接逐行查看或手工删改每一条命令，可使用不含变量、循环或配置区块的朴素版本：

```bash
bash parallel_refine/run_experiments_plain.sh
```

[`run_experiments_plain.sh`](run_experiments_plain.sh) 固定使用默认 experiment config、seed 1–5 和 GPU 映射 `0/1/1/2/2`。DNN 按 recipe 分为四批，每批 5 个后台任务，因此不超过 `max_job=12`；日志固定写入 `parallel_refine/logs/plain/`，下一次运行会覆盖同名日志。根据首轮结果，plain 默认流程不再训练或评估 BDT。

两个 scaling 对照使用各自完全展开的 plain 脚本，并写入相互隔离的日志目录：

- [`run_experiments_plain_a1m_6layers.sh`](run_experiments_plain_a1m_6layers.sh)：固定使用 `a1m_6layers.json`，日志写入 `parallel_refine/logs/plain_a1m_6layers/`；
- [`run_experiments_plain_a500k_4layers.sh`](run_experiments_plain_a500k_4layers.sh)：固定使用 `a500k_4layers.json`，日志写入 `parallel_refine/logs/plain_a500k_4layers/`。

脚本开头集中配置 config 路径、`SEEDS`/`GPU_IDS`、recipe 子集、`MAX_JOBS`、各阶段开关、日志目录和 Y 锁定开关。所有 GPU 阶段都通过 `CUDA_VISIBLE_DEVICES` 选择脚本指定的物理 GPU；experiment config 中的设备保持 `[-1]`。

默认脚本执行完整六阶段流程：

1. 生成/验证 event-disjoint A/B/Y split，并串行构建公共 A/B/Y processed cache，避免五个 seed 竞争生成同一文件；此时不运行模型、不计算 Y 指标；
2. 并行训练五个 Parallel seed；
3. 为每个 Parallel checkpoint 生成 B-train/B-val frozen feature cache；
4. 将每个 `(seed, recipe)` 作为独立任务并行训练 DNN；默认 5 seeds × 4 recipes 共 20 个任务，以 `MAX_JOBS=12` 分批限制同时运行的进程数，每个任务使用独立日志；XGBoost 开关保留用于复现旧实验，但默认关闭；
5. 进入锁定阶段后，为各 Parallel checkpoint 并行生成 Y frozen feature cache，这是第一次在 Y 上运行训练后的模型；
6. 运行 `evaluate --model direct_dnn`，统一评估 direct Parallel 和 DNN，并为 direct Parallel 生成 track-origin confusion matrix 和 track-pair match/other 图片。

Stage 1 的 `PROCESSED_SPLITS` 默认包含全部 A/B/Y split；Y 在这里仅做与模型无关的确定性预处理。`RUN_Y_CACHE` 与 `RUN_Y_EVALUATION` 分开控制，因此可以复用已有 checkpoint-bound Y cache，仅重跑评估。默认两个 Y 开关均为 `true`，代表这是最终锁定后的完整运行脚本；在 pilot 或任何仍会根据 B 调整方案的阶段，应先把二者都改成 `false`，避免提前在 Y 上运行模型或查看最终指标。
