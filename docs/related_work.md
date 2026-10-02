# SerialFlavour 相关研究与跨领域数据集调研

调研日期：2026-10-02。面向当前论文的 **frozen post-refinement、辅助任务与主任务的相互作用，以及跨领域复现**。

**AI for Science 补充见第 10 节：**列出数据规模、建议子集、已发表多任务 Transformer 与需要自行改造的部分，并给出小规模训练协议。该方向优先考虑 QM8、QM7-X，以及 Materials Project＋CrystalTransformer。

## 1. 结论与优先选择

**有比 PAD-Net（2018）、MTI-Net（2020）更新且相关的工作，但需要按训练协议区分。** 2022–2025 年的 InvPT、TaskPrompter、MLoRE、TaskDiffusion、BIM 等研究，说明如何组织任务特征仍是活跃问题；它们主要研究联合训练时的交互。与本项目二阶段协议更接近的线索是 **Post-hoc Concept Bottleneck Models（ICLR 2023）** 和 **ScaLearn（Findings ACL 2024）**。另一个特别值得深入的方向是 **part-based recognition**：它把局部语义预测用于全局分类，与 track-origin → jet-flavour 的关系更直接。[PCBM 原文](https://openreview.net/pdf?id=nA5AZ8CEyow)、[ScaLearn 原文](https://aclanthology.org/2024.findings-acl.699.pdf)、[Part-based models 原文](https://people.eecs.berkeley.edu/~daw/papers/part-iclr23.pdf)

建议优先推进以下组合：

| 目标 | 首选 | 原因 |
|---|---|---|
| 用较小工程改动验证“全局主任务＋局部辅助任务” | **MASSIVE 的单语言 intent＋slot** | 句子分类对应 jet 分类，token 标注对应 track origin；同一样本具有两级监督 |
| 与论文现有 CV 文献直接衔接 | **NYUD-v2：分割＋深度＋法向** | 大量近期 MTL 方法沿用，可比较共享特征与辅助预测的增量作用 |
| 保留“局部结构帮助全局分类”，并进一步做规模研究 | **PartImageNet → PartImageNet++** | 有部件监督和对象类别；2024 年 PIN++ 扩展到 ImageNet-1K 类别范围 |
| 快速检验全局属性读出、连接 concept bottleneck 文献 | **CUB-200-2011** | 适合 embedding-only、attribute-only、concat/residual 对照 |
| 专门检验 DNN-g 的关系结构解释 | **PartNet** | 点、部件实例、对象层次接近 track、vertex、jet；但需处理标签捷径 |
| 增加 AI for Science 方法学检验 | **QM8＋QM7-X；或 MP＋CrystalTransformer** | 前者覆盖低成本回归与逐原子辅助，后者连接 2025 年已发表多任务架构；详见第 10 节 |

以上是基于结构适配与实施成本的判断，不是实测效果排名。数据源与具体限制见第 5 节。

**机制主论点建议放在“在有限读出能力下，任务相关结构是否仍可被更有效地利用”。** 当前证据不足以声称辅助任务包含不可恢复的独有信息，也不足以把 refinement 增益定义成辅助任务是否已经“耗尽”的充分必要判据。

## 2. 当前项目：本次阅读确认了什么

### 2.1 当前主稿的研究对象

已阅读论文的 main、全部 sections、参考文献和 TODO。当前标题将工作定位为 *A Post-hoc Probe for Auxiliary-task Exhaustion*；摘要仍待完成。方法是：在 A 上联合训练主任务和辅助任务；冻结整个上游；在不重叠的 B 上，仅用主任务标签训练下游；在共同测试集评估。主稿描述的主要模型为约 122k 参数的 GN2-inspired Transformer。

- **DNN-o**：只读全局 pooled embedding。
- **DNN-a**：加入池化 origin 概率、pair-weighted embedding 和 pair 统计。
- **DNN-g**：读取完整 frozen track embeddings 与预测 pair 图，再与全局 embedding、origin 汇总拼接。
- **DNN-o 的上游仍然经过多任务训练**。它控制显式辅助读出，不能充当“没有辅助监督”的 single-task baseline。
- DNN-g 相对 DNN-o 同时增加了对完整 track 表示的访问、关系处理和容量；二者差值不能全部归因于 pair 预测。

来源：[Method](/D:/hep_analysis/gn2_study/SerialFlavour-paper/sections/02_Method.tex)、[cache.py](/D:/hep_analysis/gn2_study/SerialFlavour/src/parallel_refine/cache.py:44)、[graph_refiner.py](/D:/hep_analysis/gn2_study/SerialFlavour/src/parallel_refine/graph_refiner.py:123)。

### 2.2 哪些已有结果应继承，哪些需要补齐

旧材料由独立 subagent 只读核查，并回查相关报告、CSV 与部分 manifest。以下状态只覆盖本次检查范围。

| 项目 | 已有证据 | 对下一步的影响 |
|---|---|---|
| 增强 upstream task head | 已做过 128–64–32 task head 扩展 | 按当前协议确认/补齐，不能当作全新建议 |
| single-task / auxiliary supervision 消融 | 旧 modular 管线有 P3 jet-only、P1 origin、P2 pair、P0 两项 | 旧 pair-target 定义与当前不同，不能直接拼入主实验 |
| truth/oracle | 旧管线有 T1/T3 等诊断 | 继承思路；当前 DNN-a/g 仍需同协议 predicted/truth 比较 |
| OOF、CCA、重复 pooled 对照 | 已运行，有报告和 CSV | 加强解释与独立测试 |
| 新增 feature block 的线性 residual probe | 已运行；多个 residual 接近随机参考 | CCA-private 强预测力不能直接证明新增 block 有独有信息 |
| 完整容量匹配、uniform/shuffled pair 图 | 找到设计建议，未找到当前主协议的完整结果 | 是机制实验优先缺口 |

本地依据：

- [增强 task head 与 GN2 探索](/D:/hep_analysis/gn2_study/SerialFlavour/local/paper_materials/update_2026-9-11_extension.md:25)。
- [旧 P0–P3 / oracle 报告](/D:/hep_analysis/gn2_study/SerialFlavour/local/results/Archived/modular_analysis_3seeds/modular_analysis_3seeds_p0_p1_p2_p3_results_report_zh.md:23)。
- [E2 probe 的结论边界](/D:/hep_analysis/gn2_study/SerialFlavour/local/results/parallel_refiners_e2/embedding_probe/report.md:5)及[block residual 结果](/D:/hep_analysis/gn2_study/SerialFlavour/local/results/parallel_refiners_e2/embedding_probe/report.md:99)。
- [关系结构控制的既有设计](/D:/hep_analysis/gn2_study/SerialFlavour/local/reports/experiment1_2_and_loss_weight_design_20260907.md:282)。

**证据边界：**E2 probe 使用 B-val，GNN checkpoint 又曾用 B-val 选择。event-grouped OOF 保护的是 probe 拟合环节，不能将整个选择过程变成独立测试。该报告第 133 行关于 concat CCA-private 的解释与前面的 block residual 修正不一致，应采用后者。GN2 大模型旧探索记录了测试涉及公开 checkpoint 的训练数据，不能据此认定大尺度增益已经消失。

另一个重要区别是：当前实现将同一个 task-head hidden-dims 配置用于 jet、origin、pair 三个头，旧增强头扩展因而同时改变三者。该扩展有价值，但并不纯粹隔离主任务头容量；后续应加入“只增强 jet head，辅助头结构固定”的实验。重新训练上游仍可能改变共享表示，需与冻结同一上游后的读出比较一起解释。[三头共用配置](/D:/hep_analysis/gn2_study/SerialFlavour/src/parallel_model.py:56)

主稿也明确标记：部分 data-size 图使用旧结果，部分最优 allocation 按测试集表现选择。这些现象可以提出假说，暂不宜作为普遍 scaling law。当前主结果主要是 light-jet rejection 改善，并伴随 heavy-flavour confusion 的取舍。[Results](/D:/hep_analysis/gn2_study/SerialFlavour-paper/sections/03_Results.tex)、[Discussion](/D:/hep_analysis/gn2_study/SerialFlavour-paper/sections/04_Discussion.tex)

## 3. 近期研究：哪些最接近，能够支持什么

### 3.1 先区分三种协议

~~~mermaid
flowchart TB
    A["联合多任务训练：更新共享表示与各任务头"] --> B["训练中交互：特征融合、路由、联合解码"]
    A --> C["训练完成后冻结上游"]
    C --> D["独立主任务读出：只用 embedding"]
    C --> E["独立辅助读出：embedding + 预测局部/关系信息"]
    F["分别训练的任务模块或概念库"] --> G["冻结模块后学习目标任务组合"]
~~~

D/E 对应本项目核心比较；B 是 PAD-Net/MTI-Net 后续工作的主要位置；G 是 ScaLearn/PCBM 等邻近方向。**网络中有两个预测阶段，不自动等于上游冻结、数据不重叠的二阶段训练。**

### 3.2 联合训练中的任务交互：更新 Introduction 的文献

“与本项目关系”是本报告的比较判断，公开论文的性能不代表 SerialFlavour 的预期收益。

| 工作与已核实发表信息 | 主要做法/研究对象 | 联系及边界 |
|---|---|---|
| **InvPT**，ECCV 2022 | 在逐步提高分辨率的 Transformer 解码中交互任务特征 | MTI-Net 的后续线索；研究联合 dense prediction |
| **TaskPrompter**，ICLR 2023 | 空间与通道层面的任务 prompting | 有选择地组织任务视图；task prompt 不等于已训练辅助头的 soft 输出 |
| **MLoRE**，CVPR 2024 | 用低秩专家混合组织共享与任务专属计算 | 提供容量/任务共享的比较对象，可借鉴低成本 fusion |
| **Going Beyond Multi-Task Dense Prediction with Synergy Embedding Models**，CVPR 2024 | 用 synergy embedding 与交互模块增强解码 | 补充 2024 年任务交互脉络，不是冻结后读出增益的直接证据 |
| **Optimizing Dense Visual Predictions Through Multi-Task Coherence and Prioritization**，WACV 2025 | 联合考虑 coherence 与 prioritization；正文明确 fine-tune backbone | 帮助设计“交互架构 vs 损失权重”的分离对照 |
| **TaskDiffusion**，ICLR 2025 | 在 decoder 中联合去噪，学习任务关系 | 迭代 refinement 与本项目 post-hoc 训练协议不同 |
| **BIM**，ICCV 2025 | 双向和多尺度扫描，控制任务/位置交互复杂度 | 接近“交互压缩会丢失结构”的问题，不能据此证明 frozen readout 的机理 |

一手来源与复现入口：

- **InvPT**：[论文](https://arxiv.org/abs/2203.07997)；**TaskPrompter / InvPT**：[作者代码与模型索引](https://github.com/prismformore/Multi-Task-Transformer)、[TaskPrompter README](https://github.com/prismformore/Multi-Task-Transformer/blob/main/TaskPrompter/README.md)。
- **MLoRE**：[CVPR 2024 正文](https://openaccess.thecvf.com/content/CVPR2024/papers/Yang_Multi-Task_Dense_Prediction_via_Mixture_of_Low-Rank_Experts_CVPR_2024_paper.pdf)。
- **Synergy Embedding**：[CVPR 2024 正文](https://openaccess.thecvf.com/content/CVPR2024/papers/Huang_Going_Beyond_Multi-Task_Dense_Prediction_with_Synergy_Embedding_Models_CVPR_2024_paper.pdf)。
- **Coherence/Prioritization**：[会议条目](https://openaccess.thecvf.com/content/WACV2025/html/Fontana_Optimizing_Dense_Visual_Predictions_Through_Multi-Task_Coherence_and_Prioritization_WACV_2025_paper.html)、[正文](https://openaccess.thecvf.com/content/WACV2025/papers/Fontana_Optimizing_Dense_Visual_Predictions_Through_Multi-Task_Coherence_and_Prioritization_WACV_2025_paper.pdf)。
- **TaskDiffusion**：[ICLR 条目](https://proceedings.iclr.cc/paper_files/paper/2025/hash/2f2b1d6bbd50865eca40e2774a057eef-Abstract-Conference.html)、[作者实现与模型](https://github.com/YuqiYang213/TaskDiffusion)。
- **BIM**：[ICCV 条目](https://openaccess.thecvf.com/content/ICCV2025/html/Cao_Enhancing_Mamba_Decoder_with_Bidirectional_Interaction_in_Multi-Task_Dense_Prediction_ICCV_2025_paper.html)、[正文](https://www.openaccess.thecvf.com/content/ICCV2025/papers/Cao_Enhancing_Mamba_Decoder_with_Bidirectional_Interaction_in_Multi-Task_Dense_Prediction_ICCV_2025_paper.pdf)。

**建议：**Introduction 可选 TaskPrompter、MLoRE、TaskDiffusion、BIM 说明演进，不必全部罗列。机制章节更应依赖下面的冻结组合、局部监督和任务关系研究。多数 dense prediction 论文把多个输出都当作目标；本项目优先 jet-flavour，需要分别报告主任务收益与辅助任务质量。

### 3.3 更接近二阶段协议的工作

#### A. ScaLearn：冻结任务模块，再学习目标任务组合

**Frohmann et al., Findings ACL 2024，ScaLearn: Simple and Highly Parameter-Efficient Task Transfer by Learning to Scale。**

原文 §2–3 明确区分 source-task learning 与 transfer：先获得任务 adapters，冻结这些模块，再学习目标任务的缩放/组合参数与新任务头。实验覆盖 GLUE、SuperGLUE、HumSet；“8 个参数”的极简例子只计算特定配置的 transfer 参数，不代表整个模型或分类头。[论文](https://aclanthology.org/2024.findings-acl.699.pdf)、[引用信息](https://aclanthology.org/2024.findings-acl.699/)、[代码](https://github.com/CPJKU/ScaLearn)

**意义与差别：**任务知识的学习与使用可以分开优化。ScaLearn 主要组合分别训练的内部 adapters；我们的辅助头来自共同训练的上游，并检验相对 embedding-only 的收益。

**可借鉴实验：**在 concat MLP/GNN 前增加简单 task-block scaling 或小型 residual fusion。若极小读出已经有效，就不必把收益全部归因于下游复杂性。该文的语言模型预训练资源不能直接用作本项目“低资源”的证据。

#### B. Post-hoc Concept Bottleneck Models：语义读出与残差支路

**Yuksekgonul, Wang and Zou, ICLR 2023，Post-hoc Concept Bottleneck Models。**

PCBM 将已有 embedding 投影到 concept bank，再训练可解释分类器；PCBM-h 加入从 embedding 出发的 residual predictor。目标包括解释、模型编辑和保留预测性能，概念可来自外部标注或多模态模型。[正文](https://openreview.net/pdf?id=nA5AZ8CEyow)

**意义与差别：**它提供“表示 → 任务相关中间量 → 目标读出”的先例。我们的 origin/pair 是物理任务预测，不等同于 CAV projection；保留 embedding bypass 后，也不能声称全部决策都由语义概念解释。

**可借鉴实验：**比较 auxiliary-only、embedding-only、concat、additive residual。auxiliary-only 很强但 concat 无增益，可能意味着替代性视图；concat 改善则仍需排除维数、局部输入和容量因素。

另有 **TMLR 2024 独立复现**指出部分原始结果未能充分复现。应以自己的公平对照为准，不预设 PCBM 必定无损。[复现论文](https://openreview.net/pdf?id=8UfhCZjOV7)、[复现代码](https://github.com/dgcnz/FACT)

#### C. Part-based recognition：局部语义监督怎样帮助全局分类

**Sitawarin et al., ICLR 2023，Part-Based Models Improve Adversarial Robustness。**

该工作以 part segmenter 加小型分类器端到端学习，用局部监督改善对象分类和鲁棒性，并比较有/无部件监督。其关注点是鲁棒识别，协议不等同于冻结诊断。[作者正文](https://people.eecs.berkeley.edu/~daw/papers/part-iclr23.pdf)、[实现](https://github.com/chawins/adv-part-model)

**对我们的意义：**比“像素分割＋深度”的对称多任务更直接连接全局主任务与局部辅助任务。可检验：局部监督已改善上游表示后，预测的部件结构是否仍帮助第二阶段分类。

**Li et al., ECCV 2024，PartImageNet++ Dataset: Scaling up Part-based Models for Robust Recognition** 提供 ImageNet-1K 全类别的部件标注与 multi-scale part-supervised model。它适合提供“有局部监督但不显式重用输出”的比较思路。[正式正文](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/09027.pdf)、[实现](https://github.com/LixiaoTHU/PartImageNetPP)

作者在 2026 年另有 *PartImageNet++ Dataset: Enhancing Visual Models with High-Quality Part Annotations* 的 arXiv 扩展条目。本次确认条目与摘要，未完成新增实验和发表状态的全文复核，先以 ECCV 2024 版本作为已核实引用。[2026 条目](https://arxiv.org/abs/2601.01454)

### 3.4 主辅任务关系：机制与上游控制

#### D. ForkMerge：从主任务角度评价辅助学习

**Jiang et al., NeurIPS 2023，ForkMerge: Mitigating Negative Transfer in Auxiliary-Task Learning。**

该文以主任务为中心，研究缓解 auxiliary-task learning 的负迁移。它提醒我们：任务相关不保证给定训练策略下能改善主任务。它研究上游学习过程，不能证明冻结后的下游已经“消除了训练期梯度冲突”。[会议正文](https://proceedings.neurips.cc/paper_files/paper/2023/file/60f9118a849e8e9a0c67e2a36ad80ebf-Paper-Conference.pdf)、[原始条目](https://arxiv.org/abs/2301.12618)

建议以主任务验证指标选择辅助配置；辅助准确率更高，不自动意味着下游收益更大。本次阅读主要覆盖摘要和方法说明，尚未逐项核验全部消融。

#### E. Royer et al.：保留简单 scalarization 的强对照

**NeurIPS 2023，Scalarization for Multi-Task and Multi-Domain Learning at Scale。**

该文研究任务/领域组合、模型规模和损失权重，并用 population-based training 搜索权重。复杂 MTL 优化算法需要与认真调过权重的简单加权损失比较；小模型指导大模型选权重的经验仍受该文任务范围限制。[会议条目](https://proceedings.neurips.cc/paper_files/paper/2023/hash/368559ed8ede03b21f624feaeb3a5867-Abstract-Conference.html)、[正文](https://proceedings.neurips.cc/paper_files/paper/2023/file/368559ed8ede03b21f624feaeb3a5867-Paper-Conference.pdf)

**可检验问题：**refinement 是否降低主任务对 auxiliary loss weights 的敏感性？应比较 native、embedding-only、auxiliary-rich 的性能曲面、最优性能和权重敏感程度；大 gain 不能直接指认“错误的 loss weights”。

#### F. 从共享表示看资源竞争

**Rethinking Resource Competition in Multi-Task Learning: From Shared Parameters to Shared Representation，IEEE Access 2024。**

该经验研究认为若干梯度解释量与性能的因果关系较弱，提出分析共享表示如何被各任务头利用。其 Feature Disentanglement 视角与本项目相关，但不能推广成“梯度冲突完全无关”。本次核验到期刊信息和摘要，未复现其分析。[期刊条目，DOI 10.1109/ACCESS.2024.3429281](https://ieeexplore.ieee.org/document/10599476/)

**对我们的意义：**同一 frozen representation 的不同读出，是分析表示使用方式的工具，但需要容量、优化和输入访问控制；读出增益并不是内部表示信息量的直接测量。

#### G. 非对称关系与辅助任务训练不足

- **Graffeuille et al., 2024 arXiv，Enabling Asymmetric Knowledge Transfer in Multi-Task Learning with Self-Auxiliaries。** 研究 A 帮 B、B 却损害 A 的非对称关系。对我们有用的是方向性：origin → flavour 与 flavour → origin 应分别评价。本次按预印本记录，未核实后续正式发表。[原始条目](https://arxiv.org/abs/2410.15875)
- **Li et al., 2024 arXiv，Unprejudiced Training Auxiliary Tasks Makes Primary Better: A Multi-Task Learning Perspective。** 讨论辅助任务权重偏小造成训练不足，并结合不确定性与梯度信息。这提示“没有 refinement 收益”也可能是辅助预测质量不够。本次按已核实 arXiv 版本记录，期刊线索未完成核验。[原始条目](https://arxiv.org/abs/2412.19547)

#### H. 2026 年的任务优化方向

**Choi et al., CVPR 2026，TaskForce: Cooperative Multi-agent Reinforcement Learning for Multi-task Optimization。**

它用任务梯度 Gram matrix 和损失构建优化策略，在 NYU-v2、Cityscapes、QM9 上评估。这是近期训练期协调的实例，关注点是优化器，第一轮迁移不必先引入这种复杂机制。本次获取 CVF 摘要与正文部分可检索内容，未复核所有理论条件和代码可运行性。[CVPR 条目](https://openaccess.thecvf.com/content/CVPR2026/html/Choi_TaskForce_Cooperative_Multi-agent_Reinforcement_Learning_for_Multi-task_Optimization_CVPR_2026_paper.html)、[正文](https://openaccess.thecvf.com/content/CVPR2026/papers/Choi_TaskForce_Cooperative_Multi-agent_Reinforcement_Learning_for_Multi-task_Optimization_CVPR_2026_paper.pdf)

### 3.5 对论文定位的建议

保留 PAD-Net/MTI-Net 作为起点，然后按三个段落组织：

1. **任务间特征交互持续发展。** 用 TaskPrompter/MLoRE/TaskDiffusion/BIM 说明联合训练中的组织方式。
2. **任务知识的学习与使用可以分开。** 用 ScaLearn/PCBM 连接冻结后的组合，明确模块来源和实验目的。
3. **我们的具体问题。** 在同一 frozen multitask upstream 下比较原主头、embedding-only 与辅助/关系 readout，并研究数据分配、容量、辅助质量的影响。

本次没有查到一篇已核实论文同时完整匹配“共同 MTL 上游＋冻结后在不重叠 B 训练＋局部/关系辅助预测＋embedding-only 控制＋固定总预算 scaling”的全部条件。**这不是不存在先例的证明，不能据此写“首次”。** 潜在贡献更适合落在控制严谨的诊断、结构读出与条件依赖的实证结论。

## 4. 把 post-refinement 变成可检验的机制问题

以下是基于当前实现与文献提出的分析框架，均为待验证假说。

### 4.1 区分“信息存在”与“给定读出能够利用”

设完整 frozen token/track 表示为 \(H\)，全局池化为 \(g=P(H)\)，辅助预测为 \(u=A(H,g)\)。在固定训练完的模型参数、确定性推断且没有额外输入时：

\[
I(Y;u\mid H)=0,
\qquad
I(Y;u\mid g)\ \text{则可能大于零。}
\]

因此，相对完整 H 没有新增信息，与相对压缩 g 有额外预测价值，可以同时成立。即便某个辅助量也是 g 的函数，它仍可能改善有限容量模型的可读性或样本效率，但这时不能称为增加 Shannon 信息。实际性能差也不是条件互信息的直接估计。

建议分开测量：

\[
\Delta_{\rm readout}=S(r_g(g))-S(h_{\rm native}(g)),
\]
\[
\Delta_{\rm aux}=S(r_{g,u}(g,u))-S(r_g(g)).
\]

S 是方向统一为“越大越好”的主任务指标。第二项只有在下游预算、输入访问范围、选择规则得到控制后，才接近辅助读出的增量价值。DNN-g 还需增加 **H-only readout**，因为它已经绕过全局 pooling。

### 4.2 五个竞争解释与对应实验

| 假说 | 如果成立，预期看到什么 | 优先对照 | 不能据此推断什么 |
|---|---|---|---|
| 原主任务头偏弱/优化受限 | 加强原头后 DNN-o 优势缩小 | 按当前协议补齐已有 strong-head 对照；匹配容量和训练预算 | 辅助监督没有价值 |
| 全局 pooling 丢失可用结构 | H-only 已追回多数收益 | 同样访问 H，不用 predicted origin/pair；匹配参数 | pair 预测提供独有信息 |
| 预测关系帮助组织局部计算 | predicted graph 优于同容量 uniform、打乱/重连图 | 固定节点和 context，仅改变 edge；对照也重训 refiner | 任意 test-time 破坏导致退化就是因果证明 |
| 第二阶段主任务优化改善边界 | embedding-only 或 logits-only 已解释许多收益 | 强 native head、logits-only MLP、冻结后同结构再训练 | 下游“纠正了上游梯度冲突” |
| 辅助质量与权重决定剩余收益 | 改变权重/噪声后收益系统变化 | 零权重、单项/双项监督；固定 A/B 和读出；oracle | 辅助 accuracy 高必然更有用 |

**关系控制做两类。** 推断时替换边，说明已训练模型是否依赖该输入，同时引入分布偏移；在 uniform/shuffled 边上重训同结构模型，更适合评估真实关系的增量价值。节点和边按同一置换同步重排应保持结果不变，这是正确性检查；只改变对应关系才是结构破坏。

**DNN-a 的精确限制：**对于其中的 pair-weighted embedding 分支，

\[
m_i=\sum_j\alpha_{ij}h_j,\qquad
z_{\rm pair}=\sum_i a_i m_i
=\sum_j\left(\sum_i a_i\alpha_{ij}\right)h_j.
\]

该分支仍是一阶重加权汇总；额外 mean/max/sum pair 统计另算。DNN-a 不完整保留 edge identity。DNN-g 的非线性消息传递更有机会利用关系，但是否确实如此需由控制实验判断。[汇总实现](/D:/hep_analysis/gn2_study/SerialFlavour/src/parallel_refine/cache.py:44)

### 4.3 Probe、oracle、calibration 的证据边界

- **Probe**：分别用线性、小 MLP、较强非线性读出，固定折、搜索预算和样本。CCA、表示相似度、residual 单独分类失败都不能证明条件独立；residual 与 g 的交互也可能重要。已有 residual 结果应约束论述，不能变成“辅助完全无信息”。
- **Oracle**：truth 辅助量可诊断预测质量是否是瓶颈，但属于额外信息条件。它是参考上界式实验；是否真的达到上界还受读出和优化限制，不能并入可部署收益。
- **Calibration**：ECE/NLL/Brier 与 ROC/rejection 分别报告。对最终标量 discriminant 的严格单调变换不改变排序和固定效率 rejection；多类概率变换则可能改变复合 discriminant 排序。仅凭分数分布变化，不能把 rejection gain 命名为 calibration gain。
- **无增益**：只说明在测试过的读出、预算、样本与指标下未发现收益。若要支持“已充分利用”，应预定最小有意义收益阈值，再看不确定区间能否排除它；五个 seed 未显著优于零不等于等效。

## 5. 可迁移的数据集：适配、成本与风险

### 5.1 选择标准

重点筛选：同一样本有主辅标签；测试时辅助真值不作为输入；能够冻结上游并独立训练读出；有可维护的划分与获取入口；能区分全局、局部、关系层次。

“迁移”在此主要指 **跨领域复现方法论并重新训练上游**。不是将 JetSet checkpoint 直接应用于图像/语言，也不是默认跨领域权重迁移有效。下面的成本是相对工程判断，未实测 GPU 时间。

| 数据集 | 规模/标注概况 | 推荐主任务 → 辅助任务 | 第一阶段优先度 | 成本 |
|---|---|---|---|---|
| MASSIVE | 1.0：51 语言、约百万平行语句；1.1 增至 52 语言；60 intents、55 slot types | intent → token slots | 高：最容易保留全局＋局部结构 | 低至中，先单语言 |
| NYUD-v2 | 1,449 标注 RGB-D 帧，464 scenes；另有 407,024 未密集语义标注帧 | segmentation → depth、normal | 高：直接对接 dense MTL 文献 | 中 |
| PartImageNet | 24,095 图像、158 类、11 super-categories | 细粒度 object class → part segmentation | 高：全局分类＋局部监督 | 中 |
| PartImageNet++ | 覆盖 ImageNet-1K 类别的部件标注；实际样本需按发布过滤清单统计 | object class → multiscale parts | 第二步：规模扩展 | 高 |
| CUB-200-2011 | 11,788 图像、200 类、312 attributes、15 part locations | bird species → image attributes | 高：快速机制试验 | 低至中 |
| PartNet | 26,671 shapes、24 object categories、573,585 part instances | object class → part semantics / same-part-instance | 第二步：关系机制 | 中至高 |
| ShapeNetPart | 16,881 shapes、16 类、50 semantic parts | object class → part segmentation | 备选：点云原型 | 中 |
| PACO | 75 object classes、456 object-part classes、55 attributes | object crop class → parts、attributes | 第二步：复杂真实场景 | 高 |
| QM9 | 原始 133,885 小分子，结构＋量子化学性质 | 一个预定性质 → 若干其他性质 | 中：多任务回归扩展 | 低至中，小图模型 |

各行一手来源、版本区别与实施建议如下；没有将“数百万像素/平行译句”视为数百万独立样本。

### 5.2 MASSIVE：最自然的全局＋局部迁移

**来源与事实。** MASSIVE 论文正式发表于 ACL 2023；原始版本是 51 语言的平行 intent/slot 数据，1.1 新增 Catalan。官方 JSONL 包含原句、slot 标注、intent、原始 id 和 train/dev/test partition。[作者介绍](https://www.amazon.science/publications/massive-a-1m-example-multilingual-natural-language-understanding-dataset-with-51-typologically-diverse-languages)、[官方数据与建模代码](https://github.com/alexa/massive)

**建议映射。**

- 输入为原始文本 token，主任务为 intent classification，辅助任务为 slot tagging。
- 全局句向量对应 g，token embeddings 对应 H；slot probabilities 对应 origin probabilities。
- 第一版只做 embedding-only、pooled slot 和 token-level structured readout，不必强加 pair task。
- 如需关系扩展，可由标注 span 派生“两个 token 是否同一 slot 实例”。这属于新增的派生监督；两个不同 span 即使 slot type 相同，也不应标成同一实例。padding、特殊 token 和无实体 token 的 pair 需明确屏蔽/定义。

**划分。** 首轮用一个语言，保留官方 dev/test，只在 train 内拆 A/B。多语言扩展按同一个源语句 id 分组，避免一个语句的翻译同时进入 A、B 或测试；也要检查同语言重复文本。报告唯一源语句数与语言数，不能将翻译扩增的行数当作独立语义样本量。

**指标与陷阱。** 主指标 intent macro-F1/accuracy，辅报 slot span-F1、联合 exact match；slot 类型较多不代表 BIO 标签数恰为 55。scenario 是较粗的 intent 层次，不宜未经说明当作独立辅助机制。预训练文本 encoder 要作为固定共同资源；若研究从小数据开始的 scaling，另设随机初始化小 Transformer。数据许可及 SLURP 继承条款应在下载时固定版本核对。

**最适合回答：**局部辅助预测是否在句子 embedding 已经多任务训练后仍有可用增量。语序与 track 集合不同，需要保留 positional encoding，而不能直接照搬无序集合模型。

### 5.3 NYUD-v2：与近期 CV MTL 最容易形成共同基准

**来源与事实。** 原始数据包含对齐的 RGB/depth、对象类别和实例标注。常用 1,449-frame 基准为 795 train / 654 test；近期实现提供预处理的数据包。原始 407,024 帧不是同等完整的语义监督数据。[NYU 官方数据页](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html)、[近期可复用数据流程](https://github.com/YuqiYang213/TaskDiffusion)

**建议映射。** 仅 RGB 为输入，semantic segmentation 为主任务，depth/normal 为辅助；输入端不提供真值 depth。上游输出冻结的 feature maps 与预测 depth/normal。下游只用 segmentation labels，对比 features-only decoder 与加入辅助预测的同容量 decoder。

此处主任务是密集输出，应保留空间 feature map；不能先全局池化再强行恢复逐像素预测。它检验的是“冻结后的辅助读出”原则，不能单独验证 jet 的全局 pooling 机制。法向可由深度几何推导，两项监督并非完全独立的标注来源。

**划分与统计。** 从 795 个训练帧中按 scene 分组拆出 A/B/validation；官方 654 帧作为最终 holdout。若复用上游公开 checkpoint，必须核对其训练图像是否已经包含 B：已经见过 B 的模型只能作为公开模型上的后处理实验，不能称作严格 A/B 分离。分割 mask 类别数、ignore index、depth 单位/裁剪和 normal 有效区必须跟随所选预处理版本。

**指标。** 主报 segmentation mIoU、各类 IoU；辅报 depth RMSE、normal angular error。以 scene/image 为统计单元，像素不是独立重复。样本很少，适合低数据机制研究，不适合单靠它宣称百万样本 scaling。使用 13 类或 40 类等不同协议时应明确，不能直接合并各论文表格。

### 5.4 PartImageNet / PartImageNet++：优先推荐的 CV 全局分类路线

**来源与事实。** PartImageNet 作者仓库区分 **PartImageNet_Seg**（各 split 类别相同）与 **PartImageNet_OOD**（类别不相交）。README 的 109/19/30 类、16,540/2,957/4,598 图表对应原来的类别不相交划分，不能直接用于常规 158 类闭集分类。[官方仓库](https://github.com/TACJu/PartImageNet)

**建议映射。** 主任务为细粒度对象类别，辅助为部件分割；frozen patch features 对应 H，image embedding 对应 g。比较：

1. g-only 分类；
2. g＋predicted part 面积、位置、pooling 汇总；
3. 相同 H 上的无辅助 attention/DeepSets；
4. predicted-part 引导的局部汇总。

这是本报告提出的实验，不是数据集默认 benchmark。第一轮选 Seg split，并检查每个目标类在 train/dev/test 的覆盖；需要预先定义部件标签的统一方式。

**最关键的两项控制。** 第一，部件词表按 super-category 设计，某些 part 标签可能直接暗示大类。优先在同一 super-category 内做细粒度分类，或使用合并后的类无关部件词表，再检验结论能否跨组重复。预测这些标签本身不是测试泄漏，但可能让“结构机制”退化成重编码类别。第二，ImageNet 预训练可能已见过数据集图像及分类标签；干净 A/B 研究应核对图像重叠，使用已排除重叠的预训练来源或从头训练。所有方法共享同一预训练模型只能保证部分公平，不能消除数据见过的问题。

**规模扩展。** PIN++ 的 ECCV 2024 发布覆盖 ImageNet-1K 类别；作者同时发布部件层次关系、类别映射及 discarded-data 清单。数据卡展示的标注文件体积不代表已经包含全部原图。实际可用 N、类别覆盖与人工/自动标注流程应在下载后核实，不把 1K 类别覆盖等同于每张 ImageNet 图像都有相同质量的 dense mask。[论文](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/09027.pdf)、[官方标注发布](https://huggingface.co/datasets/lixiao20/PartImageNetPP)、[代码](https://github.com/LixiaoTHU/PartImageNetPP)

**指标。** 细粒度 accuracy/macro-F1、part mIoU；后续加入 corruption 或背景变化时，报告所有模型在同样扰动下的退化。不能用普通精度改善自动推出 adversarial robustness。

### 5.5 CUB-200-2011：快速验证语义属性读出

**来源与事实。** 数据包含 200 种鸟、11,788 图像、312 属性和 15 个部位位置，常用官方划分为 5,994 train / 5,794 test。[Caltech 数据页](https://www.vision.caltech.edu/datasets/cub_200_2011/)、[原始技术报告](https://authors.library.caltech.edu/records/cvm3y-5hh21/files/CUB_200_2011.pdf)

**建议映射。** species 为主任务、image-level attribute prediction 为辅助，训练共享 encoder；B 上只给 species labels，输入使用预测属性。比较 g-only、attributes-only、concat、residual。若需要局部结构，再增加 part localization，但首轮不必这样做。

**关键限制。** 优先使用图像级属性与可见性/置信度，不能不加说明地改用类别平均属性。类别原型属性本来就可能成为类码；部分 CBM 预处理还会筛选属性或按类聚合，与原始逐图像标注不等价。辅助概率的细微变化也可能编码属性语义之外的类别信息，所以“预测属性有用”不等于“模型以人类预期的属性因果推理”。

**划分与成本。** 从官方 train 中拆 A/B/validation，test 保留；按类别分层并检查图像重复。200 类下 B 每类样本可能很少，适合小型 head、预先限定的搜索预算与低数据曲线。这个数据集主要检验语义视图的可读性，不能独立支撑 pair-graph 机理。原图权利与下载条款需跟随官方发布核对。

### 5.6 PartNet / ShapeNetPart：关系结构最接近，但不是无陷阱的首选

**PartNet** 提供细粒度语义、部件实例和层次标签：26,671 shapes、24 类、573,585 part instances。[官方项目](https://partnet.cs.stanford.edu/)、[原始论文](https://arxiv.org/abs/1812.02713)

**建议映射。** 点集 → 对象分类；点的部件语义 → origin 类比；同一部件实例 pair → common vertex 类比。使用固定采样的较少点先实验，屏蔽 self/padding pair，再考虑稀疏图。实例 identity 与 semantic part type 必须区分：椅子的两条腿可能同类，却不是同一个实例。

**控制与适用边界。** 对象分类是此处拟议的迁移任务，不是 PartNet 原始主要 benchmark。许多 part segmentation 实现按对象类训练或输入真值对象类别；若把该实现用于主对象分类，会泄漏目标。需要改成类别未知的通用 predictor。类专属 part vocab 也可能近似给出对象类别，建议合并语义、研究同类内的结构任务或显式报告这种层次关联。按完整 shape id 划分，任何点采样和视图都跟随同一对象。

**ShapeNetPart** 更适合快速点云原型，常用 16,881 shapes / 16 类 / 50 semantic part labels；但 semantic-part equality 并不等于同一部件实例，不适合直接拿来验证“同一顶点”类比。上述目标类别输入问题同样存在。[PointNet 原始论文中的数据说明](https://geometry.stanford.edu/lgl_2024/papers/qsmg-pdlps3dcs-17/qsmg-pdlps3dcs-17.pdf)、[ShapeNet 官方入口](https://shapenet.org/)

官方 ShapeNet/PartNet 入口包含注册获取流程，代码公开不等于资产可以任意再分发。图构建的全 pair 成本随点数平方增长；相较当前最多 40 tracks，直接使用上千点会显著增加计算和 cache，首轮需要控制点数。

### 5.7 PACO：更丰富的真实部件/属性组合，适合后续扩展

PACO（CVPR 2023）提供 75 object classes、456 object-part classes、55 attributes；部件词汇本身为 200 个共享 part 类。PACO-LVIS 官方 split 为 45,790/2,410/9,443 images，另有视频来源的 PACO-EGO4D。[论文](https://openaccess.thecvf.com/content/CVPR2023/html/Ramanathan_PACO_Parts_and_Attributes_of_Common_Objects_CVPR_2023_paper.html)、[官方 schema 与统计](https://github.com/facebookresearch/paco/blob/main/docs/PACO_DATASET.md)

**建议映射。** 若追求与 jet 一一对应，先定义 object crop 分类为主任务，辅助为部件/属性；crop 的获取方式对各方法一致。使用真值 box 定义样本时，应称为给定对象 crop 的分类，不能冒充端到端检测结果。可用共享 part vocabulary 降低直接编码对象类的捷径。

**限制。** 属性和部件标注不是所有实例都穷尽覆盖，unknown 与未标注不能当负例。图像内多个 crop 共同进入同一 split；视频还需按视频/对象实例分组。原始数据入口涉及 LVIS/COCO 与 Ego4D 各自条件。它更适合第一轮有结果之后研究缺失辅助标签和现实噪声，不宜作为最省事的起点。

### 5.8 QM9：适合规模与回归，不能独立验证局部辅助机制

QM9 原始集合有 133,885 小分子的几何与量子化学性质；常见 loader 会排除部分结构，因此实验必须记录过滤后的 N 和 target 定义。[原始 Data Descriptor](https://www.nature.com/articles/sdata201422)、[作者数据索引](https://qmml.org/datasets.html)、[2025 年整理版与来源许可说明](https://zenodo.org/records/15390655)

**建议映射。** 将预先选定的一个性质，如 dipole moment，作为主任务，若干其他性质为辅助；B 阶段输入 frozen graph embedding 与预测性质，仅用主任务真值。用共同小型分子图 encoder 测试是否存在 frozen multitask regression 的读出收益。

**重要控制。** 避免把可由辅助真值代数重建的目标作为主要机制证据，例如同时用 HOMO/LUMO 辅助预测来解释 gap 收益；高度相关的热力学量也应分组处理。这类预测组合可以有实际价值，但更接近误差互补/约束一致性，不能代表广泛的辅助任务机理。

QM9 的天然辅助标签主要是分子级性质，不是每原子真实 origin 或 latent pair labels；输入原子类型、键和几何的简单重建也不等于新增物理监督。报告主任务 MAE、各辅助 MAE、训练集拟合的标准化；采用固定 molecule split，并视问题增加化学结构分组的 holdout。其优势是完整多目标和较易做 N 扫描，局限是结构类比弱于 PartNet。

## 6. 建议实施路线

### 6.1 先做两个小而完整的迁移实验

**默认组合：MASSIVE 单语言＋NYUD-v2。** 前者保留全局/局部关系且文本数据容易处理；后者建立与近期 CV MTL 工作的联系。若论文希望坚持“全局分类＋局部辅助”的统一叙事，则将 NYUD-v2 换成 **PartImageNet 同一 super-category 内的细粒度分类**，但应先解决预训练图像重叠和 split 问题。

若短期仅够一个试验：

- 强调跨领域泛化，先 MASSIVE。
- 强调 CV related work 的直接比较，先 NYUD-v2。
- 强调辅助结构怎样帮助全局分类，先 PartImageNet。
- 仅快速检查 post-hoc semantic readout 是否能迁移，可先 CUB；它不能替代后续结构实验。

不建议第一轮同时接入九个数据集。先判断 \(\Delta_{\rm aux}\) 是否在公平基线下存在，再投入 PIN++ 的规模实验或 PartNet 的图机制研究。

### 6.2 每个首轮数据集的最小实验矩阵

下面“同一上游”意味着缓存来自同一个 checkpoint、相同推断模式和输入预处理。R0–R5 均只能用 B 的主任务真值训练下游。

| 编号 | 模型/输入 | 要回答的问题 |
|---|---|---|
| U0 | 同容量 encoder，仅训练主任务；相同 A | 辅助监督本身对表示/原头是否有益 |
| U1 | 标准 MTL 上游，原主任务头；相同 A | 固定上游的起点 |
| U2 | MTL 上游，只增强主任务头，固定辅助头结构；相同 A | 原主头是否过弱；不同于已有三头同时增强，需另训上游 |
| R0 | 冻结 U1，g-only；dense task 用对应 feature maps | 一般再读出的收益 |
| R1 | 冻结 U1，auxiliary-only | 辅助预测单独能做多少 |
| R2 | 冻结 U1，g＋auxiliary summaries | 同一上游下辅助读出的增量 |
| R3 | 冻结 U1，H-only，匹配 R4 的结构/容量 | 完整局部输入本身解释多少收益 |
| R4 | 冻结 U1，H＋预测关系/局部语义 | 结构化辅助是否比 R3 有用 |
| R5 | R4 的 uniform/shuffle/重连对照，重新训练下游 | 是否使用了真实预测结构 |
| O | 与 R2/R4 对应的 truth/oracle 替换 | 辅助质量受限的诊断，单列结果 |
| T | 同架构上游用全部 A∪B 主训练预算训练，原头评估 | 两阶段是否值得牺牲一部分上游数据 |

第一批可只做 U0、U1、R0、R1、R2、T。只有看到可靠辅助增量后，再加 U2 和结构读出；MASSIVE/CUB 的第一版不要求人造 pair 任务。对 U0 也训练与 R0 同规格的读出，可以把“辅助监督改善表示”与“改善原主头训练”进一步分开。

**两种公平性要同时保留：**

1. 固定 U1 比 R0/R2/R3/R4，分析读出机制。
2. 固定总样本预算比两阶段与 T，分析实际性价比。

这两种比较的分母不同。相对只用 A 训练的 native gain，不能写成相对全部 A∪B 的 deployment gain。

### 6.3 数据、模型容量和统计规则

1. **先锁定 test。** 保留官方测试划分；从官方 train 中建立 A、B 和必要的验证子集。没有独立 test 的公开协议，应明确哪个 split 被锁定，不能反复用它选择 allocation。
2. **按独立对象划分。** JetSet 按事件、NYUD 按场景、MASSIVE 多语言按源句 id、点云按 shape、PACO 按图像/视频实例。先划分再做采样、augmentation 和 feature cache。
3. **固定选择预算。** 特征归一化、降维、任务权重、超参数与 allocation 都在训练/验证范围选择。可共用验证集，但要记录它参与了哪些选择；机制 probe 若再次复用它，应标为探索。
4. **容量不是只看最终 MLP 宽度。** 匹配整个 trainable readout 的参数量和大致计算量，包括投影、GNN、attention；同时比较同架构去掉/打乱辅助输入，避免只有容量曲线而没有输入消融。
5. **小规模筛选与最终确认分开。** 先用有限 seeds 识别实现问题、筛选极差配置；最终沿用五个 upstream seeds。下游多 seeds 用于估计优化随机性，而不是制造更多独立 upstream 重复。
6. **保留分层统计。** 对每个 upstream seed 先平均 downstream seeds，再计算相对同 upstream 的配对差/比；报告五个 upstream 汇总值的均值和样本 SD。样本 bootstrap 与训练 seed 变异分开报告，bootstrap 应按场景/事件等分组；不能把 SD 标成置信区间。
7. **预先指定主指标和最小收益。** classification accuracy 的百分点、mIoU 点数、MAE 相对下降与 rejection ratio 不应混为一个“平均提升”。主指标之外保留类间混淆/辅助质量，检查收益是否只是工作点取舍。
8. **记录真实资源。** 上游/下游训练时间、cache 体积、总推断延迟与显存。冻结只降低第二阶段训练成本；预测辅助头和关系处理仍可能增加部署成本。

### 6.4 先解耦 A/B，再研究固定总预算

首轮先固定上游 A 与 checkpoint，增加 B，测读出学习曲线；再固定 B，改变 A，测上游质量如何改变辅助增量。等这两条曲线清楚后，再做固定 A＋B 的 allocation 扫描。

一个资源可控的起点是：嵌套的三档 A、三档 B，先选一个共同容量；再用两档上游容量复核最关键的对比。具体样本数应由数据集的独立对象数量决定，而非照搬 JetSet 的百万级设置。所有档位共享锁定测试样本，使用验证集或预先规则选 allocation。

如果 B 增大后收益消失，可能是小数据正则化优势消失；如果 g-only 随读出容量增大追回收益，可能是可读性；如果 H-only 追回而 g-only 不能，可能是 pooling；如果仅真实关系图稳定更好，才更支持关系组织解释。这些是判读规则，不是预定的实验结论。

### 6.5 项目内最值得先补的三个实验

1. **将已有 strong-head 扩展对齐当前主协议，并增加只增强主头的控制。** 已有三头同时增强还改变了辅助预测能力；新控制固定辅助头结构。检查 DNN-o 增益是否缩小，而 DNN-a/g 相对它的增量是否仍在，不要只比较相对不同 native baseline 的百分比。
2. **DNN-g 的 H-only / uniform-edge / edge-shuffle 控制。** 保持 context、节点输入、参数和训练预算不变；推断干预与重新训练的对照分开报告。
3. **把已有 exploratory probe 固定规则后迁到未参与选择的样本。** 重点比较 full concat 的增量和不同读出能力，不再把高维 concat 的 CCA-private 当作新增 block 专属信息。

这三个实验与跨领域首轮可以互相支撑：本项目负责更细的结构诊断，外部数据集负责检验这些现象是否超出单一 JetSet 任务。

## 7. 阅读与引用优先级

### 第一批：直接决定论文叙事与实验

1. **ScaLearn，2024**：重点读 §2–3 的冻结边界、task head 和 transfer 参数统计；这是二阶段“任务知识再使用”的近邻。
2. **PCBM / PCBM-h，2023；独立复现，2024**：重点读 concept-only 与 residual 的角色，避免把可解释概念和全部类别信息混为一谈。
3. **Part-Based Models Improve Adversarial Robustness，2023**：重点读有/无局部监督对照和端到端训练协议。
4. **PartImageNet++，ECCV 2024**：重点读局部监督、规模、数据获取与预处理；将数据集论文和其中的模型区分。
5. **Royer et al.，2023；ForkMerge，2023**：用于设计上游权重与主任务目标的控制。

### 第二批：更新 CV 背景与选择可复用实现

- **TaskPrompter、MLoRE、TaskDiffusion、BIM**：覆盖 2023–2025 年任务特征交互；不必完整复现所有模型才能检验我们的假说。
- **Coherence/Prioritization，2025**：区分交互结构、任务权重和预训练资源。
- **TaskForce，2026**：了解最新优化方向；不作为首轮实现依赖。
- **Self-Auxiliaries / Unprejudiced Auxiliary Training**：补充方向不对称和辅助训练不足的解释；引用时保留本报告的版本核验边界。

**可立即复用的实现入口**包括 [Multi-Task-Transformer](https://github.com/prismformore/Multi-Task-Transformer)、[TaskDiffusion](https://github.com/YuqiYang213/TaskDiffusion)、[MASSIVE](https://github.com/alexa/massive)、[adv-part-model](https://github.com/chawins/adv-part-model)、[PartImageNetPP](https://github.com/LixiaoTHU/PartImageNetPP)。本次确认了公开说明/入口，没有安装依赖、下载 checkpoint 或验证运行；“提供代码”不等于“已复现”。例如 TaskDiffusion README 仍指定较旧的 Python/PyTorch 组合，集成前应先单独做环境与数据 smoke test。

## 8. 调研范围、阅读深度与尚未核实的事项

### 8.1 检索范围

本次围绕以下关键词族向 2022–2026 年研究扩展，同时保留必要的老基准：multi-task dense prediction / task interaction、frozen / post-hoc / two-stage task transfer、auxiliary negative transfer / scalarization / asymmetric transfer、concept bottleneck、part-based recognition，以及对应的多标签/结构数据集。来源优先使用会议论文集、作者论文、作者仓库、机构数据页。第三方摘要只用于找线索，没有作为最终技术判断的主要依据。

这是一份面向本项目决策的专题调研，不是带完整纳入排除流程的系统综述，也不是所有截至当日论文的穷尽清单。

### 8.2 阅读深度

| 层级 | 本次覆盖 | 能支持的结论 |
|---|---|---|
| 本地原文/实现/报告 | 当前论文全部 sections；相关 feature cache/GNN；旧材料由 subagent 回查产物；主代理复读 E2 probe | 当前研究问题、输入协议、历史结果的使用边界 |
| 正文重点段落与方法细节 | ScaLearn；PCBM；part-based models；若干 CV 正文可检索段落 | 训练阶段、模块输入输出与对照的主要区别；不等于全篇结果复现 |
| 论文条目、摘要、作者 README | 表中其余工作及数据集入口；部分论文另取得正文片段 | 研究方向、发表信息、公开资产和已明确描述的协议 |
| 暂未完成核验 | PIN++ 2026 扩展的新增实验；两篇 2024 预印本的后续发表；所有资产下载完整性与具体运行成本 | 不作新增结果或可运行性承诺 |

初轮检索后段网络服务连续返回连接错误；补充访问 arXiv 的浏览器自动审批也两次超时，没有形成通过或拒绝的安全判断。因此初轮保留已取得的一手材料，对尚未补核的细节明确降级。第 10 节的后续检索已取得新增一手材料，其阅读边界另列于该节末尾。报告中的超链接指向证据或官方入口，不表示全部文件已在本机下载。

### 8.3 实施前需要完成的窄范围核查

- 固定选中数据集的版本、过滤规则、实际 split 数量和许可文件；核对 images、labels、checkpoint 是否可获取。
- 检查预训练/公开 checkpoint 与 A/B/test 的样本重叠。仅把数据重新拆分，无法使已经见过它的 checkpoint 恢复独立性。
- 固定 auxiliary truth 的语义：类别级或实例级、人工或派生、缺失或负例、连续目标的单位。
- 对最终准备写入论文的核心引用再次核对完整正文及 BibTeX。尤其不能把预印本年份、在线发表年份与正式会议年份混用。

## 9. 最终建议

下一步最有价值的路线是：**在现有 JetSet 上补齐容量与关系控制，同时选择一个全局/局部结构清楚的外部数据集，复现同一 frozen-readout 比较。** 首选 MASSIVE；需要 CV 直接衔接时增加 NYUD-v2，需要全局分类与规模研究时转向 PartImageNet/PIN++。

针对新增的 AI for Science 方向，优先级改为 **QM8 跑通回归协议、QM7-X 检验局部辅助机制**；希望优先依托近期已发表的多任务 Transformer 时，采用 **MP 固定快照＋CrystalTransformer**。这些是不同领域的候选路线，首轮无需全部实施。

论文的解释应从“辅助任务是否有尚未耗尽的独有信息”收敛到更可验证的三个问题：原头是否充分利用全局表示；全局池化是否隐藏局部结构；预测辅助关系是否在匹配输入和容量后仍降低学习难度。跨领域结果与这些控制实验结合，才有机会把当前观察推进为可靠的方法论。

## 10. AI for Science 补充：数据规模、已发表架构与小规模检验

补充日期：2026-10-02。本节回应“增加 AI for Science 数据集，并优先提供已发表的多任务 Transformer”的要求。**首轮建议 QM8 或 Tox21 跑通协议，QM7-X 检验局部辅助机制，Materials Project＋CrystalTransformer 连接近期已发表的多任务研究。** 以下子集大小、任务配对和缩小后的网络均为本项目的实验建议，不是原论文已经报告的结果。

### 10.1 数据集总览：全量规模与实际建议使用量

“样本量”按分子、晶体、蛋白质或模拟轨迹计数；构象、残基、时间窗口是下一级观测，不能直接当作相互独立的样本。下表的“首轮规模”包含拟选实验池；实际训练量还需扣除验证、测试，并拆成 A/B。已提供官方测试集的，保留该测试集。

| 数据集 / 领域 | 公开规模与标签 | 推荐主任务 ← 辅助任务 | 首轮规模建议 | 架构与定位 |
|---|---|---|---|---|
| **QM8 / 量子化学** | **21,786 分子**；MoleculeNet 口径为 12 个激发态性质任务；部分文件/loader 有 16 列，需核实计算方法和重复命名 | 第一激发能 E1-CC2 ← 第二激发能、振子强度；另测不同理论级别的同一性质 | 全量或 10k 分子 | **MTL-BERT 小模型移植**；最快的完整多目标回归检验 |
| **QM9 / 量子化学** | 原始 **133,885 分子**；常用 12 项性质，具体 loader 的目标数和过滤后 N 有差别 | HOMO ← 偶极矩模长、极化率、热容等预定性质 | 20k，随后 50k | **TorchMD-Net ET＋多头改造**；便于 N/容量扫描 |
| **QM7-X / 量子化学** | **6,950 个母分子、41,537 个平衡结构、4,195,237 个总结构**；42 项结构/物性条目，包含原子级标签 | 分子 HOMO ← 原子 Hirshfeld 电荷、原子极化率 | 先仅平衡结构；按母分子选 2k–4k，保留其平衡构象，必要时设每组上限 | **小型距离感知 Transformer＋全局/原子头**；最贴近局部辅助机制 |
| **QMugs / 药物分子量子化学** | **665,911 分子、1,992,984 构象**；分子/原子/原子对性质，含 GFN2-xTB 与 DFT 两种精度 | DFT HOMO ← 原子电荷；后续加入预测量子化学键级 | 10k–20k 分子，每分子先取 1 个构象 | **ET 或距离感知 Transformer＋原子/对头改造**；结构机制的扩展 |
| **Tox21 / 生物活性与毒理筛选** | MoleculeNet 原论文 **8,014 分子、12 端点**；PyG 当前文档对应 **7,831 图**，有缺失标签 | 一个预定 assay 端点 ← 其余端点 | 全量 | **MTL-BERT Small**；低成本多任务分类 |
| **ToxCast / 生物活性筛选** | MoleculeNet 口径 **8,615 分子、617 端点**；有效 N 随清洗变化，标签稀疏 | 一个预定端点 ← 先选 10–30 个覆盖充分的端点 | 全量分子、少量端点 | **MTL-BERT Small**；任务相关性/负迁移扩展 |
| **Materials Project 固定快照 / 晶体材料** | CrystalTransformer 使用的 MP：**69,239 晶体**（2018-06-01）；MP*：**134,243**（2023-06-23） | PBE 带隙 ← 形成能；再加入磁性性质 | 从同一快照选 10k–20k，后续 50k | **CrystalTransformer / ct-UAE，Nature Communications 2025**，已有明确多任务训练 |
| **JARVIS-DFT / 晶体材料** | 固定 **dft_3d_2021：55,723 条**；当前官方索引 dft_3d 为 **75,993 条**；各性质可用量不相同 | OptB88vdW 带隙 ← 形成能；弹性等另作缺失标签扩展 | 10k–20k，先核实主辅共同有标签的数量 | **Matformer，NeurIPS 2022＋多头改造**；周期几何更明确 |
| **NetSurfP-3.0 使用的数据 / 蛋白质结构特征** | **10,337 训练蛋白＋500 验证蛋白**；TS115：115；CB513：513 个区域、来自 434 蛋白；CASP12：21 蛋白 | 残基 SS8 二级结构 ← 相对溶剂可及性 RSA、无序、二面角 | 从官方训练集取 2k–5k 蛋白；测试保留 | **NetSurfP-3.0，NAR 2022** 是预训练 Transformer＋多任务网络；小型纯 Transformer 为改造方案 |
| **rMD17 / 分子动力学** | **10 种分子**，每种约 **100k 构象**；azobenzene 为 **99,988**；能量＋原子力 | 能量 ← 原子力 | 先 aspirin/ethanol 中一种；A＋B 总训练预算 **≤1,000 构象** | **TorchMD-Net ET，ICLR 2022**；作为导数监督对照，优先级较低 |
| **PDEBench 1D 可压缩流体 / 科学计算** | 论文单配置 **10,000 轨迹**，1,024 空间格点、100 时间步、3 个场；发布文件时间轴需另查 | 未来密度场 ← 同一未来时刻速度、压力场 | 单一周期边界配置，1k–2k 轨迹，空间降至 128/256 点 | **MPP 的 AViT，NeurIPS 2024** 提供已发表的多物理 Transformer；此处为缩小后的多场适配 |

规模的一手依据：[MoleculeNet Table 1](https://doi.org/10.1039/C7SC02664A)、[PyG 数据统计](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.MoleculeNet.html)、[QM7-X Table 1–2](https://www.nature.com/articles/s41597-021-00812-2)、[QMugs Table 1–2](https://www.nature.com/articles/s41597-022-01390-7)、[CrystalTransformer 数据与方法](https://www.nature.com/articles/s41467-025-56481-x)、[JARVIS 版本索引](https://atomgptlab.github.io/jarvis/databases/)、[NetSurfP-3.0 Datasets](https://pmc.ncbi.nlm.nih.gov/articles/PMC9252760/)、[rMD17 原始发布](https://figshare.com/articles/dataset/Revised_MD17_dataset_rMD17_/12672038)、[PDEBench 正文](https://papers.neurips.cc/paper_files/paper/2022/file/0a9747136d411fb83f0cf81820d44afb-Paper-Datasets_and_Benchmarks.pdf)及[数据说明](https://proceedings.neurips.cc/paper_files/paper/2022/file/0a9747136d411fb83f0cf81820d44afb-Supplemental-Datasets_and_Benchmarks.pdf)。

**下载体积与训练规模分开看：**

- QM7-X 官方八个压缩 HDF5 分片合计约 **9.6 GB**；提取平衡结构并不会自动省掉原分片下载。公开页还提供重复分子清单，清洗应一并处理。[Zenodo 数据页](https://zenodo.org/records/4288677)
- QMugs 含结构与物性的 **structures.tar.gz 约 7 GB**；全套波函数等数据解压后超过 **7 TB**。上述首轮方案只需结构/物性文件。[QMugs 原文 Data Records](https://pmc.ncbi.nlm.nih.gov/articles/PMC9174255/)
- rMD17 的完整压缩包约 **1,016.9 MiB**。样本高度相关是限制训练规模的原因，磁盘体积并非主要问题。[作者归档](https://archive.materialscloud.org/records/pfffs-fff86)
- PDEBench 官方下载说明中整个 1D CFD 类别约 **88 GB**；官方 Hugging Face 列表有单个黏性周期边界文件约 **12.4 GB**，不同发布入口/配置体积不同。首轮应选一个文件，避免默认下载整个类别。[下载说明](https://github.com/pdebench/PDEBench/blob/main/pdebench/data_download/README.md)、[文件列表](https://huggingface.co/datasets/pdebench/1D-Compressible-Navier-Stokes/tree/main)
- 其余数据的确切压缩体积本次未逐一下载核验；不把论文 PDF、预训练权重或 feature cache 的体积当作原始数据体积。

### 10.2 哪些架构已经发表并真正做了多任务

| 架构 | 已发表内容与证据 | 用于本项目时需要改什么 |
|---|---|---|
| **MTL-BERT，Research 2022** | 共享 SMILES BERT，同时微调多个性质任务；论文并非仅在许多数据集上分别训练。作者分类脚本提供 **3 层、d=128、2 heads** 的 Small 配置；回归脚本的 Small 为 **3 层、d=128、4 heads** | 采用小配置、统一分子级拆分、增加 A/B 冻结协议；迁到 QM8/QM9 是我们提出的实验，不声称论文已验证此组合 |
| **CrystalTransformer / ct-UAE，Nature Communications 2025** | 明确训练形成能＋带隙，并扩展总能量、总磁矩；作者仓库提供 2/3/4 任务脚本，还研究冻结原子嵌入后用于 CGCNN | 对同一固定数据池重新建立不重叠 A/B/Y；缓存完整上游 H、g、辅助预测后做本项目的读出比较 |
| **NetSurfP-3.0，Nucleic Acids Research 2022** | 使用 ESM-1b Transformer 表示，联合预测残基结构性质；主体采用 CNN/LSTM 多任务网络，正文也测试两层、8 heads 的 Transformer 下游替代 | 完整模型是混合架构；缩小纯 Transformer 是本项目改造。不能把原文探索过的 Transformer 替代结构写成其最终最佳网络 |
| **TorchMD-Net Equivariant Transformer，ICLR 2022** | 能量与原子力联合监督；力由能量对坐标求导，属于物理耦合的多目标训练 | QM7-X/QMugs 的电荷、原子极化率、键级头需新增；QM9 多物性同时训练也需另行实现，不能把原文单性质结果视为该改造的成绩 |
| **MPP / Axial ViT，NeurIPS 2024** | 单一 Transformer 处理多种物理系统，沿空间/时间轴组织 attention，并处理不同物理场 | 原文是多物理预训练；我们先在单一 1D 方程上联合预测三个场，再冻结做单个主场 refinement |
| **Matformer，NeurIPS 2022** | 已发表的周期图 Transformer，提供 JARVIS 基准与代码 | 本次未确认原仓库已有与此处一致的联合多头实验；按“可靠骨干＋拟议多任务改造”使用 |
| **CheMLT-F，Journal of Cheminformatics 2026** | 更新的多任务 Transformer 线索：SMILES/蛋白双编码器，部分冻结，共享多端点；论文报告约 **96M 总参数、34M 可训练参数** | 更适合作近期相关文献。其预训练使用约 58M 分子，不作为首轮小规模复现起点 |

来源与入口：[MTL-BERT 发表记录](https://pubmed.ncbi.nlm.nih.gov/39285949/)、[论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC11404312/)、[分类配置](https://github.com/zhang-xuan1314/MTL-BERT/blob/main/classification.py)、[回归配置](https://github.com/zhang-xuan1314/MTL-BERT/blob/main/regression.py)；[ct-UAE 作者仓库](https://github.com/fduabinitio/ct-UAE)；[NetSurfP 原文](https://doi.org/10.1093/nar/gkac439)、[作者代码](https://github.com/Eryk96/NetSurfP-3.0)；[TorchMD-Net 论文](https://arxiv.org/abs/2202.02541)、[代码](https://github.com/torchmd/torchmd-net)；[MPP 会议正文](https://proceedings.neurips.cc/paper_files/paper/2024/file/d7cb9db5ade2db7814fbd01ee59f4c7b-Paper-Conference.pdf)、[代码](https://github.com/PolymathicAI/multiple_physics_pretraining)；[Matformer](https://github.com/YKQ98/Matformer)；[CheMLT-F 原文](https://link.springer.com/article/10.1186/s13321-026-01199-1)。

**对现有文献叙事的新增价值：**CrystalTransformer 同时包含多任务上游和冻结嵌入迁移，值得加入核心阅读清单。但其 ct-UAE 主要是可迁移的元素嵌入参数，不能等同于每个输入晶体的完整末层表示；也没有因此证明“显式辅助预测比同一上游的 embedding-only 更好”。原文 MP* 预训练与 MP 下游来自不同年份快照，我们需要另查身份重叠，不能直接沿用公开权重来构造严格独立的 A/B/Y。这里比较的是协议差异，并未据此断言原论文存在测试泄漏。[原文方法与冻结实验](https://www.nature.com/articles/s41467-025-56481-x)

### 10.3 QM8 / QM9：成本较低的多目标回归起点

**QM8 更适合先跑通。** 其样本数在两万级，容易覆盖完整学习曲线。建议预先指定 E1-CC2 为主任务，先用 E2-CC2、f1-CC2、f2-CC2 作辅助；再单独增加其他计算方法的 E1，区分“不同性质的协助”与“同一性质跨计算精度的协助”。只输入分子结构/SMILES；B/Y 上用于 refinement 的辅助值全部由冻结模型预测。

对 QM8 的目标表需要实际核对：[DeepChem 当前源码](https://github.com/deepchem/deepchem/blob/master/deepchem/molnet/load_function/qm8_datasets.py)的任务列表有 16 项，并出现重复的 PBE0 名称，而 MoleculeNet 表列 12 任务。应对照原始计算方法、基组、CSV 列和 loader 输出逐项建立映射，不能凭列名把它们计成 16 个独立科学任务，也不能未经核验就删除可能代表不同计算设置的列。振子强度为无量纲量，不能和激发能直接混算原单位平均 MAE。

**模型建议：**第一轮采用 MTL-BERT 风格的 3 层、d=128、FFN=256 共享编码器，4 个 attention heads，独立小回归头。按标准层结构估计约 0.4–0.6M 参数，最终以实际实现计数；这是缩小配置的预算估计。已有 SMILES 输入足够建立流程，若要研究几何再增加原子距离编码版本。无需先复现大语料预训练。

**QM9 负责规模与任务组合复核。** 取 20k 分子，在相同测试集上增加到 50k；可用小型 TorchMD-Net ET 骨干加多个全局读出头。固定 HOMO 为主任务，辅助从偶极矩模长、极化率、热容等选 2–3 项；不把 LUMO 与 gap 同时作为 HOMO 的主要辅助，因为它们存在直接代数关系。也不要用 U0、U、H、G 之间的强物理关联作为唯一成功例子。标准 QM9 不提供与 track origin 对应的丰富逐原子监督，因此其结论是“全局性质间的任务交互”。

回归损失先按 A-train 的各任务均值/标准差归一化；最终分别报告原单位 MAE。可报告跨任务标准化指标作补充，但主结论应由预先指定主性质决定。[QM8/QM9 数据定义](https://doi.org/10.1039/C7SC02664A)、[QM9 原始论文](https://www.nature.com/articles/sdata201422)

### 10.4 QM7-X：本项目最值得优先做的 AI for Science 机制实验

**推荐配对：分子 HOMO 为主任务，逐原子 Hirshfeld 电荷和原子极化率为辅助。** 数据原文明确给出 HOMO 的 eH、电荷的 hCHG、原子极化率的 atPOL，以及 atNUM/atXYZ 等输入字段。它提供真实计算得到的原子环境标签，而不必人工把原子类型重建包装成新的物理监督。[原文 Table 2](https://www.nature.com/articles/s41597-021-00812-2)、[作者机构保存的正文](https://publications.uni.lu/bitstream/10993/46074/1/Hoja21.pdf)

**首轮样本组织：**先只提取平衡结构，按母分子身份拆分，再在训练池中取 2k–4k 个母分子；不要从 4.2M 结构中逐行随机切分。一个母分子的平衡构象及其所有扰动结构必须同组。随后增加每个母分子的少量非平衡构象，才能区分“更多独立分子”和“同一分子更多构象”的效应。重复分子清单也应参与分组。

**推荐小模型：**

- 原子种类嵌入，pair distance 的径向基函数编码作为 attention bias；padding mask；4 层、d=64 或 128、4 heads、FFN=2d。
- 主头读取 attention/mean pooled 表示 g；两个逐原子 MLP 分别预测电荷、原子极化率。首轮只用标量标签，先降低旋转等变实现的复杂度。
- A 上联合训练后冻结。R0 读取 g；R2 加入预测原子性质的分布统计；R3/R4 使用相同的小型集合读出器，分别读取 H 与 H＋预测原子性质。
- 原子任务先在每个分子内平均，再对分子平均，防止大分子仅因原子多而主导损失；标准化规则也要记录是否按元素区分。

这里采用距离的设计借鉴分子 attention 的既有思路；“该尺寸＋这几个任务＋A/B 协议”是我们提出的方案，并非已发表的同名多任务模型。[Molecule Attention Transformer 原文](https://arxiv.org/abs/2002.08264)

**两个有判别力的控制：**其一，在同一分子同一元素内打乱预测电荷/极化率与 H 的对应，并重新训练 refiner；这比跨所有元素任意打乱更能隔离局部环境对应关系。其二，给主任务头更多容量，检查改善是否只是原头过弱。电荷受总电荷约束，仅取均值可能几乎没有信号，应同时考虑方差、分位数或局部特征读出。若统计需要元素身份/数量，控制模型应具有同样访问权限。

HOMO 与这两个局部标签是否产生稳定正迁移仍需实测；“具有物理含义”不自动意味着有效辅助。如果只有 truth/oracle 有收益而 predicted 没有，应继续检查辅助预测质量及读出对误差的适应能力，不能据此认定部署方案已经有效。

### 10.5 QMugs：同时检验局部性质与预测关系

QMugs 比 QM7-X 分子更大，适合第二轮。首轮选 10k–20k 个规范化分子身份，每个只取一个构象，再扩展到所有构象；同一分子的不同构象、不同计算精度记录必须保持同组。ChEMBL ID 之外还应核对规范化结构身份，避免同分子异 ID 跨组。

**建议按难度分两步：**

1. 主任务 DFT HOMO；辅助为原子电荷。固定一种电荷定义，不混合 Mulliken、Löwdin、Hirshfeld 标签。
2. 增加 **预测量子化学键级** 的 pair head。可参考数据中的 Wiberg–Löwdin 等原子对物性，做 H-only、真实预测 pair、打乱 pair 三种冻结读出。[QMugs 属性表](https://pmc.ncbi.nlm.nih.gov/articles/PMC9174255/)

原子对头可对每个无向对使用对称特征，如 h_i＋h_j、|h_i−h_j|、h_i⊙h_j，再预测连续键级。若加入距离，H-only 对照也需要同样的距离和邻接信息。明确预测的是量子化学键级，输入已有的普通化学键类型不能当成独立辅助真值。全 pair 损失还应防止大量接近零的远距离对压倒有效关系。

**模型：**沿用 QM7-X 的距离感知 Transformer，或缩小 TorchMD-Net ET 后增加原子/对头；先 4 层、d=128，按实际参数和显存调整。首轮将原子数较多的分子放到后续规模实验，并记录筛选后 N、元素范围与分布。

GFN2-xTB 与 DFT 同一性质可形成有意义的跨精度对照，但如果在 B/Y 直接输入真实 xTB 计算值，就增加了外部计算信息；它应单列为另一种应用协议。本项目主比较应保持“辅助值由 A 上训练的模型预测”。第一轮也没有必要下载密度矩阵或波函数。

### 10.6 Tox21 / ToxCast：小样本分类与负迁移

**Tox21 是最低成本的分类首选。** 共享 SMILES Transformer，12 个 sigmoid 输出；预先指定一个端点为主任务，另外 11 个作辅助。可以先以 SR-p53 为候选，启动前仅依据训练部分的有效标签和阳性数决定它是否适合作主任务；一旦锁定，不再按测试集 refinement 增益更换主任务。

缺失标签使用 masked BCE，不能填成阴性。每个任务先按有效样本数归一化，再组合任务损失。主指标建议预先指定 PR-AUC 的具体实现，同时报告 ROC-AUC；记录各 split 的有效标签/阳性数，并按分子组评估不确定性。SMILES 枚举只在划分后进行，同一分子的所有字符串表示保持同组。

**模型优先沿用 MTL-BERT Small 的共享编码器＋多头结构。** 第一轮从随机初始化开始，以减少公开预训练数据重叠和额外资源混杂；因此应称“采用 MTL-BERT 结构的小模型实验”，不声称复现了原论文完整预训练成绩。后续如用公开权重，U0/U1 共享同一初始权重和预训练访问范围。[论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC11404312/)、[Small 配置](https://github.com/zhang-xuan1314/MTL-BERT/blob/main/classification.py)

作者分类脚本目前可见的流程包括对各任务数据表分别拆分、回归列先做整表标准化；这不适合直接照搬到我们的严格协议。应先按规范化分子身份合并所有端点，再统一划分 A/B/validation/test，归一化仅拟合训练部分。这里是代码片段的协议核查，没有运行其项目，也没有据此重判论文结果。

**ToxCast 是任务选择的扩展。** 首轮只取 10–30 个标签覆盖充分的端点，阈值只由训练集确定；再研究增加任务数量、主辅相关性与 refinement 增益。高度稀疏的 617 任务一起训练会把“关系结构”与“缺失模式/数据量差异”混在一起，不宜作为第一步。[MoleculeNet](https://doi.org/10.1039/C7SC02664A)

这两个数据集主要提供分子级端点，适合检验全局辅助读出、任务正负迁移和类别不平衡；它们不能替代 QM7-X 的局部机制实验。

### 10.7 Materials Project / JARVIS：近期已发表多任务 Transformer 的材料路线

**如果“已有多任务 Transformer 论文”是首要条件，选 MP 固定快照＋CrystalTransformer。** 从作者公开的 2-task 脚本出发，以带隙为主、形成能为辅助；缩小宽度/层数到预算允许的档位，保留一档原配置作后续确认。首轮取 10k–20k 晶体，按材料 ID、规范结构及必要的化学体系分组，避免重复结构跨 A/B/Y。[作者数据入口与多任务脚本](https://github.com/fduabinitio/ct-UAE)

建议首先缓存该 Transformer 的**完整上游输出 H、池化/原读出表示 g、辅助性质预测**。只冻结和迁移元素嵌入表回答的是另一个问题，不足以替代本项目的 frozen-upstream 机制比较。

主辅选择需要注意：

- PBE 带隙＋形成能是清楚的起点；若主任务改成形成能，就不要同时把同精度总能量当作唯一关键辅助，因为元素参考能与组成会提供接近直接的换算关系。
- 金属/非金属标签若由带隙阈值得来，也属于主标签的派生监督，应单列。
- 总磁矩可作为后续第三任务，但其缺失、零值占比及任务尺度应先检查。不要假设增加物理任务一定有益。
- 原论文前端借助数据增强处理对称性。缩小实现需要检查原子排序、旋转/平移等变换的一致性，不能因使用 Transformer 就默认满足晶体对称性。

**JARVIS 的优点是固定版本入口明确。** 采用 dft_3d_2021 可减少持续更新的影响；先选择形成能与 OptB88vdW 带隙，记录二者共同有效的 N，再考虑 mBJ、弹性模量等缺失更多的字段。Matformer 是适合周期结构的已发表骨干，给共享表示增加两三个独立回归头就是可控的改造；此处不把它标成原论文已有的多任务结果。[官方版本表](https://atomgptlab.github.io/jarvis/databases/)、[Matformer 仓库](https://github.com/YKQ98/Matformer)

这一方向主要检验全局物性间的交互。若之后要加原子级磁矩/电荷，必须另查具体版本是否提供逐原子标签，不能从“材料数据库有磁性字段”推断已具备局部监督。

### 10.8 蛋白质：NetSurfP 数据与 ProteinGLUE 的取舍

**建议主任务 SS8，辅助 RSA＋无序或二面角。** 每条蛋白序列是独立对象，残基是局部预测位置。refiner 应输出每个残基的 SS8；这对应局部主/辅任务交互，不能沿用 jet-level 的单个全局分类头。

完整 NetSurfP-3.0 可作为已发表参考，其多任务训练集大小见总表。若坚持小规模纯 Transformer，建议 4–6 层、d=128、4 heads、FFN=256，共享序列编码器上接 SS8 分类、RSA 回归等残基头。该小模型为新设计；如果从头训练表现明显欠拟合，应先检查容量/训练充分性，再将 refinement 现象解释为读出机制。使用大型预训练蛋白模型则应单独记录权重规模、特征提取成本和预训练暴露范围。

二面角用 sin/cos 表示并处理缺失残基；loss 先按蛋白内有效残基平均。冻结后，使用相同容量的 token MLP 或一层局部 attention，对比 H-only 与 H＋预测 RSA/其他辅助。长序列分窗必须在蛋白分组之后，不能把同一蛋白不同窗口拆入 A/B/test。[NetSurfP 方法与数据](https://pmc.ncbi.nlm.nih.gov/articles/PMC9252760/)、[代码](https://github.com/Eryk96/NetSurfP-3.0)

**ProteinGLUE 是较方便的替代数据格式。** SS3/SS8/ASA/BUR 来源的序列集合为 **8,803 train、1,102 validation、1,102 test**，共 **11,007**；但原文明确指出 ASA/BUR 与 SS3/SS8 的拆分是分别采样的。联合训练前必须按序列身份重新统一所有任务划分，否则同一蛋白可能通过辅助任务进入另一侧。其“multi-task benchmark”也不自动表示原基线已经将全部下游标签联合训练。[ProteinGLUE 原文](https://pmc.ncbi.nlm.nih.gov/articles/PMC9512797/)

首轮避免 SS3 ← SS8，以及 ASA ← BUR/RSA 的直接派生配对作为核心证据：SS3 可由 SS8 合并，ASA/RSA 通过残基类型对应的最大面积换算，BUR 又来自可及性阈值。选 SS8＋RSA 更能检验不同结构性质的互动。PPI/EPI 并非所有这些蛋白都同时有标签，本次不建议拼成一个标签齐全的数据集。

### 10.9 rMD17：用于导数监督的边界实验

数据小模型容易训练，但**十万构象不等于十万独立分子**。作者明确提醒时间相关性，并建议训练规模不超过 1,000；本项目应把 A＋B 的训练总预算一起计入这一限制。可从 A=800、B=200 起步，验证/测试使用另留样本，并检查原轨迹索引与近邻相关性。官方五组 split 与五个训练初始化 seed 是两种不同重复，不应混算。[rMD17 作者说明](https://figshare.com/articles/dataset/Revised_MD17_dataset_rMD17_/12672038)

推荐 TorchMD-Net ET 作为已发表的能量/力基线，但其原子力来自 \(\hat F=-\nabla_R\hat E\)。这带来两个区别：

1. 力监督是对同一势能面的导数约束，不能把正迁移直接解释为通用辅助任务的“独有信息”。
2. 求导结果依赖表示关于输入的 Jacobian。若 H 仅指某个输入点的缓存 hidden values，就不能直接沿用第 4 节“辅助输出完全是 H 的函数”的假设；需扩展输入定义，或另用从完整等变节点表示直接预测力的头，并明确这是新改造。

如果以 refiner 改写能量，却继续使用旧上游的力，二者可能不再满足梯度一致性；因此首轮只做静态预测诊断，不据此声称得到可用于稳定动力学模拟的新势能模型。[TorchMD-Net 原文](https://arxiv.org/abs/2202.02541)

### 10.10 PDEBench：增加与化学/生物不同的科学领域

选一个 1D 可压缩 Navier–Stokes 配置，以历史若干帧的密度、速度、压力为共同输入，联合预测固定未来时刻的三个场。**主任务设为未来密度，辅助为同一未来时刻速度与压力。** refinement 只接收冻结上游的隐藏场和预测辅助场；不能输入真实未来辅助场。方程残差约束如要加入，应另作实验，以免和数据监督的辅助作用混淆。

MPP 的 AViT 提供已发表架构参考，但不必复现跨多套 PDE 的大型预训练。小规模版本建议：

- 取 1k–2k 条独立轨迹，统一空间降采样到 128 或 256；记录降采样方法，以免只是改变了问题难度。
- 用 4–8 点一组的 patch 编码，4 个 Transformer blocks、d=128、4 heads；空间/时间 attention 分开，输出三个独立场头。这是参考 MPP 的新小配置。
- 先固定预测时距；第一轮检验单个未来帧的 density error，再考虑多步 rollout。
- R0 是读取完整 frozen hidden field 的主场 decoder；R2 加预测速度/压力。保持相同空间分辨率、decoder 容量及原始输入访问权限。这里没有必要把空间信息压成一个 g 才做人为更弱的基线。

按整条模拟轨迹拆分 A/B/validation/test，同一轨迹的所有时间窗只能属于一组；改变黏性系数或边界条件是后续 OOD 实验，先与同分布 refinement 分开。报告按轨迹汇总的 relative L2 / nRMSE，并规定零范数处理方式。该任务适合验证多场互动，工程量高于 QM8/Tox21。[PDEBench](https://github.com/pdebench/PDEBench)、[MPP 论文](https://proceedings.neurips.cc/paper_files/paper/2024/file/d7cb9db5ade2db7814fbd01ee59f4c7b-Paper-Conference.pdf)

### 10.11 共用的小规模训练与冻结协议

以下是建议起点，不是调参完成的配置或运行速度承诺。

**模型与计算预算：**普通序列/距离 attention 模型优先约 **0.2–1M** 上游参数，refiner 约 **10k–100k**；ET 的向量通道、周期图与蛋白长序列需单独计数，不能只按 d 比较容量。两档读出容量足以开始。以单张 12–24 GB GPU 为设计目标，首个小批次测显存后调整 batch；本次未实际验证显存或训练用时。

**训练起点：**普通小 Transformer 可用 AdamW、学习率 3e−4、weight decay 1e−2、dropout 0.1、5% warmup，最多 100 epochs，依据主任务验证指标早停。预训练模型微调、ET 力训练不应机械沿用这些数值，应先采用其作者配置。回归使用标准化 MSE/Huber，最终报原单位 MAE；分类使用 masked BCE/CE。只小范围选择学习率和辅助权重，不立即加入多种复杂 MTL 优化器。

为避免增加辅助任务数量就自动增加总 loss 权重，起点可写为
\[
L=L_{\rm main}+\lambda_{\rm aux}\frac{1}{K}\sum_{k=1}^{K}L_{{\rm aux},k},
\]
其中每项已按其有效标签、分子/蛋白/轨迹单位归一化；缺失任务的 batch 归一规则需固定。先比较 \(\lambda_{\rm aux}\in\{0,0.1,1\}\)，只用验证集选择。归一化并不保证各任务梯度同量级，仍需记录每项拟合质量。

**A/B 与统计：**

1. 没有官方锁定 split 时，先按独立对象建立约 80% 开发池、10% validation、10% test；官方已有测试的保持测试边界。开发池再按 80/20 分 A/B；比例只是起点，不作为最佳分配结论。
2. 上游 U0/U1 只在 A 训练；冻结编码器与全部任务头后，B 上只使用主任务标签训练 refiner。辅助标准化、词表拟合和目标选择也记录使用了哪部分数据。
3. 先做 U0/U1、g-only 或任务适当的 H-only、aux-only、concat，以及全部 A∪B 原模型的 T 对照。局部任务进一步做相同容量的 H-only vs H＋aux，不应人为移除对主任务必需的空间/序列信息。
4. 每个样本允许的元素、坐标、键、序列、历史场信息在比较两侧保持一致。oracle 单列；真实低精度计算值单列；公开预训练另列。
5. 先 2–3 个 upstream seeds 筛选实现问题；确认阶段 5 个 upstream seeds，每个 2 个 downstream seeds 起步，按第 6 节先在 upstream 内平均。多构象/多残基/多窗口 bootstrap 按其母对象分组。
6. 首轮主结果用固定上游的 **R2−R0 或 R4−R3** 配对差；同时报告相对同总预算 T 的变化。二者分别回答“辅助读出有无增量”和“两阶段是否值得”。
7. 对最终候选，保留一个指纹/描述符＋简单模型或领域常用小模型参照，检查从头训练的 Transformer 是否明显欠拟合；该参照帮助判断实验有效性，不参与替换核心的同上游比较。

**缓存估算：**20k 分子、平均 30 原子、d=128 的 FP16 H 约 154 MB；10k 蛋白、平均 300 残基、d=128 约 768 MB，均只计算 H，不含索引、辅助预测、padding 和其他副本。这些是形状乘积的估算，不是下载实测。先缓存紧凑 H/g/预测标量，只有确有关系实验时再保存完整 pair 张量。

### 10.12 推荐的最小实施组合与核验边界

| 优先次序 | 实验 | 最主要的科学问题 | 第一版完成范围 |
|---|---|---|---|
| **1：快速验证** | **QM8＋小型 MTL-BERT 结构**；需要分类时换 **Tox21** | 多任务辅助预测能否帮助固定表示的主任务读出？ | 单一预定主任务、2–3 个 seed、两种读出容量 |
| **2：机制主实验** | **QM7-X 平衡结构＋全局/原子多头 Transformer** | 原子级物性是否在 H-only 控制之外提供有效的读出组织？ | HOMO＋两项局部辅助、母分子拆分、局部对应打乱 |
| **3：已发表架构路线** | **MP 固定快照＋CrystalTransformer 双任务** | 近期材料 MTL 架构中是否仍存在可重复的 post-refinement 增量？ | 10k–20k 晶体、带隙＋形成能、同总预算对照 |
| **4：跨领域确认** | **蛋白 SS8＋RSA** 或 **PDEBench 密度＋速度/压力** | 现象能否扩展到序列位置/物理场预测？ | 二选一，先不同时展开 |
| 后续 | QMugs、ToxCast、JARVIS、QM9 扩大规模 | 关系结构、任务数量、材料版本与数据量如何影响增益？ | 依据首轮结果选择 |
| 特殊对照 | rMD17 | 导数辅助和物理一致性是否改变结论？ | 单分子、≤1k 总训练构象、静态评估 |

**综合建议：如果只增加两项 AI for Science 实验，选择 QM8＋QM7-X；若更看重近期已有多任务 Transformer 的可引用实现，选择 MP/CrystalTransformer＋QM7-X。** 后者覆盖全局性质交互与局部物性辅助，和现有 JetSet 机制的联系更完整。这是基于任务结构与实施成本的建议，尚无本项目实测排名。

本次新增核验包括：数据论文的规模/标签表、官方版本或下载列表；CrystalTransformer 的方法与冻结实验及作者多任务脚本说明；MTL-BERT 分类/回归小配置与拆分片段；NetSurfP 多任务架构、MPP 的 AViT 方法段落。尚未完成下载后 schema/缺失率审计、公开 checkpoint 重叠审计、端到端运行和资源测量。部分网页直接访问受限时使用了一手来源可检索正文；因此“入口存在”和“当前环境能完整复现”仍应区分。

