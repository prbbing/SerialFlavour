# 跨领域 post-refinement 测试案例：数据集、已发表架构与训练协议

更新日期：2026-10-03。本文整理 CV、Science 与 NLP 候选，供下一阶段选择和复现。除已有 QM9 历史实验外，这些路线尚未在本项目训练或验证运行环境。公开实现的检查属于源代码核查，不代表已经成功复现论文。

本轮的约束是：**优先复用已发表的上游多任务架构，不为寻找正结果自行开发新的上游模型或辅助头。** 冻结特征导出、数据划分、模型选择，以及我们的下游读出仍需适配。本文把论文事实、代码现状、本项目建议分别说明；不同论文的数字不能跨数据版本、类别映射和训练预算直接比较。

## 1. 案例总览与筛选原则

| 领域 / 数据集                         | 对应上游架构                                                            | 本项目拟定主任务             | 辅助任务与输出                                                  | 证据和接入状态                                                                          |
| ------------------------------------- | ----------------------------------------------------------------------- | ---------------------------- | --------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| CV / NYUv2                            | MTAN，CVPR 2019                                                         | 语义分割                     | 深度图、表面法向图                                              | 已发表三任务组合，独立密集输出；适合先做最小闭环                                        |
| CV / NYUv2                            | MTI-Net，ECCV 2020                                                      | 语义分割                     | 深度；已发表扩展还加入法向、边缘的多尺度预测                    | 有单任务与辅助配置消融；需区分最终输出与中间辅助输出                                    |
| CV / PASCAL-Context                   | MTI-Net，ECCV 2020                                                      | 语义分割                     | 人体部件、显著性、边缘、法向                                    | 已发表五任务组合，密集辅助读出丰富；部分监督是教师生成标签                              |
| Science / Materials Project（MP）     | CrystalTransformer / ct-UAE，Nature Communications 2025                 | PBE 带隙                     | 形成能；三、四任务版本再加入总能、总磁矩                        | 已发表多性质头；论文的多任务收益主要是元素嵌入迁移收益，完整冻结协议是本项目建议        |
| Science / QM9                         | 现有 TinySchNet；另考察 Equiformer＋EMPP、DeepMoleNet                   | HOMO–LUMO gap                | 现有电荷、键类型；文献中的位置重建或 ACSF 重建                  | 现有组合是本项目实现且历史结果为负；另外两个已发表方案均有直接接入限制                  |
| Science / rMD17                       | TorchMD-Net 的 Equivariant Transformer（ET），ICLR 2022                 | 构型能量                     | 各原子的力，来自能量对坐标的负梯度                              | 已发表能量—力联合架构；将该架构用于本文 rMD17 协议是推荐，属于导数监督边界案例          |
| Science / ADMET                       | MTGL-ADMET，iScience 2023                                               | CYP2C9 抑制分类              | CYP2D6 抑制、呼吸毒性、Caco-2 渗透性、血浆蛋白结合率（PPB）     | 作者已有五任务示例及输出头；需重建分子级划分，不能直接继承公开 CSV 分区                 |
| NLP / MASSIVE                         | 官方 XLM-R 并行双头，ACL 2023                                           | 句子意图分类                 | token 槽位预测                                                  | 已发表架构及作者代码；没有匹配 intent-only 消融，正迁移待本项目验证                     |
| NLP / SNIPS                           | 并行 Joint BERT（2019 预印本）；或 Stack-Propagation，EMNLP-IJCNLP 2019 | 句子意图分类                 | token 槽位预测                                                  | 前者有联合/分别训练消融，但常用代码为第三方；后者有作者代码，但辅助输出已受意图预测影响 |
| NLP / SemEval Restaurant14 / Laptop14 | RACL，ACL 2020                                                          | aspect term extraction（AE） | opinion extraction（OE）、aspect sentiment classification（SC） | 已发表三任务架构；AE 作为唯一主任务是本项目建议，关系消融不是 AE-only 正迁移证明        |

“辅助监督先改善 upstream，才优先投入 post-refinement”可以作为资源筛选规则，但目前是研究猜想，不是已证明的必要条件。至少应分别检验：

1. 相同 A 数据和预算下，辅助监督是否改善上游主任务；
2. 冻结后的主任务重新读出是否超过 native；
3. 辅助预测是否进一步超过容量和输入层级可比的 embedding-only。

选入正式重点路线前，在 A 验证上检查正迁移及稳定性；最终效应由锁定 Y 报告。已经完成的负结果、预先列出的候选失败及停止原因也应保留，避免只报告成功案例。

## 2. 所有领域共同采用的训练协议

以下是**本项目推荐协议**，不是任何一篇论文原有的二阶段实验。

### 2.1 A / B / Y 的角色

| 分区    | 用途                                    | 可以使用的标签                                                     | 禁止的用途                                 |
| ------- | --------------------------------------- | ------------------------------------------------------------------ | ------------------------------------------ |
| A_train | 训练 ST 与 MT 上游                      | ST 仅主任务；MT 使用主任务和预定辅助监督                           | 接触 B/Y 样本的其他任务标签进行上游训练    |
| A_val   | 上游 checkpoint、辅助权重及任务配置选择 | 按预先约定评估主任务；辅助指标仅作诊断                             | 用最终 Y 调辅助组合                        |
| B_train | 上游全部冻结后训练主任务读出            | 仅主任务标签；输入辅助量必须由冻结模型预测                         | 用辅助真值训练或替换预测输入，继续更新上游 |
| B_val   | 读出早停与超参数选择                    | 主任务标签                                                         | 与 A_val 共用样本后宣称两阶段选择独立      |
| Y       | 最终独立评估                            | 主任务用于最终评分；辅助真值可在方案锁定后评分或作隔离 oracle 诊断 | 训练、任务筛选、归一化拟合、反复选择模型   |

划分必须先于跨任务样本合并和训练，按图像/场景、分子、构型或文本来源身份执行。冻结包括编码器、任务头、任务交互模块及归一化状态；使用 eval 模式关闭 dropout 并固定 BatchNorm 统计。外部 ImageNet 等预训练可作为另行声明的公共初始条件，ST/MT 必须相同；无法排除 B/Y 暴露的领域预训练权重不用于严格实验。

上游优化加权主/辅助损失；下游只优化主任务损失。每任务损失按有效标签/像素数量归约，缺失标签不能填零当负样本。上游目标标准化只由 A_train 拟合；下游输入标准化只由 B_train 拟合。所有逆变换、物理单位和无效值掩码须固定并记录。

### 2.2 需要保留的对照

| 对照                             | 作用                                                               |
| -------------------------------- | ------------------------------------------------------------------ |
| ST-native                        | 主任务监督训练的真正单任务上游；不能以 MT 的 embedding-only 代替   |
| MT-native                        | 已发表多任务上游自己的主任务输出                                   |
| ST / MT embedding-only           | 检查冻结表征质量与读出训练的收益；使用相同 B 和相同读出预算        |
| MT embedding＋aux-pred           | 本项目 post-refinement；辅助输入只能是冻结预测                     |
| 容量匹配 embedding-only          | 排除增加通道或参数造成的收益；相同读出类型、深度和搜索预算         |
| 局部/多尺度 embedding-only       | 排除把局部辅助图与仅全局池化表示比较造成的输入层级优势             |
| 辅助分支 hidden-feature 对照     | 区分任务专用隐藏表示与显式辅助预测的作用；已有任务交互模型尤其需要 |
| native 初始化读出（架构兼容时）  | 保留 native 作为 epoch 0 可选候选，区分优化失败与输入价值          |
| shuffle / oracle（第二阶段诊断） | 打乱辅助预测或使用辅助真值探查机制；不与标准方法混作部署结果       |

ST 首先复用论文的单任务基线，并对齐 backbone、预训练和主任务 head。MT 的交互模块可能带来额外容量，必须报告参数和计算差异；必要时另做“相同 MT 架构但关闭辅助损失”的监督消融。这种同结构消融不自动等于论文的 ST 模型。

若辅助输出是同一冻结表示的确定性函数，它并未增加原始输入的信息。可能的价值是对有限容量、有限 B 样本的读出提供更易学习的任务表述。不能仅凭读出收益断言增加了信息或发现了不可恢复的新物理量。rMD17 的力还涉及输入坐标的导数，须单独讨论。

### 2.3 NLP 序列输出的共同约定

三条 NLP 路线均只读取文本；不扩展到语音识别。共享编码器输出记为 H（有效 token × hidden dimension），辅助分布记为 P。冻结缓存同时保存原句身份、token/word 对齐、有效长度、文本产生的 mask、主任务原预测和辅助预测；不能把辅助真值、真实 aspect 位置或带标签标记的输入文本保存为模型输入。缓存身份应包含上游 checkpoint、代码版本、tokenizer 和标签映射。

优先使用连续 logits 或完整概率分布；原代码若返回 log-probability，须明确转换或保留其数值含义。PAD、CLS/SEP 及 WordPiece 的非首子词如何处理必须固定：词级评价使用相同对齐规则，读出只聚合有效 token。公开标签 schema 可以预先固定；由数据估计的词表、类别权重、归一化及特征选择只使用相应训练分区。不能把 BIO 标签数、原始槽位类型数和任务头数量混为一谈。

主任务为句子意图时，强 embedding-only 必须拥有完整 H，可使用逐 token 投影、masked pooling 与原模型已有类型的分类头；辅助增强组用相同序列处理方式读取 H＋P。主任务为 AE 时，两组都必须保留逐 token 的位置与上下文，并使用相同的序列预测头；全局池化向量不构成合格的 AE 基线。若加入主任务 native logits，两组共同加入；任务专用 hidden-feature 则另列对照。

三条路线先使用原有输出模块及其配置选项，不新增上游 pair head、rationale head 或其他辅助任务。parallel 描述前向连接，joint 描述联合优化；二者可以同时成立。并行头不读取对方预测，但共享表示会受两类梯度影响，并不意味着统计独立。Stack-Propagation 和 RACL 已有任务间连接，必须单独记录其影响；冻结后的 B 训练不是这些论文原有的联合训练。

## 3. CV：NYUv2 与 MTAN

### 3.1 数据集和任务

NYU Depth V2 是室内 RGB-D 数据。原始资源含 464 个场景、1,449 个密集标注帧，以及大量未标注视频帧。多任务基准常用 795 个训练图像和 654 个测试图像。主实验只使用这一标注子集，不默认为获得更多数据加入原始视频帧。[数据集主页](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html)、[MTAN 论文](https://arxiv.org/pdf/1803.10704)

模型输入是 RGB。深度是监督目标，不能作为“辅助输入真值”送进模型。采用 MTAN 已发表三任务组合：

| 本项目角色 | 任务          | 输出                       | 评价                                 |
| ---------- | ------------- | -------------------------- | ------------------------------------ |
| 主任务     | 13 类语义分割 | 每像素 13 类 logits / 概率 | 数据集级 mIoU，越高越好              |
| 辅助任务   | 深度估计      | 每像素一个连续值           | AbsErr / RelErr 等，固定有效深度掩码 |
| 辅助任务   | 表面法向估计  | 每像素三维单位向量         | 平均/中位角误差等                    |

法向属于派生几何监督，需固定标签生成版本；不是完全独立采集的一套物理观测。各任务采用共同裁剪与空间增强；翻转时法向分量也要正确变换。

### 3.2 已发表架构及多任务收益

MTAN（Multi-Task Attention Network）在共享 SegNet 编码—解码网络上，为各任务设置注意力模块，从共享特征中选择任务相关通道/空间内容，再接任务预测头。公开图像到图像代码直接输出分割、深度、法向；预测头采用卷积，法向在输出端归一化。推理不要求真实标签。[CVPR 2019 论文](https://openaccess.thecvf.com/content_CVPR_2019/html/Liu_End-To-End_Multi-Task_Learning_With_Attention_CVPR_2019_paper.html)、[作者模型代码](https://github.com/lorenmt/mtan/blob/master/im2im_pred/model_segnet_mtan.py)

原论文 Table 3 的等权 MTAN 相对单任务 attention 基线 STAN：

| 指标                |   STAN | MTAN（等权） |
| ------------------- | -----: | -----------: |
| 历史语义分割 mIoU ↑ |  15.73 |        17.72 |
| 深度绝对误差 ↓      | 0.6935 |       0.5906 |
| 法向平均角误差 ↓    |  32.09 |        31.44 |

这些是该论文的历史评价协议。作者 README 后来说明 mIoU 改为基于整个数据集的混淆矩阵计算，并修正无效像素及增加增强；因此不能把当前代码的 mIoU 与表中数字直接比较。重新复现 STAN 与 MTAN 时必须使用同一版指标。论文总多任务收益也不能当成分割单一主任务的收益。[论文 Table 3](https://arxiv.org/pdf/1803.10704)、[代码维护说明](https://github.com/lorenmt/mtan)

### 3.3 原设置与本项目训练建议

作者原模型脚本使用 Adam、学习率 1e-4、batch size 2、200 epochs，以及每 100 epochs 衰减为 0.5 的 StepLR；提供等权、uncertainty weighting、DWA。第一轮建议固定等权，不同时开展损失加权搜索。[训练脚本](https://github.com/lorenmt/mtan/blob/master/im2im_pred/model_segnet_mtan.py)

我们把官方 795 张训练图像再划分为 A_train / A_val / B_train / B_val，654 张官方测试图像锁定为 Y。建议目标比例为训练池的 60% / 10% / 20% / 10%；若随机按图像划分，对应约 477 / 80 / 159 / 79。**正式协议优先按 scene 分组，实际数量因此可偏离这些目标，须报告最终 manifest。**

冻结后缓存共享解码特征、任务注意力特征及深度/法向预测。第一轮读出沿用原模型的卷积预测头类型：embedding-only 与 embedding＋预测图保留相同空间分辨率，并匹配训练参数量；不把空间辅助图的效果仅与一个全局向量 MLP 比较。语义 logits 可以另作共同输入，两组必须一致；辅助法向保持连续向量，辅助深度保留连续预测。

## 4. CV：NYUv2 / PASCAL-Context 与 MTI-Net

### 4.1 已发表架构和可导出的辅助读出

MTI-Net（Multi-Scale Task Interaction Network）使用 HRNet 的多尺度特征，在各尺度先生成任务特征和初始预测，再通过任务间 distillation 与跨尺度 feature propagation 交互，最后输出主任务结果。交互主要发生在**任务隐藏特征**上，不能简化为把其他任务预测图拼接进主任务。[ECCV 2020 论文](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123490511.pdf)

作者代码保留四尺度的 deep_supervision 字典，包含任务特征和预测。NYUv2 的法向/边缘可能仅存在于这些中间预测，不能假定 forward 的最终字典中存在对应全分辨率输出。我们可以直接导出已实现的中间头，固定尺度后用于读出；无需新增上游辅助任务头。[模型代码](https://github.com/SimonVandenhende/Multi-Task-Learning-PyTorch/blob/master/models/mti_net.py)

### 4.2 NYUv2：任务和辅助监督消融

MTI-Net 的 NYUv2 基准以分割和深度为输出任务；已发表扩展加入法向 N 和边缘 E。对我们的单一主任务检验，分割为主，其余监督按辅助处理。作者配置采用 **40 类**语义分割，不能与 MTAN 的 13 类结果直接比较。[任务类别配置](https://github.com/SimonVandenhende/Multi-Task-Learning-PyTorch/blob/master/utils/config.py)

论文 Table 2a：

| 模型 / 任务设置     | 分割 mIoU ↑ | 深度 RMSE ↓ |
| ------------------- | ----------: | ----------: |
| ST 基线             |       33.18 |       0.667 |
| 普通 MTL            |       32.09 |       0.668 |
| MTI-Net，分割＋深度 |       35.12 |       0.620 |
| MTI-Net，增加 N     |       36.22 |       0.600 |
| MTI-Net，增加 N＋E  |       37.49 |       0.607 |

这里有两层证据：完整方案比分割 ST 提高 4.31 个 mIoU 点；在 MTI-Net 内加入 N＋E 比基础双任务提高 2.37 点。但最后一项比只加 N 的深度结果略差，辅助任务并非对所有指标单调有益。论文 +10.91% 是跨分割/深度的聚合指标，不是分割提升 10.91%。[论文 Table 2a](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123490511.pdf)

第一轮建议先复现分割＋深度，再验证已发表 N、N＋E 消融。沿用现有任务模块，通过配置启用这些已发表任务；辅助配置只在 A_val 选择。NYUv2 的 A/B/Y 和场景分组沿用第 3 节，各类别版本分别维护 manifest。

### 4.3 PASCAL-Context：数据与任务

这里指 MTI-Net 采用的 **PASCAL-Context 多任务版本**，包含自然图像及不同来源的任务标签。它不是把任意 PASCAL VOC 下载包直接当成五任务完整数据。类别映射使用作者代码：语义分割 21 类，人体部件输出 7 类（含背景）。不能误用常见 PASCAL-Context 59/60 类语义分割配置。[数据加载器](https://github.com/SimonVandenhende/Multi-Task-Learning-PyTorch/blob/master/data/pascal_context.py)、[类别配置](https://github.com/SimonVandenhende/Multi-Task-Learning-PyTorch/blob/master/utils/config.py)

| 本项目角色 | 任务         | 输出与来源                                      | 辅助评价               |
| ---------- | ------------ | ----------------------------------------------- | ---------------------- |
| 主任务     | 语义分割     | 21 通道；作者数据版本中的语义标签               | mIoU                   |
| 辅助任务   | 人体部件分割 | 7 通道；人体部件标注，存在无人体/无有效标签图像 | mIoU，严格有效区域掩码 |
| 辅助任务   | 显著性       | 单通道预测；论文版本采用教师生成监督            | 采用原版显著性指标     |
| 辅助任务   | 边缘         | 单通道预测；由该数据版本的边界监督构造          | 原版边缘评价           |
| 辅助任务   | 表面法向     | 三通道预测；论文版本采用教师蒸馏标签            | 角误差                 |

显著性和法向不应称为原始人工真值；必须记录教师、生成来源、标签有效范围，以及教师是否可能看过 Y。监督类型本身可以研究，但无法核验教师暴露时要明确该条件，而不是宣称所有训练资源与 Y 严格独立。[论文数据说明](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123490511.pdf)

论文 Table 2b 的五任务 MTI-Net(a) 相对 ST，分割 mIoU 从 60.07 提升到 64.27，增加 4.20 点；跨任务聚合增益为 +2.74%。法向平均角误差却由 14.59 变为 14.75，略有劣化。这提供了已发表主任务正收益，同时说明多任务整体收益不能替代逐任务结果。[论文 Table 2b](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123490511.pdf)

### 4.4 训练设置与冻结读出

作者公开 HRNet-W18 NYUv2 基础配置为 ImageNet 预训练、Adam、学习率 1e-4、weight decay 1e-4、batch size 8、100 epochs、poly scheduler；其基础配置只有分割和深度。PASCAL 配置启用五任务，损失权重按分割/部件/显著性/边缘/法向为 1/2/5/50/10。以上是所核查公开配置，不能当成所有论文消融都采用的唯一超参数。[NYUv2 配置](https://github.com/SimonVandenhende/Multi-Task-Learning-PyTorch/blob/master/configs/nyud/hrnet18/mti_net.yml)、[PASCAL 配置](https://github.com/SimonVandenhende/Multi-Task-Learning-PyTorch/blob/master/configs/pascal/hrnet18/mti_net.yml)

本项目建议优先 W18 而非扩大 backbone。保持原多尺度交互和任务头，仅改变数据边界及按主任务 A_val 选择 checkpoint。PASCAL 将作者训练池按 60/10/20/10 分成 A_train / A_val / B_train / B_val；作者验证分区锁定为本项目 Y，不能继续让原训练脚本每轮用它选模型。确认图像 ID、重复图像和标签覆盖后记录具体数量。

冻结后分别缓存 backbone 多尺度特征、任务交互后的主任务特征、已有辅助头的预测与任务隐藏特征。至少对照“同尺度特征＋辅助预测”与“同尺度特征”，并另对照辅助 hidden features。已有 MTI-Net native 已进行任务交互，post-refinement 能否再获益需要实测，不能由论文的端到端收益直接推出。

## 5. Science：Materials Project 与 CrystalTransformer

### 5.1 数据版本、目标与已发表任务组合

Materials Project 是持续更新的计算材料数据库，不是固定版本的数据文件。CrystalTransformer 论文使用 MP（2018-06-01，69,239 个材料，60,000/5,000/4,239 划分）与 MP*（2023-06-23，134,243 个材料，80/10/10 划分）。前端在 MP* 训练，再将元素嵌入迁移到 MP 上的后端模型。[论文数据与划分](https://www.nature.com/articles/s41467-025-56481-x)

| 已发表设置      | 主任务（本项目选择） | 辅助监督           | 输出形态       |
| --------------- | -------------------- | ------------------ | -------------- |
| MT@2p，优先     | PBE 带隙 Eg，eV      | 形成能 Ef，eV/atom | 两个材料级标量 |
| MT@3p，后续消融 | Eg                   | Ef、总能 E         | 三个材料级标量 |
| MT@4p，后续消融 | Eg                   | Ef、E、总磁矩 M    | 四个材料级标量 |

该带隙是计算标签，不能直接称为实验基本带隙。总能、磁矩需核对快照字段、每晶胞/每原子的约定及单位。若改以形成能为主任务，总能与组成参考能存在强代数关联，不建议把该配对作为核心独立辅助任务证据。

### 5.2 架构与增益的准确解释

CrystalTransformer 将原子种类 one-hot 与坐标分别线性嵌入、拼接，使用 Transformer encoder，并由首 token 表示接材料性质头。公开两任务脚本使用宽度 256、8 层、8 个 attention heads、前馈宽度 512；两个 head 各输出一个标量。周期边界处理、原子顺序及旋转行为需要按数据加载器和增强设置核查。[论文 Methods](https://www.nature.com/articles/s41467-025-56481-x)、[两任务模型](https://github.com/fduabinitio/ct-UAE/blob/main/ct/model_mt_2.py)、[训练脚本](https://github.com/fduabinitio/ct-UAE/blob/main/train/main_mt_2.py)

论文多任务元素嵌入迁移给 CGCNN 后，MT@2p 的 MAE 为形成能 0.068 eV/atom、带隙 0.357 eV，相对单任务嵌入分别改善约 4% 与 0.5%；加入磁矩后带隙 MAE 为 0.367 eV，出现负迁移。这不是完整 CrystalTransformer native 的同预算 ST/MT 消融，也不是冻结所有前端后再输入辅助预测的实验。[论文 Table 2](https://www.nature.com/articles/s41467-025-56481-x)

### 5.3 本项目推荐协议

我们推荐复用现有 MT@2p，主任务 Eg、辅助 Ef。尽管只有一个辅助头，仍适合作为跨性质案例，不要求每个领域都人为增加多个辅助任务。严格路线从 A 训练完整 ST/MT CrystalTransformer，冻结全部层和 head，导出逐原子表示 H、首 token 表示 g、预测 Eg 与 Ef；B 仅用 Eg 标签训练读出。**这一完整冻结适配是本项目建议，文献尚未证明它的收益。** 单独复制元素嵌入表给一个重新训练的 CGCNN 是另一种迁移协议，应另设实验名。

作者两任务脚本当前默认 SGD、学习率 0.006、momentum 0.9、weight decay 0.001、batch size 128、500 epochs、每 20 epochs 的 StepLR（gamma 0.5），并对两个标准化目标计算等权 MSE。这些默认值作为复现起点；目标标准化及选择规则必须改为 A_train / A_val 边界，不能沿用全数据采样统计。[公开训练脚本](https://github.com/fduabinitio/ct-UAE/blob/main/train/main_mt_2.py)

建议先取一个固定快照、通过质量筛选后的 20k 材料试验池，按 56/5/14/5/20 分成 A_train / A_val / B_train / B_val / Y，即 11,200 / 1,000 / 2,800 / 1,000 / 4,000。按 material ID 和等价结构去重分组；必要时增加组成分组测试作为单独泛化协议。记录 CIF、晶胞规范、原子顺序和坐标单位；首 token 的选择使原子排序尤其需要审计。不要为修补对称性临时开发新架构；发现缺陷先评估作者处理方法是否满足用途，再决定是否保留候选。

正式实验不能直接加载在更大 MP* 上训练的权重而忽略它与 B/Y 的材料重叠。作者 ct-UAEv1.0 可作为版本核对来源；严格实验应锁定代码 commit 与数据快照，独立训练。材料级辅助输出只能支持性质间读出检验，不能写成有逐原子量监督的局部任务案例。[作者仓库](https://github.com/fduabinitio/ct-UAE)、[论文归档版本](https://doi.org/10.5281/zenodo.14557908)

## 6. Science：QM9 的保留案例与已发表备选

### 6.1 数据和已有项目任务

QM9 是约 134k 个小有机分子的平衡构型与量子化学性质数据，包含 H/C/N/O/F、最多 9 个重原子。具体样本数需经过未表征分子剔除和本项目解析筛选确定。它与 rMD17 的非平衡构型和力监督不同。[原始数据及说明](https://figshare.com/articles/dataset/Quantum_chemistry_structures_and_properties_of_134_kilo_molecules/978904)

目前主任务是 gap，单位 eV；辅助任务是原子 Mulliken 电荷与 RDKit 推断的无序 pair 键类型。后者是离散化学图标签，不是量子化学计算的连续键级。SchNet 是已发表 backbone，但我们 TinySchNet＋电荷/键头的完整组合是**本项目实现**，不能称为已发表多任务模型。

历史 full 实验 ST-native MAE 为 0.098634 eV，MT-native 为 0.103823 eV，MT 约差 5.26%；辅助任务并未改善 upstream。代码修正与新的 v2 配置尚无 100k 正式结果。历史结论及其边界见 [QM9 full 结果](qm9/qm9_gap_charge_bond_full_results_zh.md)。这一案例可保留为负例，不应为了满足筛选原则把它重新包装成正迁移模型。

### 6.2 已发表候选及接入限制

| 候选                        | 已发表辅助任务与证据                                                          | 为什么不直接列为当前核心复现                                                                                                            |
| --------------------------- | ----------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| Equiformer＋EMPP，ICLR 2025 | 位置遮蔽后的径向/方向分布重建；gap MAE：引用基线 30 meV，1-Mask 27，3-Mask 26 | 辅助分支训练依赖真实目标性质条件；清洁输入评估时关闭该分支。B/Y 不能输入真实 gap，改用预测条件或遮蔽视图导出均需额外适配                |
| DeepMoleNet，JCIM 2021      | 多性质预测＋径向/角向 ACSF 重建；作者稿消融 gap 35.4→33.2 meV                 | 该结果来自多性质模型；单目标结果 32.1 meV 更好。公开核心文件未提供可直接运行的完整网络；去掉其他性质并改成 gap＋ACSF 不等于已验证原方案 |

EMPP 的 3-Mask 指分别构造三个各遮蔽一个原子的视图；30 meV 基线来自此前 Equiformer 结果，仍需同划分重跑才能作为我们的辅助收益依据。[正式论文](https://proceedings.iclr.cc/paper_files/paper/2025/file/7ab7073a147f0a4ee5c76995800d8f14-Paper-Conference.pdf)、[训练/评估实现](https://github.com/ajy112/EMPP/blob/main/engine.py)

DeepMoleNet 发表信息与上述数值分别见 [JCIM 论文](https://pubs.acs.org/doi/10.1021/acs.jcim.0c01224)、[作者公开稿 Tables 3、7](https://chemrxiv.org/engage/api-gateway/chemrxiv/assets/orp/resource/item/60c755aeee301c5046c7b1f7/original/transferable-multi-level-attention-neural-network-for-accurate-prediction-of-quantum-chemistry-properties-via-multi-task-learning.pdf)。[公开核心文件](https://github.com/Frank-LIU-520/DeepMoleNet/blob/master/DeepMoleNet.py)的现状不支持把仓库描述为完整可复现实装。

因此，在“保持 gap 主任务、复用完整公开多任务架构、不自行开发新辅助头”的约束下，**当前核查的 QM9 候选尚没有满足全部要求的优先方案**。这只是本轮候选核查结论，不是证明不存在任何此类工作。禁止把 HOMO/LUMO 加入 gap 辅助输入后以其差值恢复主目标，再视作一般性的 post-refinement 收益。

### 6.3 现有路线的训练协议

若继续审计历史负例，复用现有 [v2 配置](../config/qm9_gap_charge_bond_refine_v2_100k.json)：A_train 56,000、A_val 5,000、B_train 14,000、B_val 5,000、Y 20,000；独立验证，固定样本池。上游 5 seeds，每个上游对应 5 个下游 seeds。

当前配置的 backbone 为 hidden 64、3 个 interaction blocks、32 个 RBF、5 Å cutoff；上游最多 100 epochs、batch 128、学习率 1e-3、weight decay 1e-5，以 gap 验证 MAE 选模型，patience 20。下游最多 300 epochs、patience 10；MLP 和 set/graph 读出的学习率分别为 1e-3 与 3e-4，native 初始化另有配置。该预算属于本项目，不是 SchNet 原论文复现。

保持电荷单位、pair 对称性、完整键类型概率及同容量读出对照；不因为更换领域而丢弃这些严谨性要求。此处仅登记路线，本文没有启动新训练或改动配置。

## 7. Science：rMD17 与 TorchMD-Net ET

### 7.1 数据与任务关系

rMD17 对 MD17 构型重新进行更高精度的能量/力计算，含 10 个分子；通常每个分子约 100k 构型，azobenzene 为 99,988。采用 PBE/def2-SVP、严格收敛设置，提供坐标、核电荷、能量、力和原轨迹索引。能量单位为 kcal/mol，力为 kcal/(mol·Å)。官方资源还提供五套训练/测试划分，并因轨迹时间相关建议不超过 1,000 个训练点。[rMD17 数据与使用说明](https://figshare.com/articles/dataset/Revised_MD17_dataset_rMD17_/12672038)

本项目推荐能量为主任务，原子力为辅助监督。首轮每个分子独立训练，优先选 ethanol 或 aspirin；不要把跨分子泛化与单分子构型拟合混成一个目标。

### 7.2 架构与证据边界

TorchMD-Net ET 使用旋转等变注意力，维护标量/向量原子表示，预测标量能量。力由

\[
\widehat{\mathbf F}_i=-\frac{\partial\widehat E}{\partial\mathbf r_i}
\]

得到，**不是另一套独立力 head**。它支持能量—力联合训练，适合检验导数监督，但不等价于多性质独立辅助 readout。[ICLR 2022 论文](https://arxiv.org/pdf/2202.02541)、[ET 模型文档](https://torchmd-net.readthedocs.io/en/2.4.4/models.html)

原论文的该组结果在原始 MD17 上评估，使用 950 训练、50 验证点，剩余为测试；不能改标为 rMD17 成绩。原能量/力权重为 0.2/0.8，但它没有在我们所需的同架构 energy-only 对照下直接证明“辅助力改善能量”的效应大小。因此，**ET 是已发表架构，ET＋本文 rMD17 划分及 post-refinement 是本项目推荐，尚无相应收益实测。**[论文 MD17 实验与训练说明](https://arxiv.org/pdf/2202.02541)

原论文 MD17 设置使用 Adam（β1=0.9、β2=0.999、epsilon=1e-8）、初始学习率 1e-3、1,000 步 warm-up、batch size 8；验证损失 plateau 30 epochs 后将学习率乘 0.8，最低 1e-7，不使用 weight decay 和 dropout。完整 ET 为 6 层、128 维、32 个 RBF；本项目移到 rMD17 后仍需重新验证，按能量 A_val 选择模型的规则也属于我们的协议适配。[论文训练部分与 Appendix A / Table 4](https://arxiv.org/pdf/2202.02541)

### 7.3 推荐训练和缓存协议

在一套官方划分的 1,000 个训练点内部，建议 A_train / A_val / B_train / B_val = 600 / 100 / 200 / 100，官方测试分区锁定为 Y；不再额外抽取 Y 构型补充开发样本。根据原轨迹索引检查邻近相关构型，能分组时采用时间块与隔离间隔；若因此偏离官方随机协议，必须另标协议并报告实际数量。五套官方划分是 split 重复，不是五个初始化 seed。

保持作者已有 ET 模型及损失形式；先做 energy-only 与 energy＋force，以 A_val 能量 MAE 选择 checkpoint。原始 0.2/0.8 权重作为起点，不能在 Y 搜索。模型规模优先采用论文 Table 6 已给出的 3 层、80 维、16 个 RBF、约 273k 参数版本；不自行设计小网络。推荐上游起步上限 1,000 epochs，以 A_val 能量 MAE 早停，patience 100；该上限和早停是我们的建议，不是原论文设定。依赖及具体训练实现锁定官方版本。当前仓库对旧 ET 的维护状态已有变化，避免直接以最新默认模型替换已选 ET。[作者仓库](https://github.com/torchmd/torchmd-net)

冻结参数不意味着禁止输入梯度：B/Y 上生成预测力时仍需坐标 requires_grad，并启用坐标梯度，完成后 detach 缓存。对整个过程使用 inference_mode 或 no_grad 会破坏导数输出。力的生成成本也应计入辅助缓存和推理开销。 若标准化能量，导数与力的单位变换必须一致：能量减去常数不改变力，能量除以尺度则导数也除以该尺度，不能独立缩放后直接宣称能量—力一致。

缓存标量/向量表示、预测能量及预测力。能量读出须保持旋转不变；若采用力范数等不变量摘要，要明确这是**本项目推荐的下游输入适配**，不是 ET 原论文的 post-refinement。向量表示不能直接展平后用不具旋转一致性的 MLP 宣称保留原模型对称性。embedding-only 也应拥有相同局部/向量信息处理机会。

仅修正能量而保留原力，通常不能保证新的能量—力对仍满足上述梯度关系。本案例第一阶段只评估静态能量预测，不据此宣称可用于保守分子动力学。

## 8. Science：ADMET 与 MTGL-ADMET

### 8.1 数据和已发表五任务示例

ADMET 是吸收、分布、代谢、排泄、毒性性质的统称，不是一个统一版本的标准数据集。本文具体采用 MTGL-ADMET 论文汇集的 24 个端点：18 个分类、6 个回归，约 43,291 个不同化合物，标签稀疏，来自多个公开来源。[iScience 2023 论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC10654589/)

选择作者已有五任务示例，避免重新设计任务架构：

| 本项目角色 | 任务                 | 类型与输出         | 主/辅助评价                          |
| ---------- | -------------------- | ------------------ | ------------------------------------ |
| 主任务     | CYP2C9 inhibitor     | 二分类，一个 logit | ROC-AUC 为主；同时报告 PR-AUC        |
| 辅助任务   | CYP2D6 inhibitor     | 二分类，一个 logit | ROC-AUC / PR-AUC                     |
| 辅助任务   | Respiratory toxicity | 二分类，一个 logit | ROC-AUC / PR-AUC                     |
| 辅助任务   | Caco-2 permeability  | 回归，一个标量     | MAE / RMSE / R²，核对原标签尺度      |
| 辅助任务   | PPB                  | 回归，一个标量     | MAE / RMSE / R²，核对百分比/变换约定 |

预测时不需要其他端点真值。辅助任务名单已在作者示例提供，先固定该组合；不把 A 或 Y 上事后发现相关性的端点不断加入正式实验。[作者训练示例](https://github.com/dubingxue/MTGL-ADMET/blob/main/Experiments/Training.py)

### 8.2 架构和论文增益

MTGL-ADMET 使用共享 ResGCN 原子表示、任务专用 attention pooling、主任务中心的 gate 交互及各任务输出 tower。公开示例采用 64 维隐藏表示，并返回五任务预测。attention 权重是聚合系数，不是有独立原子级监督的局部辅助任务；task gate 使用隐藏特征，不能称为在预测值层面融合。[作者模型](https://github.com/dubingxue/MTGL-ADMET/blob/main/Experiments/model.py)

论文 Table 1 的 CYP2C9 ROC-AUC：ST-MGA 为 0.764±0.017，MTGL-ADMET 为 0.794±0.013；另一个多任务 MGA 为 0.798±0.019，略高于 MTGL。ST-MGA 与 MTGL 结构不同，0.030 的差不能全部归因于辅助监督。论文自身 Single/MT 消融报告整体约 3.16% 改善，但该整体值不是 CYP2C9 特定收益，更不是 post-refinement 收益。[论文 Table 1 与消融](https://pmc.ncbi.nlm.nih.gov/articles/PMC10654589/)

该模型提供明确的多个独立标量辅助头，适合我们的接口；不过其辅助输出仍是分子级量。因此它检验多个性质之间的读出，不应替代有逐像素/逐原子监督的局部任务案例。

### 8.3 公开数据的重叠审计

2026-10-03 对作者公开 CSV 做只读统计：共有 48,390 行、43,291 个不同原始 SMILES 字符串；CYP2C9 有标签的 test 分区为 640 行。按未规范化、未去空格的 SMILES 精确匹配：

| 检查范围                                | 与 CYP2C9 test 重叠的行数 |
| --------------------------------------- | ------------------------: |
| CSV 全部 training 行                    |                       101 |
| training 中至少有上述五任务之一标签的行 |                        21 |
| training 中有 CYP2C9 标签的行           |                         1 |

这是我们对**当前公开 CSV**的审计，不是对论文实际运行的数据暴露作定论；原训练预处理可能进一步筛选。它已足以说明不能直接继承 CSV 的 group 字段来建立严格归纳实验。尚未做 RDKit canonicalization，化学等价分子的重叠还需进一步核查。[被审计公开 CSV](https://github.com/dubingxue/MTGL-ADMET/blob/main/Data/admet.csv)

### 8.4 推荐训练协议

先进行有效分子解析、规范化身份和冲突标签审计，将同一分子的全部端点记录绑定，再统一划分 A/B/Y。缺失端点保留 mask，不能只保留五任务全标签分子。建议按独立分子组以 56/5/14/5/20 划分；兼顾主任务正负比例及辅助覆盖，最终数量以去重 manifest 为准。scaffold 分组可作为另一组泛化实验，不能与随机分子划分结果混合汇总。

作者示例使用 Adam、学习率 1e-3、weight decay 1e-5、batch 128、最多 200 epochs、patience 50；分类为带训练集正类权重的 BCEWithLogitsLoss，回归为 MSE，dropout 0.2。作者早停按多任务验证汇总评分，本项目改为 CYP2C9 的 A_val ROC-AUC，避免辅助指标替主任务选模型。[训练代码](https://github.com/dubingxue/MTGL-ADMET/blob/main/Experiments/Training.py)

在作者现有结构内关闭辅助监督，构成同结构主任务监督消融，同时保留原论文 ST-MGA 为结构不同的参考。ST/MT 应使用相同主任务标注样本与相同主任务更新预算；MT 额外使用 A 中辅助-only 分子的输入暴露和计算须单列，必要时另做仅主标签共有分子的监督对照。任何属于 B/Y 的分子，其其他性质标签也不能进入 A。

冻结后缓存共享原子表示、主任务 gate 后表示及四辅助输出。B 仅用 CYP2C9 标签；比较分子 embedding、embedding＋四辅助预测，以及等容量/其他任务 hidden-feature 输入。回归辅助输出的尺度由训练分区拟合，分类保留连续 logit/概率，不先硬阈值成类别。

公开示例依赖较旧的 DGL/PyTorch，存在五任务固定配置。第一轮使用作者同一任务组合并锁定依赖，不把“仓库存在”当成运行已验证；若必须更换主任务，再核查 gate 数和任务顺序是否支持，不自动开发新 gate 架构。

## 9. NLP：MASSIVE 与官方 XLM-R 并行双头

### 9.1 数据集、版本与任务

MASSIVE 面向个人助手的一次性指令理解，来自英语 SLURP 语句的多语言本地化。ACL 2023 论文使用 MASSIVE 1.0，含 51 种语言、18 个场景、60 个意图、55 种槽位类型；1.1 增加 Catalan，其他语言数据保持不变。它有约百万条跨语言记录，但同一源语句的本地化版本不是百万个独立语义样本。[正式论文](https://aclanthology.org/2023.acl-long.235/)、[数据与代码](https://github.com/alexa/massive)

| 本项目角色 | 任务                  | 输出与评价                                                          |
| ---------- | --------------------- | ------------------------------------------------------------------- |
| 主任务     | intent classification | 句子级 60 类 logits；主指标 accuracy，补报 macro-F1                 |
| 辅助任务   | slot filling          | 每个有效词/子词的槽位 logits；补报作者口径的 slot F1                |
| 联合诊断   | 完整语义 frame        | intent 与全部 slots 同时正确的 exact match；不替主任务选 checkpoint |

例如 “what is the temperature in new york” 的意图是 weather_query，局部槽位包括 weather_descriptor 与 place_name。槽位标签来自同一语句的实体/语义片段标注，不是把意图标签复制到各 token。55 是原始槽位类型数，最终分类头维度由作者标签编码决定。

首轮推荐固定 MASSIVE 1.0 的 en-US，随后才扩展其他语言。公开 Hugging Face 数据卡列出 en-US 的 train/dev/test 为 11,514 / 2,033 / 2,974，总计 16,521 条有标签记录；论文的约百万总量还计入另外保留的竞赛集合，不能把它当成首轮可用训练标签。实际导入以固定归档和 manifest 复核数量。[数据卡](https://huggingface.co/datasets/AmazonScience/massive/blob/main/README.md)

### 9.2 已发表架构、输出接口及收益证据

论文的 XLM-R Base（约 270M 参数）共享一个预训练 Transformer encoder，池化表示接意图分类头，序列表示接槽位分类头。作者尝试 first / mean / max 三种意图池化。两个头没有 intent→slot 或 slot→intent 预测连接，属于 **parallel joint model**；联合损失为意图损失加加权槽位损失。[论文 §5.1](https://aclanthology.org/2023.acl-long.235.pdf)

作者的 XLMRIntentClassSlotFill 已有完整双头实现；intent_num=None、slots_num=None 时仍返回 intent_logits、slot_logits，无需真实标签。导出 H 时读取同一次 encoder 前向输出，不能重新训练新的辅助头。**这是已发表架构及作者实现；下面的单语言冻结实验是我们的协议适配。**[模型源码](https://github.com/alexa/massive/blob/main/src/massive/models/xlmr_ic_sf.py)

论文主要比较不同 encoder、完整多语言训练和英语 zero-shot 设置，**没有提供匹配的 intent-only 单任务消融**。因此，不能把它的联合成绩当成“槽位监督已经改善意图主任务”的证据；本项目必须补做 ST/MT。单语言结果也不能直接与论文跨语言平均成绩比较。

### 9.3 原训练设置与本项目 A / B / Y 建议

论文每类模型进行了 128 次超参数搜索，使用 Adam，并按所有 locale 的联合 exact match 选择 checkpoint；这是多语言 benchmark 的设置。我们不继承这套昂贵搜索，也不沿用联合指标来选择意图主任务。[论文 §5.1、Table 5](https://aclanthology.org/2023.acl-long.235.pdf)

建议保留官方 test 2,974 条为 Y，在 train 内按源语句组以 75% / 25% 划分 A_train / B_train，在 dev 内以 50% / 50% 划分 A_val / B_val；目标数量约为 8,636 / 2,878 和 1,016 / 1,017。意图分层与组身份优先，实际数量允许偏离，必须保存 manifest。先审计跨官方分区的完全重复文本；若存在重复组，优先保护 Y，开发池剔除重叠记录并重新报告数量。

多语言扩展必须按官方 id（源 SLURP 语句 ID）绑定全部 locale，确保同一源语句不跨 A/B/Y；不能以 (id, locale) 为独立随机划分单位。输入只使用 utt，不使用包含真值槽位标记的 annot_utt，也不将 scenario / intent / judgments 当作文本特征。中文、日文预处理需复核作者字符分隔及词—子词对齐规则；它们不是首轮自动加入的语言。

起步配置是本项目建议：复用 XLM-R Base 与作者已有双头类，固定一种池化（先 mean），Adam、学习率 2e-5、有效 batch 32、最多 20 epochs、warmup 为更新步数的 10%、weight decay 0.01，按 A_val intent accuracy 早停，patience 5；相同指标时用意图交叉熵决定。槽位权重首轮固定 1；若搜索，只在 A_val 用事先限定的 {0.5, 1, 2}。这些数字不是论文最佳配置。

ST 使用相同基础预训练、tokenizer、意图头和训练样本，只优化意图损失；MT 加槽位损失。共享编码器在 A 中都进行 fine-tuning，B 中全部冻结。正式预算中同时记录实际更新步数、搜索次数和 GPU 时间。已在完整 MASSIVE 上微调的公开任务 checkpoint 接触过潜在 B/Y，不能替代 A 上重新训练；基础语言预训练暴露仍需声明，不能宣称已证明与所有文本完全无重叠。

### 9.4 冻结缓存和下游训练

缓存 H、原意图 logits 和槽位 logits / 概率，保留作者实际标签通道。B 仅用 intent 标签优化句子读出，使用 B_val intent accuracy 选择。推荐先沿用已有小型分类读出：逐 token 投影后 masked mean/max pooling，再接分类 MLP；embedding-only 和 H＋slot 采用相同 token 层级、池化、深度、搜索预算并匹配参数量。槽位分支 hidden-feature 另列对照。

下游最多 100 epochs、学习率候选 {1e-3, 3e-4}、patience 15 是推荐起点。B 样本有限，先验证线性/小型读出，再扩大容量；辅助标签不参与 B 训练或早停。正式测试同时报告 ST-native、MT-native、两种上游的 embedding-only，以及 MT embedding＋slot；只有最后一项超过强、同容量的 MT embedding-only，才支持冻结辅助输出的增量作用。

## 10. NLP：SNIPS 的 parallel joint 与 Stack-Propagation 两条路线

### 10.1 数据集和主辅任务

SNIPS 是个人助手指令理解数据，常用 train/dev/test 为 13,084 / 700 / 700，包含 7 种意图，例如播放音乐、预订餐馆、查询天气。使用已有论文和代码采用的词级槽位版本，锁定 tokenization 与 BIO schema；不要混用不同发布版本的槽位类型计数。[Joint BERT 数据说明](https://arxiv.org/pdf/1902.10909)、[Stack-Propagation 数据入口](https://github.com/LeePleased/StackPropagation-SLU)

| 本项目角色 | 任务              | 输出与评价                                 |
| ---------- | ----------------- | ------------------------------------------ |
| 主任务     | 句子意图分类      | 7 类预测；accuracy 为主，macro-F1 为补充   |
| 辅助任务   | 词级 slot filling | token 槽位分布；词级对齐后计算严格 span-F1 |
| 联合诊断   | semantic frame    | 意图及全部槽位同时正确的 sentence accuracy |

这是全局主任务＋局部辅助任务案例，但只有一个槽位辅助任务；多个槽位类别不等于多个辅助头。SNIPS 比 MASSIVE 标签空间更小，适合验证流程；主任务准确率接近上限，小测试集上的少量正确样本差异需要逐 seed 结果与不确定性支持。

### 10.2 Parallel joint：Joint BERT，及其发表/实现边界

Chen、Zhuo 和 Wang（2019）的 Joint BERT 使用共享 BERT：CLS 表示接意图 softmax，词的首个 WordPiece 表示接槽位 softmax；两个头不读取对方预测，联合优化两类损失。原文还有槽位 CRF 变体。**本次核实到的原文是 arXiv 预印本，正式同行评审发表状态未确认；不能标成已核实的会议论文。**[原文 §3.2](https://arxiv.org/pdf/1902.10909)

| SNIPS，原文 Table 3，30 epochs | No joint | Joint BERT |          增益 |
| ------------------------------ | -------: | ---------: | ------------: |
| intent accuracy ↑              |    98.0% |      98.6% | +0.6 个百分点 |
| slot F1 ↑                      |    95.8% |      97.0% | +1.2 个百分点 |

这是同一 BERT 方法联合与分别训练的消融线索，比跨不同 backbone 的比较更直接；表中未给多种子 SD，不能预设收益在我们的 A/B 划分上稳定。原文采用 uncased BERT-Base、最大长度 50、batch 128、Adam 学习率 5e-5、dropout 0.1；本项目改变数据分配后需重新按 A_val 主指标选择训练轮数。[原文 Tables 2、3 与 §4.2–4.4](https://arxiv.org/pdf/1902.10909)

常用 monologg/JointBERT 仓库明确是 **第三方实现**；源码使用 BERT pooled_output 产生 intent logits、sequence_output 产生 slot logits，标签参数均可设为 None。这满足冻结推理接口，但具体 pooling、损失归约和超参数需逐项核对，不能默认等同原文。首轮建议关闭 CRF，保留连续槽位概率；否则须区分 CRF emission、边缘概率和离散解码结果。[仓库说明](https://github.com/monologg/JointBERT)、[预测接口](https://github.com/monologg/JointBERT/blob/master/model/modeling_jointbert.py)

如果正式发表与作者实现是硬性条件，可把第 9 节的官方 XLM-R 并行双头用于 SNIPS。**它在 SNIPS 上属于本项目推荐适配，不是 MASSIVE 论文已验证的 SNIPS 方案**；只替换数据/schema，不新增上游头。Joint BERT 路线则保留为已有公开研究与第三方复现候选，不隐藏其发表状态。

### 10.3 Stack-Propagation：已发表的显式意图→槽位连接

Qin 等人的 Stack-Propagation 正式发表于 EMNLP-IJCNLP 2019。非 BERT 版本共享 BiLSTM＋self-attention encoder，先用意图 decoder 给每个 token 预测句子意图，再将该分布与对应 encoder 表示拼接给槽位 decoder；句子意图由 token 预测投票获得。两类损失联合优化，论文的可微连接允许槽位损失沿意图输出支路回传。[正式论文 §2.2–3](https://aclanthology.org/D19-1214/)

| SNIPS，原文 Table 3              | intent accuracy ↑ |
| -------------------------------- | ----------------: |
| lstm＋token-level，意图单任务    |             97.5% |
| 联合模型，without self-attention |             97.8% |
| 完整 Stack-Propagation           |             98.0% |

完整模型比意图单任务高 0.5 点，但同时改变 self-attention 等结构；这不能隔离槽位监督的全部贡献。保留真正 intent-only 与同结构关闭槽位损失的两级对照，区分表示、结构和辅助监督。[论文 Table 3](https://aclanthology.org/D19-1214.pdf)

作者 ModelManager.forward 在 n_predicts=None 时返回槽位与 token 意图的 log-probability；冻结 B/Y 时 forced_intent=None、forced_slot=None，禁止调用 golden_intent_predict_slot。需要从原 encoder 导出 H；不新增辅助头。**当前 CLI 的 differentiable 默认 False：默认使用离散 top-1 意图 embedding，启用该选项才走连续分布连接。** 因此配置必须记录该开关，不能把默认代码描述成已验证的论文可微复现。[模型源码](https://github.com/LeePleased/StackPropagation-SLU/blob/master/utils/module.py)、[CLI](https://github.com/LeePleased/StackPropagation-SLU/blob/master/train.py)

作者 CLI 起点为 Adam 1e-3、batch 16、上限 300 epochs、dropout 0.4、L2 1e-6，提供 teacher-forcing 配置。论文的 SNIPS word embedding 维度与仓库 README 示例不完全一致，应锁定配置并说明来源；不要直接把 CLI 默认值当作论文最终设置。首轮建议非 BERT 版本，以降低成本和避免近乎满分的主任务；teacher forcing 只在 A 使用，导出阶段全用自主预测，记录训练与推理条件变化。

### 10.4 两条路线的共同训练协议与比较边界

建议保留官方 test 700 条为 Y，将 train 按语句身份、意图分层以 75% / 25% 分给 A_train / B_train（目标 9,813 / 3,271），dev 700 条各分 350 条给 A_val / B_val。先审计重复文本及可用的来源/模板身份；同一重复组不得跨区，近似模板泛化可另作实验，不混入首轮随机组划分结果。词表只在 A_train 拟合，公开预训练 tokenizer 为固定公共条件。

Parallel 首轮复用 BERT-Base，学习率 5e-5、有效 batch 32、上限 30 epochs；Stack 首轮采用上述作者 CLI 起点并按 A_val 意图准确率早停，建议 patience 20。两者分别匹配 ST/MT 的更新预算、词表、初始化与选择规则。**不能把非 BERT Stack 与 BERT parallel 的性能差值解释为连接方式的因果效应。** 若要专门比较连接，必须匹配 backbone 和其余解码结构；该比较不属于第一轮冻结验证的必要步骤。

ST 不使用槽位监督；同结构监督消融保留原模块、关闭槽位损失。Stack 的 ST 训练也必须禁用槽位标签 teacher forcing，不能仅把槽位损失权重设为零却仍读取真实 slots。原论文简化的 intent-only 模型可作另一个结构对照，但两者分开报告。Stack 中开启/关闭可微路径、teacher forcing 或 decoder 连接属于已实现机制的配置/消融，需要明确标成我们的控制，不当作论文原成绩。

冻结后缓存完整 H、槽位分布和 token / sentence 意图原预测。B 仅用句子意图标签，读出和容量匹配遵循第 2.3、9.4 节；下游最多 100 epochs、patience 15。Stack 的槽位输出已受到主任务预测影响，主分析应让两组共同访问相同原意图 logits / token 意图分布，再测槽位增量；另报 H-only 对照，避免将对原意图预测的重新加工解释为独立槽位信息。

在 NLP 内，**parallel 是更清楚的首轮机制案例，Stack 是已有任务预测传递的扩展案例**。这是一项实验优先级建议，不表示 parallel 的性能必然更高，也不取消用户保留的两条路线。

## 11. NLP：SemEval Restaurant14 / Laptop14 与 RACL

### 11.1 数据集、标签来源与本项目主辅任务

Restaurant14（Res14）和 Laptop14（Lap14）分别是餐馆与笔记本评论的细粒度情感数据。原始 SemEval 提供 aspect term 及其情感；RACL 的 opinion 标签来自后续工作的补充标注。必须使用作者已发布的完整三任务预处理版本，并记录来源，不能把只有 aspect/sentiment 的原始下载包当成完整三任务数据。[ACL 2020 论文 §4.1](https://aclanthology.org/2020.acl-main.340.pdf)、[作者代码和数据](https://github.com/NLPWM-WHU/RACL)

| 数据版本，论文 Table 2 | 原训练池 | 官方 test |
| ---------------------- | -------: | --------: |
| Restaurant14           | 3,044 句 |    800 句 |
| Laptop14               | 3,048 句 |    800 句 |

首轮推荐 Res14，Lap14 作为另一领域复核，分别训练并报告，不直接合并为一个模型结果。论文还使用 Res15，但它不在本轮保留清单内。已有预处理 train/dev 文件合并回开发池时须审计重复，实际数量由 manifest 决定。

例如 “The food was delicious”：food 是 aspect，delicious 是 opinion，food 对应的 sentiment 为 positive。**本文把 AE 设为唯一主任务是本项目建议，原论文研究的是完整三任务 ABSA。**

| 本项目角色 | 任务                                | 已有输出头                                                                       | 评价                      |
| ---------- | ----------------------------------- | -------------------------------------------------------------------------------- | ------------------------- |
| 主任务     | aspect term extraction，AE          | 每词 B/I/O 三类 logits                                                           | 严格 aspect span-F1 为主  |
| 辅助任务 1 | opinion term extraction，OE         | 每词 B/I/O 三类 logits                                                           | opinion span-F1           |
| 辅助任务 2 | aspect sentiment classification，SC | 每词 positive / neutral / negative 三类 logits；训练仅有效 aspect 位置有情感监督 | SC-F1；仅作辅助诊断       |
| 联合诊断   | 完整 ABSA                           | aspect span 与情感都正确                                                         | ABSA-F1，不替代 AE 主指标 |

SC 不是句子整体情感三分类，也不是每个非 aspect 词都有情感真值。多词 aspect 的情感评价和 conflict 标签处理沿用作者口径；标签缺失不应解释为 neutral。

### 11.2 已发表架构、任务关系和冻结输出

RACL（Relation-Aware Collaborative Learning）正式发表于 ACL 2020。它用共享特征映射和任务专用 CNN 提取 aspect、opinion、context 表示，堆叠多层关系交互，协调 AE↔OE、SC 与 extraction/context 的关系，再输出三套 token 预测。它不是简单的并行独立双头；已有三任务模块均保留，冻结时包括关系传播和所有预测层。[正式论文](https://aclanthology.org/2020.acl-main.340/)

作者提供 RACL-GloVe 与 RACL-BERT。前者拼接通用与领域词向量，后者用 BERT-Large；首轮建议 GloVe 版本控制计算成本，保持外部词向量来源相同，审计领域预训练语料的潜在 B/Y 暴露。项目需声明公共预训练，而不是保证这些语料绝无测试文本。

源码 RACL(inputs, position) 返回 aspect_prob、opinion_prob、sentiment_prob，但这些变量实际是归并后的 logits；之后才分别 softmax。冻结缓存应导出未做真值位置筛选的 OE、SC logits / 概率，以及共享和任务专用 H。[模型源码](https://github.com/NLPWM-WHU/RACL/blob/master/model.py)

**掩码核查的修正：**作者代码明确规定训练/验证的情感 mask 只在真实 aspect 有效，测试时 mask 为所有非 padding 词；utils.read_data(..., is_testing=True) 也实现了这一点。因此，不能把作者测试流程笼统说成用了真实 aspect 位置。我们的 B/Y 缓存仍应直接抓取掩码之前的 sentiment logits / senti_value，并使用文本长度 mask；不能误复用训练/验证读取模式，也不能用真实 AE span 截取情感特征。[数据读取代码](https://github.com/NLPWM-WHU/RACL/blob/master/utils.py)

SC 在非 aspect token 上缺少直接监督，其未掩码输出并不自动成为校准的词级情感概率。可以预先规定用冻结预测 AE 概率软加权 SC，另报不加权版本；权重规则只在开发分区选择，不能用 B/Y 的真实 AE 位置作为输入。

### 11.3 论文增益与能支持的结论

| Restaurant14，论文 Table 3 | AE-F1 ↑ | ABSA-F1 ↑ |
| -------------------------- | ------: | --------: |
| RACL-GloVe                 |   85.37 |     70.67 |
| RACL-BERT                  |   86.38 |     75.42 |

论文 Table 4 中，RACL-GloVe 删除各关系后，Res14 ABSA-F1 分别下降 0.98、1.91、1.76、1.86 点。这支持已发表关系交互对完整 ABSA 有用；**它不是同结构 AE-only 与 AE＋OE＋SC 的辅助监督对照**。不能用这组完整 ABSA 收益预先宣称我们的 AE 主任务有正迁移，也不能把 BERT 与 GloVe 的差值归因于多任务训练。[论文 Tables 3、4](https://aclanthology.org/2020.acl-main.340.pdf)

### 11.4 原训练设置与本项目 A / B / Y 建议

作者随机取训练池的 20% 作 dev，按最小 dev 总损失选择模型。RACL-GloVe 使用 Adam 1e-4、batch 8、正则系数 1e-5；Res14 / Lap14 的 CNN kernel 为 3，堆叠层数为 4 / 3。RACL-BERT 的学习率为 1e-5。论文原选择目标与本文 AE 主任务不同。[论文 §4.1](https://aclanthology.org/2020.acl-main.340.pdf)

我们推荐官方 test 为 Y，其余完整训练池按语句/来源组分成 A_train / A_val / B_train / B_val，目标 60% / 10% / 20% / 10%。Res14 约为 1,826 / 304 / 609 / 305，Lap14 约为 1,829 / 305 / 609 / 305。若能恢复原 review ID，同一 review 的句子应归为一个组；若作者预处理不保留该字段，至少绑定完全重复文本，并把无法做 review 分组作为限制记录，不能声称已控制完整评论相关性。

三套标注、token 对齐、conflict 与有效监督 mask 必须随同一语句划分。主任务训练样本在 ST/MT 中相同；MT 仅在 A 使用 OE/SC 标签。A_train 可以用真实 aspect 位置决定 SC 损失的有效项，因为这是训练监督；冻结推理和 B 输入不能继续用该真值掩码。

首轮推荐固定作者 GloVe 模型与 Res14 的 4 层配置，沿用词向量/隐藏维度 400、CNN 通道 256，Adam 1e-4、batch 8、最多 100 epochs、patience 15，按 A_val AE span-F1 早停；MT 优化 AE＋OE＋SC 交叉熵和作者关系正则，并记录损失归约。这里只复用论文已有模块；上限与主指标选择是本项目建议。学习率或层数不在 Y 调整，不照 README “增加层数获得更好结果”的提示开展开放搜索。

检验辅助监督需保留同结构 AE-only 监督消融：关闭 OE/SC 损失及依赖它们真值的正则，不读取辅助标签，保持 encoder/AE 输出模块相同。它的关系分支仍会受 AE 梯度间接训练，所以应准确标为“同结构单任务监督”，不是论文已经发布的独立 AE 基线。进一步的 AE-only 简化抽取器只可复用已有 shared/AE 模块并关闭跨任务连接，属于本项目推荐消融；若原代码不支持简单配置，先记录实现成本，不自行开发新的上游辅助头。

### 11.5 下游序列读出、环境与资源限制

冻结后缓存共享 token 表示、AE/OE/SC 任务专用隐藏表示、原 AE logits 和两套辅助 logits。主比较以同样拥有 AE 任务隐藏表示的强 embedding-only 为基础，排除辅助组只是多访问任务分支表示的优势；共享 H-only 单列。B 仅训练 AE 的 token 读出，以 B_val AE span-F1 选择，OE/SC 真值不参与损失或输入。

推荐沿用原 CNN/逐 token 分类模块类型，embedding-only 与 embedding＋OE＋SC 保持相同上下文窗口、层数和训练参数预算；原 AE logits 若用于软加权情感或 residual 输入，两组都能访问。下游最多 100 epochs、Adam 学习率候选 {1e-3, 3e-4}、patience 15 为起点，严格 BIO span 解码固定。两个辅助头分别加入和共同加入的消融在 B_val 预先选定，Y 一次报告；oracle 真值仅作隔离诊断。

训练池拆分后，B 只有约 600 句，可能不足以稳定训练复杂读出。第一轮限制读出容量并报告逐 seed 配对结果；若扩大 B 比例，应在运行前设定比例组并相应减少 A，而不是因 Y 结果修改划分。

作者 GloVe 环境为 Python 3.6.10 / TensorFlow-GPU 1.5，BERT 版本 README 要求 TensorFlow-GPU 1.12。旧环境可运行性尚未验证；先评估隔离环境与原版小批次，再决定是否实施。兼容性修补和缓存接口适配应记录，不把完整重写成新 PyTorch 上游视为已经复现。[环境说明](https://github.com/NLPWM-WHU/RACL)

## 12. 建议的实施次序、预算与成功判据

### 12.1 实施次序

1. **NYUv2＋MTAN**：三个已有密集预测头，先验证完整 A→冻结→B→Y 流程和空间输入对照。
2. **NYUv2＋MTI-Net，再 PASCAL＋MTI-Net**：检验更强的已发表任务交互与多尺度辅助输出；两种 NYUv2 类别版本分别报告。
3. **MP＋CrystalTransformer MT@2p、ADMET＋MTGL-ADMET**：前者需快照/结构身份与首 token 审计，后者需先修复实验分组；完成后可作为 Science 的多性质案例。
4. **rMD17＋ET**：独立列为导数监督与物理一致性边界案例，保留 README 已有的“路线延后”实施状态，本文不表示立即启动。
5. **QM9**：保留已有负例与 v2 审计；EMPP / DeepMoleNet 暂作文献备选，不自行新增上游架构来强行满足候选清单。

NLP 内建议先验证 **SNIPS parallel 或非 BERT Stack 的最小闭环**，再运行 **MASSIVE en-US＋官方 XLM-R**；要求多个辅助输出时增加 **Res14＋RACL**，Lap14 留作复核。若正式发表＋作者实现为硬条件，MASSIVE 的并行路线先于 Joint BERT 预印本路线。三条 NLP 路线是用户保留的候选，尚未选定具体实现或开始训练。

上述 CV/Science 次序和 NLP 内次序是本项目推荐，不表示已选定全部实施。可运行性、数据许可、主任务正迁移与资源预算在各案例小规模检查后再决定正式规模。

### 12.2 起步预算

| 案例                     | 数据起步范围                  | 上游训练起点                              | 下游建议                                                |
| ------------------------ | ----------------------------- | ----------------------------------------- | ------------------------------------------------------- |
| NYUv2 / MTAN             | 标注 795 训练池＋654 锁定测试 | 作者 200 epochs、Adam 1e-4、batch 2       | 原卷积 head 类型；最多 100 epochs，以 B_val mIoU 选模型 |
| NYUv2 / PASCAL / MTI-Net | 作者相应训练池内部划分        | W18 原配置 100 epochs、Adam 1e-4、batch 8 | 固定尺度/分辨率，已有卷积 head 类型；最多 100 epochs    |
| MP / CrystalTransformer  | 一个快照的 20k 试验池         | 作者 MT@2p 默认设置，上限 500 epochs      | 先材料级两层 MLP，最多 300 epochs；再评估已有局部读出   |
| QM9 / 现有 TinySchNet    | v2 的固定 100k 配置           | 现有上限 100 epochs                       | 沿用 v2 的 300 epochs 与诊断对照                        |
| rMD17 / ET               | 单分子官方 1,000 开发点       | 锁定官方 ET 版本与已有小模型配置          | 能量标量读出；最多 300 epochs，保留旋转不变性           |
| ADMET / MTGL-ADMET       | 全部可解析分子，稀疏标签保留  | 作者 200 epochs、Adam 1e-3、batch 128     | 两层 MLP，最多 300 epochs；只优化 CYP2C9                |

| MASSIVE / XLM-R | 1.0 en-US；源语句分组，官方 test 锁定 | 建议 20 epochs、Adam 2e-5、有效 batch 32；intent 选择 | token 级表示与槽位分布对照，最多 100 epochs |
| SNIPS / parallel | 固定论文词级槽位版本与 700 test | Joint BERT 建议 30 epochs、Adam 5e-5、batch 32；预印本/第三方实现边界单列 | 句子意图读出，完整 H 基线，最多 100 epochs |
| SNIPS / Stack-Propagation | 与 parallel 相同数据划分；分别配对各自 ST/MT | 作者 CLI 起点：Adam 1e-3、batch 16、上限 300 epochs；锁定可微开关 | 两组共同访问原意图预测，最多 100 epochs |
| SemEval / RACL | Res14 起步、Lap14 复核，各自 800 test | 建议上限 100 epochs、Adam 1e-4、batch 8；AE span-F1 选择 | 逐 token AE 读出，OE/SC 两辅助消融，最多 100 epochs |

表中下游设置均是建议，不是已有配置或已运行结果。MLP 起步可复用本项目已有宽度 64/32 的读出；新增辅助输入时应匹配参数量。下游初始学习率候选限定为 1e-3 与 3e-4、patience 20，由 B_val 选择；CV 可复用原 head 训练率。NLP 采用第 9–11 节的具体设置，patience 15 等案例设置优先于此处通用起点。不要把某一领域的固定学习率视为所有领域已经验证最优。

每个新案例先用 1 个上游 seed、1 个下游 seed验证损失、单位、缓存、冻结边界及标签身份；通过后再做 5 个上游 seeds × 5 个下游 seeds。正式 ST/MT 使用相同 split、主任务预算、早停上限及超参数搜索次数。辅助数量增加时同时报告参数、GPU 时间和缓存生成成本；同 epochs 不等于同计算预算。

### 12.3 效应与统计

主任务改善统一报告绝对差和相对变化，并明确方向。mIoU / ROC-AUC / intent accuracy / AE span-F1 越高越好；MAE / RMSE 越低越好。不将分类 AUC 与回归 R²平均后解释成通用性能比例。

每个上游 seed 内先平均其 5 个下游结果，再在 5 个上游 seed 上计算均值和样本 SD（ddof=1）。比较读出时配对同一个上游与下游 seed；比较 ST/MT 时配对训练 seed 和 split，另报逐 seed 方向。25 个下游组合不是 25 个独立上游重复。多个 rMD17 split 的不确定性与初始化不确定性分别报告。

最终报告同时回答：辅助监督是否改善 native；冻结 embedding-only 是否改善 native；辅助预测是否超过同容量、同输入层级的 embedding-only。只有最后一项成立，才有本案例的辅助输出 post-refinement 证据。若只胜过弱 native 或弱全局 MLP，结论应限定为读出/优化提升。

## 13. 当前记录与待验证项

本文新增的是候选与训练协议文档，没有下载数据集、安装旧环境、运行候选模型或增加新模型代码。公开论文成绩是各自协议下的证据，不是本项目成绩；代码读取只验证接口和静态结构。

正式实现前，每个案例需要保存数据来源/版本与许可、文件 hash、身份分组、标签来源/缺失情况、划分 manifest、预训练暴露记录、代码 commit、解析后的参数与随机种子。未知字段或无法运行的版本先明确标为待核查，不补写假定的配置。

本清单补充 [相关工作](related_work.md) 中的候选讨论；此前探索性的新辅助任务设计不自动成为本轮实施方案。工程阶段仍遵循 [项目 README](../README.md) 与 [pipeline 说明](PIPELINE.md)，只在 cross-domain 路线推进，保留已有 QM9 历史结果与用户文档。
