# MASSIVE＋官方 XLM-R：本地小规模方法测试

日期：2026-10-03（Asia/Shanghai）。代码位于 `D:\hep_analysis\gn2_study\SerialFlavour-cross` 的 `feat/cross-domain` Worktree；专用目录为 `cross-domain/experiments/nlp_massive_xlm/`。本次真实数据 CPU 运行正常退出，`download → prepare → train → cache → refine → evaluate → analyze` 七阶段完成并通过产物 SHA256 复核。精确配置、划分、逐模型训练记录和指标见 [smoke_evidence.json](smoke_evidence.json)。

**本次证明的是工程闭环，不是方法有效性。** 使用作者原版双头类，但将 XLM-R encoder 缩为 2 层、hidden size 64，从随机初始化训练；使用公共 XLM-R tokenizer。没有加载预训练 XLM-R Base 权重，也没有使用接触过完整 MASSIVE 的任务 checkpoint。一个上游 seed、一个下游 seed、560 条分区记录及低训练预算不能支持稳定增益或论文复现结论。

## 1. 来源与架构边界

MASSIVE 数据与 XLM-R 并行联合架构来自 [ACL 2023 论文](https://aclanthology.org/2023.acl-long.235/) 和 [作者仓库](https://github.com/alexa/massive)。本案例遵循 [案例建议第 9 节](../cross_domain_case_studies_zh.md)。

作者代码固定到 commit `f966f21846043aabef9b0f974fa7970027f43738`，`official_xlmr.py` 是 [`XLMRIntentClassSlotFill`](https://github.com/alexa/massive/blob/f966f21846043aabef9b0f974fa7970027f43738/src/massive/models/xlmr_ic_sf.py) 的原样副本，SHA256 为 `4aa7cd41c48f0cbbc7e604976d8fd2fa3c0f2e7e692b7064aaaf68fe8b426dbd`；保留 Apache-2.0 LICENSE、NOTICE 与 THIRD-PARTY。没有重写上游辅助头。兼容层只负责构造缩小配置、从同一次 encoder 前向抓取 H，以及接入现有 pipeline。

共享 XLM-R encoder 的 token H 经 masked mean 接作者 intent classifier；每个 token H 接作者 slot classifier。两个 classifier 各含作者已有的 Linear–GELU 中间层与最终 Linear，不读取对方的预测。intent 为 60 类；slot 为 55 种类型加 `Other`，合计 56 通道，**不是 111 类 BIO head**。原始词上的槽位类型来自 `annot_utt`，只有首个子词参与槽位 CE，续子词、特殊 token 和 padding 标签为 -100。模型输入来自无标记的 `utt`，不包含 scenario、annot_utt、intent、worker 或 judgments。

XLM-R tokenizer 固定为 [`FacebookAI/xlm-roberta-base`](https://huggingface.co/FacebookAI/xlm-roberta-base) 的 revision `e73636d4f797dec63c3081bb6ed5c7b0bb3f2089`，词表 250,002。使用完整公共词表，不在 B/Y 上拟合或收缩词表。小 encoder 参数仍以 token embedding 为主，总参数 **16,120,180**；ST/MT 的总架构相同。ST 冻结未监督的 slot head，只优化 intent CE；MT 优化 intent CE＋slot CE，slot 权重 1。两组 encoder 在 A 均训练。作者 pooler 未被双头使用，均冻结；ST/MT 可训练参数分别为 16,108,220 / 16,116,020。

缩小模型、随机初始化、AdamW、学习率 1e-3、无 warmup 与小样本是本地工程适配，不能称为论文的预训练 XLM-R Base 实验。公共 tokenizer 的训练文本暴露未经逐样本审计；本次不存在 encoder 预训练语料暴露，因为 encoder 没有加载预训练参数。

## 2. 数据、分组与 A / B / Y

从官方 MASSIVE 1.0 S3 归档下载约 39.50 MB；归档 SHA256 为 `7df623fd2d300a4d235d6ee5bd396c9a28258d3a0ccb29abdb054506eba153f8`。该归档包含多语言文件，但只解析并保存 en-US 的 3.90 MB JSONL，不解包其他 locale。tokenizer JSON 约 9.10 MB，SentencePiece 文件约 5.07 MB。数据许可按官方发布为 CC-BY-4.0，使用时保留论文归属；模型代码为 Apache-2.0。所有单个文件均远小于 5 GB。

原始数据和处理缓存位于 `D:\hep_analysis\gn2_study\dataset_ex\nlp_massive_xlm/`；配置使用相应 WSL 路径。复核官方 en-US train/dev/test 为 11,514 / 2,033 / 2,974，共 16,521。全部原始标注均做词对齐核查；没有把竞赛隐藏集作为可用标签。

将官方源 ID 与 NFKC、大小写折叠、空白归一化后的重复文本构成并查集组。全数据审计发现 **88 个重复组、7 个意图冲突组**。跨官方分区重复以 test 优先，其次 dev，最后 train；从较低优先级池剔除 **38 条记录**。剔除的源 ID 全部保存于 split manifest。测试保护针对完整官方 test，即使本轮只使用其中 120 条，也不让其重复语句进入开发池。冲突组整体绑定，按组中排序最小的 intent 做采样层归属，不拆成独立样本。

在各官方池中按 intent 分层、组内随机打乱、逐类轮询抽样；A/B 不共享来源或重复文本组，固定 split seed 20261003。样本不足或整组造成数量偏离时不拆组；本轮各预算正好满足。ontology 由完整官方 train 定义并排序，未依据 B/Y 拟合标签映射。样本/意图分布是分层小子集，不能代表官方测试分布。

| 分区 | 实际样本 | 覆盖 intent 类数 | 首子词/词数 |
|---|---:|---:|---:|
| a_train | 240 | 60 | 1601 |
| b_train | 80 | 59 | 472 |
| a_val | 60 | 59 | 411 |
| b_val | 60 | 59 | 385 |
| y_test | 120 | 59 | 753 |

A_train/A_val 来自官方 train/dev；B_train/B_val 分别来自同一官方池中与 A 不重叠的组；Y 仅来自官方 test。A/B train 比为 240:80，符合 75%:25% 的小规模缩放；dev 两组各 60。句长上限 64，首轮选中样本均无需截断；超长输入会显式失败，不能静默切掉槽位。

## 3. 训练、冻结与对照

上游 seed 17，ST/MT 使用相同初始共享参数、主任务样本和训练预算：batch 16、最多 4 epochs、AdamW lr 1e-3、weight decay 0.01、梯度裁剪 1；按 A_val intent accuracy 最大值选择，patience 3。当前共用 fit 对相同 accuracy 保留最早 checkpoint，**没有实现案例建议中的 NLL 同分打破规则**，NLL 仅报告。ST 选中 epoch 4；MT 选中 epoch 1，epoch 4 早停。不能因为 Y 结果修改 slot 权重或继续搜索上游。

随后冻结所有上游参数并设 eval，B/Y 推理不提供任何槽位标签。缓存 H `[N,64,64]`、原 intent logits、attention mask、词首子词 mask、主任务标签与源 ID；MT 另存 slot logits `[N,64,56]`。H 与 logits 来自同一次 encoder 前向。缓存没有辅助真值；B 的普通数据加载器也会剥离 `slots_num`。缓存前后模型 state 完全相同，B 训练前后上游 checkpoint SHA256 相同。

下游只训练 intent：seed 29、batch 16、最多 10 epochs、AdamW lr 1e-3、weight decay 0.01、clip 1、B_val accuracy 选择、patience 5。本轮没有学习率搜索。

| 读出 | token 输入 | 训练参数 | 对照含义 |
|---|---|---:|---|
| native | 原 intent head | B 中 0 | A 选择后直接在 Y 评估 |
| embedding | 完整 H，64 维 | 5,980 | 两种上游均有的普通强 token 读出 |
| embedding_matched | H＋重复 H 的 56 通道，共 120 维 | 7,772 | MT 辅助读出的同容量基线；ST 也报告 |
| aux_prediction | H＋56 维冻结 slot softmax，共 120 维 | 7,772 | 仅 MT，核心辅助输出比较 |

所有 B 读出均为逐 token Linear–GELU，masked mean/max pooling 后接句子 Linear；相同深度、宽度 32、训练预算及选择流程。额外 56 通道的两组使用相同的文本词首 mask。embedding_matched 的额外通道重复真实 H，不用全零通道制造无效参数。原 intent logits 对所有读出都可见，作为 residual 基底。最后 Linear 从零初始化，epoch 0 的 logits 与 native **精确一致**，且该初始 checkpoint 可参与 B_val 选择。它属于 residual 读出适配，不是论文上游模型新增结构。

参数量匹配不等于优化完全相同：重复 H 不提供新的输入方向，slot softmax 是已有非线性头的确定性重表达。任何收益都应解释为冻结特征的可读出性或优化差异，不能解释为相对完整 H 的新 Shannon 信息。本轮没有 aux-hidden、shuffle、关系图或多学习率诊断，不以其缺失推断机制。

## 4. 实测结果

Y 120 条，实际覆盖 59 个 intent 类；模型的完整 ontology 为 60 类。主指标 accuracy，macro-F1 固定对全部 60 类平均，NLL 为 intent CE；数值为本轮实测。

| 上游 | 读出 | 正确 / 总数 | Accuracy | Macro-F1 (60) | Intent NLL |
|---|---|---:|---:|---:|---:|
| ST | native | 13/120 | 10.83% | 0.0815 | 3.9221 |
| ST | embedding | 15/120 | 12.50% | 0.0921 | 3.9065 |
| ST | embedding_matched | 15/120 | 12.50% | 0.1175 | 3.9038 |
| MT | native | 2/120 | 1.67% | 0.0005 | 4.1038 |
| MT | embedding | 2/120 | 1.67% | 0.0005 | 4.1038 |
| MT | embedding_matched | 2/120 | 1.67% | 0.0005 | 4.1038 |
| MT | aux_prediction | 2/120 | 1.67% | 0.0005 | 4.1038 |

ST 的两个 B 读出均为 15/120，比 native 13/120 多 2 条，accuracy 增加 1.67 个百分点。MT native 为 2/120，比 ST native 低 9.17 个百分点。MT 的三个 B 读出全部由 B_val 选回 epoch 0，因此 Y 预测与 MT native 相同；**MT aux_prediction 相对同容量 MT embedding_matched 的增量为 0**。

MT native 的槽位 span micro-F1 与 semantic frame accuracy 均为 0。槽位评价以词首预测恢复词级类型序列，合并连续同类型为 span、排除 Other。额外用固定版本作者 `convert_to_bio` 核验：真实 native、随机预测与 oracle 预测三种情况下，gold/predicted/matched span 数与作者续子词合并口径完全一致。这是编码/指标的验证，oracle 只在隔离校验中出现，没有输入 B，也没有用于选择。

所有 seed 均只有一次：上游 seed 17、下游 seed 29。sample SD 为 null，不报伪多种子统计或显著性。四轮随机小 encoder 训练及 MT 槽位损失的学习动态足以造成主任务弱结果；本次不能外推槽位监督在正式预训练模型中有害。

## 5. 工程证据、入口与可复跑性

环境：WSL `gn2_study_cross`，Python 3.11.6、PyTorch 2.5.1+cu124、Transformers 4.44.2、SentencePiece 0.2.1，CPU 4 threads，CUDA 不可用。新增依赖只安装进 cross 环境，并补入统一 `cross-domain/requirements.txt`；没有修改 `gn2_study`。

七阶段内部计时共约 **42.21 秒**，其中上游约 14.89 秒、refine 约 10.54 秒。该时间不含最初网络下载与依赖安装，download 阶段复用了已固定校验的本地文件。运行产物约 141.92 MB，位于 `cross-domain/results/nlp_massive_xlm/smoke_cpu_v1/`，受现有 `.gitignore` 保护，不将权重或缓存纳入 Git。

复用公共 Context、download/prepare/train/cache/refine/evaluate/analyze 调度、fit、runtime 与工件 hash；公共 pipeline 和 Jet tagging 代码没有因 NLP 适配修改。专用模块均在实验包内，通用 work-unit 枚举正确得到 12 个单元并过滤 ST 的辅助读出；scheduler 的独立多 GPU 执行没有本地验证。

本次回归 **46 tests passed**（NLP 9 项，加已有公共/QM9/NYUv2 37 项），覆盖重复/源 ID 分组、ST 辅助真值拒绝、MT 首子词梯度、官方原类无标签输出一致性、一次 encoder 前向、匹配参数量、padding 不变性、native 初始输出一致性、数据篡改拒绝和通用模块加载。新增测试文件采用独立名字，避免与 CV 的 `test_protocol.py` 收集冲突。七阶段再次运行会校验全部 SHA256 后跳过，不再训练；Bash 专用入口语法通过，实际调用专用入口也能正确定位 Worktree 并校验、跳过已完成阶段。

从 WSL Worktree 根目录执行：

```bash
cd /mnt/d/hep_analysis/gn2_study/SerialFlavour-cross
source /home/yuyang/miniconda3/etc/profile.d/conda.sh
conda activate gn2_study_cross
python -m pip install -r cross-domain/requirements.txt
python cross-domain/scripts/run.py \
  --config cross-domain/experiments/nlp_massive_xlm/config/smoke.json --stage all
```

也可使用专用入口：

```bash
PYTHON=/home/yuyang/miniconda3/envs/gn2_study_cross/bin/python \
  bash cross-domain/experiments/nlp_massive_xlm/scripts/run_smoke.sh
```

原 experiment 身份固定为 `d80ac31097cabd6b895f2ce1d80888a52138eb2ffba093d4b770204f9dfc4744`。如果改动实验模块、公共 pipeline 或配置，必须使用新的 experiment 名字；不能覆盖本轮证据。测试和 docs 不改变运行代码指纹。单独请求某阶段仍要求前置阶段工件有效。

## 6. 正式实验前仍需完成

当前适配实现覆盖随机缩小 XLM-R smoke；**加载基础预训练 XLM-R Base 并在 A 上重新 fine-tune 的接口、完整官方测试、多 seed、CUDA 与多 GPU 均未验证**。下一阶段按案例建议单独接入基础预训练 encoder 权重，不使用完整 MASSIVE 微调 checkpoint；相应更新预训练暴露声明、optimizer/warmup、NLL 同分选择和新 experiment 身份。不要把本地随机缩小模型的 lr 1e-3 直接用于预训练 Base。

完整实验应使用更大的分组 A/B 池和锁定完整官方 test，ST/MT 保持主任务样本、初始 checkpoint、最大预算及搜索次数一致；先检查 MT 主任务学习，再比较配对的 H＋slot 与同容量 H。扩展多语言需绑定同一官方源 ID 的所有 locale，首轮 en-US 的词级处理不能直接用于中日文。正式统计按下游 seed 在每个上游 seed 内平均，再对独立上游 seed 均值计算 ddof=1 SD；不会把 5×5 的 25 个读出当成 25 个独立上游重复。

本次未提交、推送或同步到其他 Worktree/SSH；保留并行任务的其他实验和文档修改。
