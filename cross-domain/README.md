# Cross-domain

本目录用于在 jet tagging 之外的领域复现 SerialFlavour 的 frozen post-refinement 方法，检验多任务上游冻结后，辅助预测或局部结构是否能改善主任务读出。研究背景和候选数据集见 `../docs/related_work.md`，工作约定见 `../AGENTS.md`。

QM9 已实现完整闭环（`data/`、`model/`、`training/`、`refine/`、`evaluate/`、`analysis/` 各一个 `qm9.py`）。实验设计与运行（含集群单元调度）见 `../docs/qm9_experiment_zh.md`，本地结果见 `../docs/qm9_results_zh.md`，集群操作速查见 `../docs/cluster_handoff_qm9_full_zh.md`。配置：`config/qm9_gap_charge_bond.json`（本地 smoke）与 `config/qm9_gap_charge_bond_full.json`（集群 100k）。rMD17 路线延后。其他候选数据集的模型和指标仍须随具体实验确定。

## 目录职责

```text
cross-domain/
├── pipeline/      # 通用执行脚本
├── config/        # 各数据集的实验配置
├── data/          # 数据读取、标签构造、划分与预处理
├── model/         # 上游模型与主/辅助任务头
├── refine/        # 冻结特征组织与下游读出
├── analysis/      # 结果汇总、对照分析与绘图
└── results/       # 按数据集、实验保存运行产物
```

`data/`、`model/`、`refine/`、`analysis/` 初期各按数据集使用一个 `<dataset>.py` 文件；配置按数据集命名，复杂实验再按需拆分。不增加额外 `src/` 层。原始数据与缓存的存储路径由配置指定，`data/` 主要存放处理代码。

`pipeline/` 管理阶段顺序并调用数据集模块；数据集差异放在对应模块中，避免在通用脚本中堆积数据集判断。`config/` 描述路径、任务、模型与训练参数，不承担数据处理逻辑。先完成一个数据集的最小闭环，出现实际重复后再提取共用部分。

## 实验流程

1. 确定主/辅助任务、标签来源、划分单位和预算，准备 A/B/测试划分；预处理仅在训练数据拟合。
2. 在 A 上训练多任务上游，以相应验证集选择 checkpoint。
3. 冻结全部上游参数，生成下游所需的表征和预测特征。
4. 在 B 上仅用主任务监督训练 embedding-only 和辅助增强读出，使用 B 的验证部分选择模型。
5. 在独立测试集比较 native head、强 embedding-only 和辅助增强读出；用 `analysis/` 汇总配对结果与不确定性。

分类、回归和序列任务使用各自适合的损失与指标；统一训练协议，不强行统一标签形状或照搬 jet rejection。检验辅助监督贡献时，应另设真正的 single-task 上游。

## 产物与边界

运行产物放在 `results/<dataset>/<experiment>/`，保留解析后的配置、数据版本与划分身份、checkpoint 来源、seed、指标及分析结果。大型原始数据、缓存、权重和预测不纳入 Git；分析代码与必要的小型汇总可以版本管理。实现产物写入前，再配置对应的 Git 忽略规则。

新领域的方法迁移、数据处理、训练、配置和分析代码均集中于本目录。外部 `src/`、`scripts/`、`configs/` 中的 Jet tagging 实现原则上保持不变，仅供数据处理、训练与评估 protocol 参考；确需修改时，先说明原因和范围并取得用户确认。`../docs/` 继续保存调研与研究记录。

实验输出和可写缓存须按数据集与实验隔离。记录训练、验证、测试及预训练暴露边界，不将探索结果或单一领域收益表述为普遍机制。

## 本地数据与运行规模

本地测试数据可存放在 `D:\hep_analysis\gn2_study\dataset_ex`，按数据集与实验隔离，具体路径写入配置；代码仍保留在本 worktree 的 `cross-domain/` 中。

原则上本地测试尽量使用小规模数据，验证数据处理、训练与评估流程后，再在 GPU 集群上运行正式实验。未经用户批准，不在本地下载、复制或生成超过 5GB 的单个数据文件；限制按单个文件计算。若所需本地测试文件超过该大小，须在保存前说明预计大小、必要性及可行的小规模替代方案，并请示用户，获得明确批准后方可继续相关操作。
