# Cross-domain 跨领域复现

## 目的与范围

本 worktree 对应 `feat/cross-domain`，在 jet tagging 之外的领域检验 SerialFlavour 的 frozen post-refinement 思路：多任务训练后冻结上游，在独立数据上仅用主任务监督训练读出，比较 embedding-only 与辅助预测/局部结构读出的增量。
这里的跨领域首先指在不同领域复现方法，不默认指跨领域迁移同一个 jet 模型。先读 `cross-domain/docs/related_work.md` 和 `README.md`；文献与候选数据集建议不等于本项目实测结果，也不等于已选定实施方案。

## 项目结构

- `cross-domain/docs/`：相关工作、候选领域/数据集、协议设计与研究记录。
- `cross-domain/README.md`：本路线目录职责与实验流程；开始实现前阅读。
- `cross-domain/scripts/`：命令入口与启动器。
- `cross-domain/src/pipeline/`：通用模块，组织数据准备、上游训练、冻结缓存、下游训练与评估。
- `cross-domain/config/`：各数据集的实验配置，只描述参数，不承担数据处理逻辑。
- `cross-domain/src/data/`：数据读取、标签构造、划分与预处理代码；原始数据和缓存路径由配置指定。
- `cross-domain/src/model/`：各数据集的上游模型与主/辅助任务头。
- `cross-domain/src/training/`：各数据集的上游训练。
- `cross-domain/src/evaluate/`：独立测试评估。
- `cross-domain/src/refine/`：各数据集的冻结特征组织与下游读出。
- `cross-domain/src/analysis/`：各数据集的结果汇总、对照分析与绘图。
- `cross-domain/results/<dataset>/<experiment>/`：隔离保存运行产物。
- 外部 `src/`、`scripts/`、`configs/` 为现有 Jet tagging 实现，仅作 protocol 参考；当前 cross-domain 已实现 QM9 完整闭环（设计见 `cross-domain/docs/qm9/qm9_experiment_zh.md`，本地结果见 `cross-domain/docs/qm9/qm9_smoke_test_results_zh.md`），其他候选数据集尚无实现。

## 研究与实现原则

- 初期 `data/`、`model/`、`refine/`、`analysis/` 各按数据集使用一个文件，模块统一放入 `cross-domain/src/`，命令入口放入 `cross-domain/scripts/`；src/pipeline 管流程，数据集模块管差异。先完成一个数据集的最小闭环，再根据实际重复提取共用部分。

- 先明确领域、主/辅助任务、标签来源、数据许可、划分单位和小规模预算，再实现最小闭环；MASSIVE、PartImageNet、QM8/QM7-X 等只是候选，不自行扩大到全部领域。
- 保持核心协议可比：A 训练并选择多任务上游，冻结全部上游参数，B 仅用主任务标签训练下游，独立测试集用于最终评估。检查预训练暴露及辅助标签由主标签直接派生的捷径。
- 按领域使用事件、分子/骨架、蛋白同源簇、轨迹等适当分组，控制重复与相关样本泄漏；预处理和特征选择只用训练数据。具体分组规则须在实验协议中记录。
- 至少区分 native head、强 embedding-only、辅助增强读出；检验辅助监督时加入真正的 single-task 上游。多任务上游的 embedding-only 不能当作 single-task baseline。
- 匹配数据、目标、容量、训练预算和模型选择；oracle/truth 辅助输入只作诊断。分类与回归按领域选指标，不机械照搬 jet rejection。
- 记录数据版本、split、预训练来源、checkpoint、配置、冻结边界、seed 层级及计算成本；保留配对结果和不确定性，不把单一领域收益写成普遍机制。
- 区分联合训练中的任务交互与冻结后的二阶段读出；引用文献支持的具体内容。保持实现简单，先验证数据与小批次运行，再扩大规模。

## 工作边界

新增方法迁移、数据集测试、配置与结果分析实现集中于 `cross-domain/`，调研记录继续放在 `cross-domain/docs/`。原则上不修改外部 Jet tagging 代码；确需修改时先说明原因与范围并取得用户确认。大型数据、缓存、权重和预测不纳入 Git，产物写入前按需配置忽略规则。

只在本 worktree 推进 cross 路线，保留用户文档和无关修改；不自行更换目录、分支、候选领域或同步其他路线。共享输入可只读复用，输出及可写缓存按领域/实验隔离并核验身份。提交、推送、跨 worktree/SSH 同步须有用户授权；远程操作先确认目标与范围。每次报告说明改动、验证证据与尚未验证部分。

## 本地数据与运行规模

集群实验直接运行用户指定的完整矩阵。以后不主动增加 pilot、缩小预算的集群试跑配置或先试跑再放大的前置流程；不以运行验收为由要求先跑子矩阵。只读静态检查不启动训练。本地小规模测试仅在用户明确要求时进行，不作为集群完整矩阵的前置条件。

本地测试数据可存放在 `D:\hep_analysis\gn2_study\dataset_ex`，按数据集与实验隔离，具体路径写入配置；代码仍保留在本 worktree 的 `cross-domain/` 中。

原则上本地测试尽量使用小规模数据，验证数据处理、训练与评估流程后，再在 GPU 集群上运行正式实验。未经用户批准，不在本地下载、复制或生成超过 5GB 的单个数据文件；限制按单个文件计算。若所需本地测试文件超过该大小，须在保存前说明预计大小、必要性及可行的小规模替代方案，并请示用户，获得明确批准后方可继续相关操作。
