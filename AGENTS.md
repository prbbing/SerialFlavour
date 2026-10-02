# Probe / refinement 机理分析

## 目的与范围

本 worktree 对应 `feat/probe-refine`，研究 SerialFlavour 多任务模型冻结后的 refinement 收益来自哪里：训练目标与分类头重训、辅助监督塑造的表征、池化之外的轨迹信息，以及辅助预测引导的关系读出。
先读 `docs/probe_design.md` 和 `README.md`；调研稿中的机制假设、实验顺序与历史结果是研究背景，不代表新增实验已执行或结论已成立。

## 项目结构

- `docs/`：机制假设、实验设计与研究记录。
- `probe-refine/`：按 probe 手段组织新增代码、专用配置、结果分析与产物；`loss_grid/` 为损失权重网格分析，目前仅有 README。外部 Jet tagging 代码作为只读 protocol 参考，研究文档在 `docs/` 中维护。
- `src/parallel_model.py`：共享 Transformer 与 jet/origin/pair 三个任务头。
- `src/parallel_refine/`：A/B/Y 划分、冻结特征缓存、表格与图读出、评估及绘图。
- `configs/parallel_refine/`：数据、上游模型、refiner 与实验组合配置。
- `scripts/`：数据准备、上游训练、缓存、下游训练和评估入口；`scripts/production/` 为已有生产调度。

## 研究与实现原则

- 优先排除简单混杂：类别权重与 logits 校正、native-sized/强 embedding-only head，然后做 H-only set readout、无边/均匀边/打乱边/真实边及条件读出对照。具体执行范围以用户当前任务为准。
- 比较时明确固定的 checkpoint、数据、目标、容量、优化预算与选择准则；区分上游辅助监督的贡献和冻结后辅助读出的增量。
- 遵守事件级训练/验证/测试边界；检查实际 split manifest 与 `shared_validation`。标准化、CCA、残差拟合等只在相应训练折拟合；B-val 上的诊断及 checkpoint 选择不能充当独立测试，Y-test 不用于调参。
- 保留逐 seed 配对结果，写清 upstream/downstream 聚合与标准差口径；五个上游乘五个下游不能当作 25 个独立上游重复。
- 不把 CCA-private、线性残差近随机、可视化分离或 rejection 增益直接解释为严格独有信息、不可恢复性或因果机制。结论必须对应实际对照与产物。
- 参考现有数据处理、训练与评估 protocol，新增 probe 实现和专用配置放在 `probe-refine/` 对应方法目录中；只读复用外部入口时不修改其文件。保持实现简单，先做小规模验证再扩大实验，区分静态检查、实际运行、完整实验和图表验证。

## 工作边界

新增 probe 的代码、专用配置、结果分析与产物按方法集中于 `probe-refine/<method>/`；`docs/` 用于调研、实验设计与研究记录，可以随研究进展新增或更新。原则上不修改外部 Jet tagging 代码，包括 `src/`、`scripts/` 和 `configs/`，仅作数据处理、训练与评估 protocol 参考；确需修改时先说明原因、范围和影响并取得用户确认。用户明确指定的文件修改按其授权范围执行。

只在本 worktree 推进 probe 路线，保留用户文档和无关修改；不自行更换目录、分支、研究方案或同步其他路线。共享输入可只读复用，输出及可写缓存须隔离并核验身份。提交、推送、跨 worktree/SSH 同步须有用户授权；远程操作先确认目标与范围。每次报告说明改动、验证证据与尚未验证部分。
