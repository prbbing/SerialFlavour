# Config

存放 QM9 实验配置，包括数据路径、任务定义、划分、模型、训练参数、seed 和输出位置。

初期按数据集命名配置文件；配置只描述参数，数据读取和预处理逻辑放在 data 中。

QM9 严谨对照配置：`qm9_gap_charge_bond_refine_v2_20k.json`（20k）、`qm9_gap_charge_bond_refine_v2_50k.json`（50k）、`qm9_gap_charge_bond_refine_v2_100k.json`（100k）。默认输出和日志分别位于 `cross-domain/results/` 与 `cross-domain/logs/`；相对路径以工作树根目录为基准。

20k、50k 与两份 100k 正式配置的上游最大训练轮数为 500；下游 `epochs` 和 `set_epochs` 均为 300。早停 patience 保持不变。小规模 smoke 配置保留其短训练预算。

正式配置已启用验证 MAE 驱动的 LR 衰减：上游/下游 patience=8/4，factor=0.5，最低 LR 为实际初始 LR 的 1%；早停计数不因衰减重置。删除相应 `scheduler` 字段可运行固定 LR 对照，必须另设 experiment 名称。详见 `../../../docs/qm9/qm9_experiment_zh.md` 的 LR 衰减说明。
