# Config

存放各数据集的实验配置，包括数据路径、任务定义、划分、模型、训练参数、seed 和输出位置。

初期按数据集命名配置文件；配置只描述参数，数据读取和预处理逻辑放在 data 中。

QM9 严谨对照配置：`qm9_gap_charge_bond_refine_v2_20k.json`（20k）、`qm9_gap_charge_bond_refine_v2_50k.json`（50k）、`qm9_gap_charge_bond_refine_v2.json`（100k）。默认输出和日志分别位于 `cross-domain/results/` 与 `cross-domain/logs/`；相对路径以工作树根目录为基准。
