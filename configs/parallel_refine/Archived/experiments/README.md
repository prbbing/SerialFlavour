# 旧方案实验归档

此目录保留旧方案实验配置，供查阅历史记录，不再用于新训练或当前结果比较。

- `experiment1/`、`experiment1_ex/`、`experiment2/`、`experiment2_ex/`
- 早期基础配置及过渡配置 `default_lr_decay.json`

共 120 个实验 JSON。当前使用的系列位于 `../../experiments/` 下的
`experiment1_ex_v2/`、`experiment2_ex_v2/` 和 `experiment3/`。

归档只移动配置并修正相对引用；已有结果目录没有移动。归档配置仍引用当前
共享组件，因此不能直接用它们重现旧训练方法；历史复现需使用对应代码版本
和运行记录。旧方案与当前方案的分界为 2026-10-05 的 `ed39f64`（LR 衰减）
及 `83740c3`（下游加权 CE）。
