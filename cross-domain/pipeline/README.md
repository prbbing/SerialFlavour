# Pipeline

存放通用执行脚本，组织数据准备、上游训练、冻结特征缓存、下游训练和最终评估。

脚本通过配置调用对应的数据集模块；具体数据处理、模型和读出逻辑分别放在 data、model、training、refine、evaluate、analysis 中。入口为 `run.py --config <配置> --stage <阶段|all>`，QM9 已可完整运行。
