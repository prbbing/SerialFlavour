# Pipeline

存放通用阶段调度、训练循环、缓存校验、读出与指标工具。领域差异由 `../experiments/<dataset>/` 中的 data、model、training、refine、evaluate、analysis 模块实现。

命令入口位于 `../scripts/`，接口说明见 `../docs/PIPELINE.md`。
