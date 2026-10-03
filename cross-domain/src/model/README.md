# Model

存放各数据集的上游模型及主任务、辅助任务头，初期每个数据集使用一个 <dataset>.py 文件。

模型结构由配置控制；明确各任务的输入、输出与损失。训练流程由 pipeline 组织，checkpoint 放在 results 对应实验目录中。
