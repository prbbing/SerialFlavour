# Refine

存放各数据集的冻结特征组织和下游读出模型，初期每个数据集使用一个 <dataset>.py 文件。

冻结全部上游参数，在 B 数据上仅用主任务监督训练读出；区分 embedding-only 与辅助增强读出，记录使用的 checkpoint 和特征。
