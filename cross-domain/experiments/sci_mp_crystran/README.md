# MP + CrystalTransformer

实现位置：`cross-domain/experiments/sci_mp_crystran/`。实验协议和实测结果见
[中文本地测试说明](../../docs/sci_mp_crystran/mp_crystran_smoke_test_zh.md)。

- `data.py`：公开固定 MP 2018-10-18 快照、哈希校验、非标准 NaN 流式兼容、材料与等价结构去重、A/B/Y。
- `model.py`：作者 ST / MT@2p 前向的缩小版、完整 H/g 导出、同架构关闭辅助监督对照、冻结残差读出。
- `training.py`：仅 A 训练/选择上游，复用公共 `pipeline.fit.fit`。
- `refine.py`：完整冻结缓存；仅 B 主任务监督；同容量 embedding / aux / shuffle / hidden 诊断。
- `evaluate.py`、`analysis.py`：最终 Y 预测、指标和配对 smoke 结果。
- `tests/`：作者前向数值对照、mask、结构分组、监督隔离、历史 NaN 兼容、epoch 0 与参数量检查。
- `scripts/verify_run.py`：七阶段 SHA256、split、冻结边界、native epoch 0 与 B_val 对称性审计。

在 WSL 的 `gn2_study_cross` 中，从 worktree 根目录运行：

```bash
conda activate gn2_study_cross
python -m pip install -r cross-domain/requirements.txt
bash cross-domain/experiments/sci_mp_crystran/scripts/run_smoke.sh
```

也可以直接复用公共入口：

```bash
python cross-domain/scripts/run.py \
  --config cross-domain/experiments/sci_mp_crystran/config/smoke.json --stage all
python cross-domain/experiments/sci_mp_crystran/scripts/verify_run.py \
  --config cross-domain/experiments/sci_mp_crystran/config/smoke.json
```

配置是 1,024 材料、单上游 seed 与单下游 seed、CPU 工程测试。历史成功结果写入
`cross-domain/results/sci_mp_crystran/mp_crystran_smoke_cpu_v3/`，源数据写入
`/mnt/d/hep_analysis/gn2_study/dataset_ex/sci_mp_crystran/`。输入文件上限为 500 MB。

现有 stage 指纹禁止原地覆写不同代码/配置的证据；修改代码/配置后须换新的 `experiment`。
`v1` 为历史 NaN 读取失败，`v2` 是添加同架构监督消融前的闭环，均保留在隔离结果目录。
本轮已补充 20k、5×5 集群配置、专用队列与多 seed 配对汇总，但没有执行训练。`v3` 历史产物保留；兼容性代码改变指纹后，smoke 配置在三层读出兼容更新后已改名为 `v5`，保持历史小模型设置；`v4`/`v5` 均尚未运行。

作者源文件、commit、MIT 许可和数据来源记录见 [THIRD-PARTY.md](THIRD-PARTY.md)。
不加载 MP* 预训练权重。保留作者首原子 token、笛卡尔坐标和原子顺序；没有周期邻居输入，
不保证平移/旋转/排列不变性。小规模结果不证明方法有效，也不等于论文复现成绩。
## 集群完整矩阵（只编写和静态检查，尚未执行）

当前配置为宽度 64 / 4 层 / 4 heads / FFN 128，上游约 145.7k 参数；下游隐藏层 128/64/32，共 43,649 参数。20k 材料与完整 5×5 seed 矩阵保持不变，实验身份为 `mp_crystran_cluster_20k_v2`。参数量为静态计算。

| 项目 | 当前集群配置 |
|---|---|
| 主任务 / 辅助 | Eg 带隙 / Ef 形成能 |
| 上游 ST / MT 参数 | 145,697 / 145,698，静态统计 |
| 下游输入与 MLP | 64→128 逐原子投影并池化；194→128→64→32→1 残差读出 |
| A_train / A_val / B_train / B_val / Y | 11,200 / 1,000 / 2,800 / 1,000 / 4,000 |
| 训练矩阵 | 15 次上游、225 次下游；共 258 工作单元 |
| H 缓存 | 每个上游 121.875 MiB，总计约 1.785 GiB；不含其他字段与临时副本 |
| 运行证据 | 当前集群 v2 与 smoke v5 未执行；历史 CPU v3 已完成 |

配置：[config/cluster_20k.json](config/cluster_20k.json)。使用说明：
[远程集群说明](../../docs/sci_mp_crystran/cluster_run_zh.md)。

**待验证风险：** 当前从随机初始化训练，A_train 11,200、B_train 2,800；尚无证据证明约 145.7k/43.6k 参数的上下游已获得充分训练。不能将弱辅助预测下的负结果解释为方法普遍无效，也不能把直接加载 MP* 权重视为无额外数据暴露。预训练的身份/标签重叠、ST/MT 比较含义与小架构兼容性见 [样本量与预训练边界](../../docs/sci_mp_crystran/training_data_risks_zh.md)。当前预算和随机初始化方案保持不变。

```bash
# 只读计划，不创建输出、不下载、不训练
GPU_POOL="0 1 2 3" bash cross-domain/experiments/sci_mp_crystran/scripts/run_cluster.sh --dry-run
# 在已分配的远程 GPU 节点执行完整 258 单元矩阵
GPU_POOL="0 1 2 3" bash cross-domain/experiments/sci_mp_crystran/scripts/run_cluster.sh
```

专用队列复用公共 `scripts/run_unit.py`，采用 unit markers，而非 `stage_state.json`。
存在 CUDA_VISIBLE_DEVICES 时，GPU_POOL 指该分配列表内的索引；队列保留设备分配映射。
公共 `run_pool.py` 不作为本实验的集群入口。不要将 smoke 的 `verify_run.py` 用于集群 unit 产物。