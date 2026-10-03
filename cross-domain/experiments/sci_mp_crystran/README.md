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

配置是 1,024 材料、单上游 seed 与单下游 seed、CPU 工程测试。当前结果写入
`cross-domain/results/sci_mp_crystran/mp_crystran_smoke_cpu_v3/`，源数据写入
`/mnt/d/hep_analysis/gn2_study/dataset_ex/sci_mp_crystran/`。输入文件上限为 500 MB。

现有 stage 指纹禁止原地覆写不同代码/配置的证据；修改代码/配置后须换新的 `experiment`。
`v1` 为历史 NaN 读取失败，`v2` 是添加同架构监督消融前的闭环，均保留在隔离结果目录。
当前入口与统计仅面向单 seed smoke；完整规模、多 seed 和 `run_pool` 尚未验证。

作者源文件、commit、MIT 许可和数据来源记录见 [THIRD-PARTY.md](THIRD-PARTY.md)。
不加载 MP* 预训练权重。保留作者首原子 token、笛卡尔坐标和原子顺序；没有周期邻居输入，
不保证平移/旋转/排列不变性。小规模结果不证明方法有效，也不等于论文复现成绩。