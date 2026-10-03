# NYUv2＋MTAN 集群 agent 操作说明

更新日期：2026-10-03。本文件供接手 `feat/cross-domain` 工作树的集群 agent 使用，说明可运行入口、验证顺序、产物和失败处理。先读 cross-domain 目录的 [AGENTS.md](../../AGENTS.md)；数据身份与论文边界见 [本地测试说明](nyuv2_mtan_smoke_test_zh.md)。

**集群直接运行完整矩阵，不添加 pilot、缩小预算试跑或先试跑再放大的前置流程。** 这是用户明确要求，以后同样遵循。

本轮开发只完成静态检查与 dry-run，没有在本地或集群下载完整数据、运行新配置或验证 GPU。历史 CPU smoke 的 32 项测试通过属于上一实现版本，不代表本轮新增磁盘加载和集群调度已经通过运行验证。

本轮本地静态证据：新增/修改 Python 模块通过 AST 解析，JSON 配置解析及划分预算检查通过，Bash 入口通过 `bash -n`；完整配置只读枚举为 173 单元、150 个读出，ST 未枚举辅助 recipe。Bash wrapper 的 `--dry-run` 也通过；未创建完整矩阵实验目录或完整数据目录。未运行新增接口的单元/集成测试，未启动训练。

## 1. 文件与配置

| 文件 | 用途 |
|---|---|
| `experiments/cv_nyu_mtan/config/cluster_full.json` | 完整 795 train＋654 test、5 上游 seeds × 5 下游 seeds |
| `experiments/cv_nyu_mtan/scripts/run_cluster.sh` | Bash 入口；支持 CONFIG、GPU_POOL、PYTHON、CONDA_ENV、RETRIES |
| `experiments/cv_nyu_mtan/scripts/run_cluster.py` | 单节点多 GPU 队列、SHA256 续跑核验、单实例锁、最终评价 |
| `scripts/run_unit.py` | 复用的通用单工作单元执行器；写带身份／SHA256 的 unit marker |
| `requirements.txt` | 所有 cross-domain 实验的统一增量依赖；只安装到 cross 专用环境 |

新增调度器没有修改现有 QM9 pool/seed 行为。它复用 Context、工作单元枚举、worker 和 `run_unit.py`，但自行组织 NYUv2 seed 生命周期，以避免现有通用 pool 只凭 status 跳过或重复重训的问题。目录迁移后，通用 `Context.code_hash` 仅纳入公共代码与当前实验的处理代码和专用脚本，不再纳入另一实验的启动器。

目录整理的离线回归测试为 **37 passed in 16.57s**，包括两套实验的模块加载、代码指纹依赖隔离及既有协议测试；两套 Bash 入口通过语法检查，迁移后的 NYUv2 wrapper 只读 dry-run 仍为 173 单元。这些检查没有执行完整数据磁盘流水线或 GPU 训练。远程定向同步与历史运行迁移边界见 [实验目录组织说明](../PIPELINE.md#指纹与同步边界)：远程不需要访问 GitHub；本实验正在运行时仍不可覆盖它的代码或配置。

完整配置是缩小版 MTAN 的方法学实验，**不是原论文全宽度复现**：通道 `[16,32,64,128,128]`，分辨率 128×160，无预训练，无增强。上游最多 200 epochs，B 读出最多 100 epochs；AdamW、零 weight decay；上游/读出 LR 为 `1e-4`／`1e-3`。固定 A_val/B_val mIoU 选择、早停与 ReduceLROnPlateau，不使用 Y 调参。由于 MaxUnpool 的实现限制，固定随机种子但关闭强制 deterministic。

## 2. 目标位置与资源预检

使用用户指定的集群主机、cross-domain checkout 和专用 conda 环境。cross-domain/AGENTS.md 要求远程操作先确认目标和范围；用户未指定目标或授权启动时，不自行 SSH、同步、提交或推送。本文不是自动启动远程实验的授权。

先确认当前 checkout 包含新配置、数据／缓存接口和本启动器，分支正确、用户已有改动保持原样。旧提交 `15e1bf8` 仅包含 CPU smoke，不含本轮集群扩展。集群上必须有后续完整代码，不能只复制 JSON 或 Bash 文件。

```bash
pwd
git branch --show-current
git status --short
command -v python
python -c 'import sys, torch; print(sys.executable, torch.__version__); print(torch.cuda.is_available(), torch.cuda.device_count())'
nvidia-smi
df -h .
```

本调度器是 **Linux 单节点、一卡一个 seed 流程**，不做多节点 DDP，不负责向 Slurm/PBS 申请资源。若平台有调度系统，先由用户或其指定方式获得单节点 GPU allocation，再在 allocation 内运行。传入 GPU IDs 必须对应获准使用的设备；进程中 `CUDA_VISIBLE_DEVICES=<所选卡>`，配置中的 `cuda:0` 指该进程可见的第一张卡。

原始 14 个 Parquet 分片总计约 2.97GB，单片约 190–233MB，固定镜像 revision 并校验各片 SHA256。完整数据按图像处理／保存，不生成超过 5GB 的整包文件。每张图像的处理数据或空间缓存单独存为 `.pt`，逐样本懒加载；内存不随整个 B/Y 缓存线性增长。

128×160、当前宽度和目标划分下，缓存估算约 **47.5GB**：ST 每像素 `(32 embedding＋13 logits)×4＋8 label` 字节；MT 再加 `4 aux prediction＋32 aux hidden` 通道；每 seed 缓存 B_train 159＋B_val 79＋Y 654＝892 张，共 ST/MT 各 5 seeds。再加约 3GB 原始分片、1.1GB 处理数据、权重和文件系统开销，建议预留 **至少 70GB，最好 100GB** 可用磁盘。此为按 shape 估算，不是实测空间或 GPU 显存。每卡 batch 4，数据 loader workers 2，CPU threads 4；在完整运行过程中记录显存、速度和 I/O。

默认数据目录是 checkout 下 `cross-domain/local_data/cv_nyu_mtan`，已加入 Git 忽略规则。集群共享/scratch 路径不同时，先复制配置并改 data_root/output_root/log_root，使用新 experiment 名称；路径可为绝对路径，或相对 repo root。不要继承本地 Windows/WSL `/mnt/d` 路径。

## 3. 必须明确的划分边界

默认 full 配置为 `image_disjoint_exploratory`：在官方 795 train 中，固定 seed `20261003` 划分 A_train/A_val/B_train/B_val＝477/80/159/79；官方 654 test 全部作为 Y。检查 row IDs 与原 RGB 内容哈希，无完全重复，但镜像没有 scene 身份。因此运行成功也不能声称 scene-disjoint 正式结果。

若用户提供经核对的原图／场景映射，可将另一个配置改为 `split_mode=scene_grouped`，设置 `scene_manifest` 并使用新 experiment。scene manifest 是 JSON 字典，键必须与本程序的 row ID 完全一致：

```json
{
  "train:shard0:row0": "真实场景ID",
  "train:shard0:row1": "对应的真实场景ID",
  "val:shard0:row0": "官方测试图对应的真实场景ID"
}
```

示例仅解释 schema，不能直接作为完整映射。必须覆盖选中 train/test 全部样本；shard 为 0 开始，row 是分片内行号。不能按行号猜测场景或把图像 ID 当作场景 ID。程序拒绝 train/Y 场景交叠；训练场景整体分配到四个 A/B 分区，比例为目标，实际数量可偏离，最终以 manifest 为准。记录映射来源、覆盖和 SHA256。正确性依赖外部映射，代码无法自行证明镜像行与原图对应。

## 4. 环境与可选的只读配置检查

在已经授权的集群目录激活专用环境；环境名按集群实际情况使用，禁止安装到 Jet tagging 原环境。

```bash
source /home/yuyang/miniconda3/etc/profile.d/conda.sh  # 如路径不同，改为实际 conda 路径
conda activate gn2_study_cross
python -m pip install -r cross-domain/requirements.txt

CONFIG=cross-domain/experiments/cv_nyu_mtan/config/cluster_full.json \
GPU_POOL="0 1" bash cross-domain/experiments/cv_nyu_mtan/scripts/run_cluster.sh --dry-run
```

dry-run 只读取配置/源码，不调用 Context.initialize、不下载、不训练、不测试 CUDA。完整配置应枚举 **173 单元**：prepare 1＋上游 10＋cache 10＋refine 150＋evaluate 1＋analyze 1。150 个下游模型来自 ST 2 recipes × 5×5＝50，MT 4 recipes × 5×5＝100；最终 Y 评价为 10 native＋150 读出＝160 行。

只读枚举可按需使用，不是训练前置条件。数据准备、缓存冻结状态、参数匹配、评价完整性和资源使用在完整矩阵运行过程中检查；失败时保留日志并处理对应单元，不另设缩小版实验。

## 5. 完整运行

在用户指定的集群目录和 GPU 上直接启动完整矩阵：

```bash
CONFIG=cross-domain/experiments/cv_nyu_mtan/config/cluster_full.json \
GPU_POOL="0 1 2 3" RETRIES=1 \
bash cross-domain/experiments/cv_nyu_mtan/scripts/run_cluster.sh
```

可设 `PYTHON=/absolute/path/to/env/bin/python`，避免环境不一致；`CONDA_ENV` 可让 Bash 入口激活现有 conda 环境。运行不自动安装依赖。生产建议按集群 supervisor 或其 allocation 内前台运行；后台方式由该集群约定决定。

调度顺序：串行 prepare → 空闲 GPU 上一个 `(variant, upstream seed)` 的 train/cache/全部 recipes×downstream seeds → 全部 seed 成功后 evaluate/analyze。10 个上游训练独立，不做跨卡模型训练。任一 seed 最终失败时，其余已派发 seed 继续完成，主进程返回非零，**不会继续最终 Y 评价**。

任务级 `flock` 防止同一 experiment 同时启动两个本 pool。不要同时启动通用 `run_pool.py`、`run.py --stage all` 或手动单元来写同一 experiment，它们不共享该锁。完成队列时调度器校验真实 unit marker，而不使用缺乏产物校验的 seed aggregate marker。

完整配置的 5 upstream seeds 为 `17/23/31/43/59`，5 downstream seeds 为 `29/37/47/61/71`。相同上游种子内先平均 5 个 downstream 结果，再对 5 个 upstream means 计算样本 SD（ddof=1）；25 个下游组合不是 25 个独立上游重复。标准比较固定配对同一 upstream/downstream seed。不能根据 Y 改宽度、epoch、辅助组合或重新选 seed。

## 6. 产物、日志与续跑

默认位置：

```text
cross-domain/local_data/cv_nyu_mtan/raw/
  shard_index_<revision>.json / source.json / 14 个原始分片
cross-domain/local_data/cv_nyu_mtan/processed/<experiment>/
  samples/<global_index>.pt / splits.json / manifest.json
cross-domain/results/cv_nyu_mtan/<experiment>/
  run_manifest.json / resolved_config.json
  upstream/<variant>/seed<us>/
  cache/<variant>/seed<us>/<b_train,b_val,y_test>/<index>.pt
  cache/<variant>/seed<us>/manifest.json
  refine/<variant>/seed<us>/<recipe>/seed<ds>/
  evaluation.json / summary.json
cross-domain/logs/cv_nyu_mtan/<experiment>/
  cluster.lock
  <variant>__seed<us>.pool.log
  unit_output/<unit>.attempt<n>.log
  units/<unit>.json
```

本入口使用 **unit markers**，不生成或依赖 `stage_state.json`。这个文件只属于本地 `run.py` 阶段入口，不能用它判断 cluster pool 成功。unit_output/seed 日志追加保存；逐单元失败状态与 error 写在通用 unit marker 中。

正常中断后重新运行相同命令即可。已完成且配置/源码 identity、文件大小、SHA256 均一致的单元跳过；失败或未完成的单元从其开始重做。训练 checkpoint 用于选择，不保存 optimizer/scheduler 状态，故**不支持某 epoch 精确续训**。不要把重做失败单元写成连续 epoch resume。

如果已完成上游或 cache 的产物损坏且存在依赖它的后续 markers，启动器拒绝混合旧下游证据，要求新 experiment。若 prepare 失效但已有训练 markers，同样拒绝；不自动删除产物或清除依赖标记。代码/config 改变也须新 experiment，Context 会阻止覆盖。历史 smoke_cpu_v2 和其 docs 证据保留；当前 smoke.json 改用 smoke_cpu_v3，**该新名称尚未在本地运行**。

停止前台任务用 Ctrl+C；pool 会终止运行中的 seed 进程组。主队列 SIGTERM 也触发清理，但不要只 kill 某个孙进程或在 prepare/evaluate 期间只 kill 父 PID；按 supervisor 约定停止整个任务进程组，再核对 `ps`／`nvidia-smi`，避免残留写入进程后重复启动。

最终完成判据是所有计划单元有效、全部上游和读出 checkpoint、cache manifests、160 行 evaluation 和 summary 实际存在，退出码为 0。只看到“epoch 增长”或 GPU 利用率不等于完成。

## 7. agent 回报要求与禁止事项

回报当前代码/配置身份、实际数据 revision/hash、scene 分组状态、最终 split counts、seed 范围、实际 epoch/best epoch、有效 GPU/环境、耗时/显存/磁盘、全部对照的配对结果，以及失败/未验证内容。引用真实 manifest/metrics 路径。初期仅 image split 时，结论必须保留场景相关性限制。

原镜像法向生成版本、数据许可字段、深度物理尺度和原 MTAN 文件逐字节对应尚未独立核查；规模化下载不消除这些限制。不得用 B/Y 辅助真值替换缓存预测、更新冻结上游，或把隐特征收益解释为新增信息。不得把未收敛／单领域效果写成普遍方法成功。

保留用户已有文档修改、QM9 历史结果和失败记录。不要删除整个数据目录、覆写旧实验名、同步其他工作树、提交或推送，除非用户另有明确授权。
