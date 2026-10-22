# QM7-X SSH GPU 调度

入口：`cross-domain/experiments/sci_qm7x/scripts/run_ssh.sh`；这里 QM7 指本轮实现的 QM7-X。

## SSH 环境与执行边界

2026-10-22 只读检查 `USTC_ATLAS_GPU`：4 张 NVIDIA A10，每张 23028 MiB；专用 Python 为 `/data/yuyang/miniconda3/envs/gn2_study_cross/bin/python`，Python 3.11.16、PyTorch 2.2.1+cu121，CUDA 可用。有 tmux，未找到 sbatch，因此使用 bash + tmux，不生成 Slurm 作业。GPU 空闲量随任务变化，这次检查不代表资源预留。

远端代码目录为 `/home/yuyang/SerialFlavour`。根分区约剩 34GB，`/data` 约剩 1.3TB；所有新增大文件默认放在 `/data/yuyang/cross-domain`。未在浅层目录检查中定位到本实验数据，不代表全机不存在。本次只读检查环境并编写本地脚本，未同步、安装、下载、处理数据或提交训练。

运行前需将当前 `cross-domain` 所需代码同步至远端，包括对应实验包、`pipeline/`、公共 `scripts/`、`requirements.txt` 和 `experiments/__init__.py`；远端旧代码不是此次提交的已验证副本。不要在运行途中覆盖代码：脚本和 Python 源码参与实验身份计算。

## 调度、配置与恢复

`--plan` 是默认模式：仅生成并清理 `/tmp` 配置、打印完整计划；不导入 torch、不创建结果目录、不启动任务。`--check` 额外导入依赖、检查 CUDA、输入存在性和输出盘预算；无模型构建/前向/训练。它不替代 prepare 的逐文件/schema/hash 审核。`--run` 先执行上述检查，再运行完整生命周期。

每个实验一次一个活动上游 seed，依次完成上游 → B 缓存 → 全部 readout → 流式 Y → 封存 receipt → 清理可重建缓存 → 下一 seed。GPU_POOL 的多个编号只按 seed 轮换，不能并行加速。两个实验可分别占用不同卡并行；当前 PASCAL batch=8 在 A10 的实际显存峰值尚未测量，脚本不擅自缩小 batch 或改变训练预算。

部署仅重写数据/输出/日志路径、PASCAL 权重校验值及 experiment 后缀，保留现有模型、A/B/Y、epochs、seed 和 recipe 矩阵。`RUN_SUFFIX` 默认 `ssh_a10`。正式启动将配置保存到 `$DEPLOY_ROOT/deploy/<dataset>/<experiment>.json`；同身份配置不可覆盖。默认日志/输出分别为 `$DEPLOY_ROOT/logs`、`$DEPLOY_ROOT/results/<dataset>/<experiment>`，使用原 lifecycle 的逐单元日志和状态。

失败后用完全相同的环境变量、代码和命令重启，校验并跳过有效完成单元；已封存并清理缓存的 seed 不重训。不提供 epoch/optimizer 中途恢复。`RETRIES=1` 默认额外重试一次失败单元。改变配置或代码需使用新 `RUN_SUFFIX`，保留旧证据。终止 tmux 中当前进程可用 Ctrl-C；失败不会触发未封存 seed 的缓存清理。

环境变量：`PYTHON`、`DEPLOY_ROOT`、`DATA_ROOT`、`CONFIG`、`RUN_SUFFIX`、`GPU_PHYSICAL`、`GPU_POOL`、`RETRIES`。已有 `CUDA_VISIBLE_DEVICES` 优先于 `GPU_PHYSICAL`，GPU_POOL 是其中的逻辑编号，默认 0；若已继承分配，不能仅设置 GPU_PHYSICAL 就认为换卡。例如 `CUDA_VISIBLE_DEVICES=2 GPU_POOL=0` 使用物理 GPU2。多个独立实验共用数据只读输入；禁止共用相同 experiment 输出身份并发运行。

## 输入、规模与启动

默认 `DATA_ROOT=/data/yuyang/cross-domain/inputs/sci_qm7x`，需已有解压后的 `1000.hdf5` 到 `8000.hdf5` 共八个源文件以及官方 `DupMols.dat`。脚本检查源文件存在和 duplicates MD5；prepare 执行完整数据审核。自动下载保持禁用，不自动解压大型原始数据。已发现专用环境的 numpy、h5py 和 torch；未执行真实 HDF5 读取或 GPU 前向。

默认 `cluster_50k.json`，保留 multi_task、5 个上游 seed × 5 个下游 seed × 5 个 recipe，共 125 readout、130 行 Y。B 缓存张量峰值估算约 187.14MB；原100k配置约 296.305MB。另有 prepared、checkpoint、元数据和原始八个 HDF5 的空间，原始全量源不能按抽样 50k 的缓存体积估算。空间检查包含配置指定的 1GiB reserve，不代表原始数据无需空间。

在远端、代码同步及输入准备完成后：

```bash
cd /home/yuyang/SerialFlavour
export CUDA_VISIBLE_DEVICES=1
bash cross-domain/experiments/sci_qm7x/scripts/run_ssh.sh --plan
bash cross-domain/experiments/sci_qm7x/scripts/run_ssh.sh --check
tmux -L cross_qm7x new-session -s cross_qm7x
bash cross-domain/experiments/sci_qm7x/scripts/run_ssh.sh --run
# Ctrl-b d 脱离；tmux -L cross_qm7x attach -t cross_qm7x 返回
```

默认物理卡为 1，与 PASCAL 默认卡分开。100k 使用以下环境变量后按同样顺序 check/run；无需修改 JSON，不要求先跑 50k：

```bash
export CONFIG=cross-domain/experiments/sci_qm7x/config/cluster_100k.json
```

若使用 ST/MT 消融，显式将 CONFIG 指向现有 `cluster_50k_st_mt_ablation.json` 或 `cluster_100k_st_mt_ablation.json`；旧 v1 all-cache 配置不受此 lifecycle 部署器支持。

使用独立 tmux server（`-L`）使首次会话继承当前 shell 的部署变量；恢复已有会话时，须在会话内核对并重新设置相同变量。

## 本次验证

本地 WSL 执行两份脚本的 `bash -n`、全部内嵌 Python 的 AST 解析、错误参数退出检查，以及 PASCAL 两份、QM7-X 四份 lifecycle 配置的 `--plan`，共六份只读计划全部通过。未运行 `--check/--run`，未进行 GPU 训练、模型前向、真实数据处理或依赖安装。远端仅做环境只读检查，未修改远端文件。

## 共享val更新

四份当前lifecycle配置统一A_val=B_val=5,000，使用完全相同的有序构型ID。50k仍实际使用50,000构型；原100k配置保留A_train=56,000、B_train=14,000、Y=20,000，实际使用95,000个独立构型，其余5,000个原验证池构型停用并留存ID。A/B训练及Y的样本和顺序保持，训练预算/seed/recipe不变。部署脚本直接读取新v3配置，不需修改启动命令；旧v2完成标记不复用。

2026-10-22本地WSL专用环境对当前QM7-X与共享生命周期执行离线合成检查：**48 passed in 53.45s**（QM7-X 37、共享11）。新增覆盖共享val去重存储、两阶段同序ID/indices/groups、原A/B/Y逐ID与顺序保持、原验证池与unused互补、val loader仅主标签、损坏共享记录/unused拒绝，以及四份当前配置数量和缓存预算。未下载真实数据、未真实训练/前向、未远程修改或同步。
