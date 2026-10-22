# PASCAL-Context SSH GPU 调度

入口：`cross-domain/experiments/cv_pascal_context/scripts/run_ssh.sh`。

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

## 输入、依赖和启动

默认 `DATA_ROOT=/data/yuyang/cross-domain/inputs/cv_pascal_context`，需提前准备解压后的 `$DATA_ROOT/PASCAL_MT/`，含 `ImageSets/Context/{train,val}.txt`、图像及完整辅助标签；不能用普通 VOC 包替代。自动下载/解压保持禁用。ImageNet HRNet 权重默认 `$DATA_ROOT/hrnet_w18_small_model_v2.pth`；设置 `PRETRAINED_PATH` 可覆盖绝对路径，`PRETRAINED_SHA256` 必须填写经核对来源的 SHA256。`--plan` 允许尚未填写权重，`--check/--run` 拒绝缺失或不匹配。

SSH 环境已找到 numpy/PIL/scipy/yaml/six，缺少 cv2、skimage。依赖统一维护在 `cross-domain/requirements.txt`；经授权后安装到上述专用环境，不修改其他训练环境。脚本不自动 pip install 或下载权重。

默认只训练 multi_task，上游 5 seed，每个 4 recipe × 5 下游 seed，共 100 readout、105 行 Y。A_train=3000、B_train=1000，共享 val=500，498 张 unused，Y=5105。缓存临时峰值估算约 37.97GB，另预留 20GB；原始数据、权重及永久产物额外占用空间，因此不应放在当前根分区。

在远端、代码同步及输入准备完成后：

```bash
cd /home/yuyang/SerialFlavour
bash cross-domain/experiments/cv_pascal_context/scripts/run_ssh.sh --plan
# 以下 checksum 用经审核的真实权重 SHA256 替换
export PRETRAINED_SHA256='<64位真实SHA256>'
export CUDA_VISIBLE_DEVICES=0
bash cross-domain/experiments/cv_pascal_context/scripts/run_ssh.sh --check
tmux -L cross_pascal new-session -s cross_pascal
# 在上述 tmux 会话中执行，继承已设置的环境变量
bash cross-domain/experiments/cv_pascal_context/scripts/run_ssh.sh --run
# Ctrl-b d 脱离；tmux -L cross_pascal attach -t cross_pascal 返回
```

默认物理卡为 0；已有同名 tmux 会话时直接 attach，不重复启动。可通过 CONFIG 显式选择 `cluster_three_variant.json`，此时会完整运行三种上游，不自动扩展默认矩阵。

使用独立 tmux server（`-L`）使首次会话继承当前 shell 的部署变量；恢复已有会话时，须在会话内核对并重新设置相同变量。

## 本次验证

本地 WSL 执行两份脚本的 `bash -n`、全部内嵌 Python 的 AST 解析、错误参数退出检查，以及 PASCAL 两份、QM7-X 四份 lifecycle 配置的 `--plan`，共六份只读计划全部通过。未运行 `--check/--run`，未进行 GPU 训练、模型前向、真实数据处理或依赖安装。远端仅做环境只读检查，未修改远端文件。
