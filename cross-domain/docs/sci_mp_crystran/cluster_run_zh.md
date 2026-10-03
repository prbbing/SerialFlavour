# MP＋CrystalTransformer 远程集群运行说明

更新：2026-10-03。**本轮只编写配置、启动器和规模化兼容代码，做 Python/JSON/Bash 静态检查及只读 dry-run，没有下载新数据、安装远程环境、连接/同步集群、提交作业或执行训练。** 历史 CPU smoke v3 结果见 [本地说明](mp_crystran_smoke_test_zh.md)；不能将历史单 seed 测试当作本轮 GPU、多 seed、缓存规模或续跑逻辑已验证。

## 1. 默认完整实验矩阵

配置：[cluster_20k.json](../../experiments/sci_mp_crystran/config/cluster_20k.json)。入口：[run_cluster.sh](../../experiments/sci_mp_crystran/scripts/run_cluster.sh)，调度实现：[run_cluster.py](../../experiments/sci_mp_crystran/scripts/run_cluster.py)。

默认 `experiment=mp_crystran_cluster_20k_v2`，从固定 MP 2018-10-18 快照筛选去重后取 **20,000 个材料**。不增加 pilot 或子矩阵前置训练，启动后直接运行完整矩阵。

| 分区 | 样本数 | 用途 |
|---|---:|---|
| A_train | 11,200 | 上游训练、Eg/Ef 标准化 |
| A_val | 1,000 | 仅按 Eg MAE 选择上游 |
| B_train | 2,800 | 完全冻结后训练读出，仅 Eg 真值；输入统计拟合 |
| B_val | 1,000 | 读出选择，保留 native epoch 0 |
| Y | 4,000 | 全部训练/选择完成后最终评分 |

沿用主任务 PBE 带隙 Eg（eV），辅助形成能 Ef（eV/atom）；不加载 MP* 预训练权重。sampling/split seed 固定为 2026；每晶胞最多 64 原子，候选池为最终样本量的两倍，material ID 与 StructureMatcher 等价结构去重后再分区。若实际筛选后不够 20k，prepare 明确报错，不悄悄缩小样本量；40k 候选结构匹配的时间/内存尚未实测。划分是材料级独立，未增加组成不重叠泛化协议。

三种上游分别是作者 ST、`mt_main_only` 同 MT 架构仅 Eg 损失、`multi_task` Eg＋Ef 损失。每种上游 seeds 为 `[1,2,3,4,5]`；每个读出下游 seeds 也为 `[1,2,3,4,5]`，全部复用一个锁定 split。

| 上游 | 适用读出 | 上游训练数 | 下游读出训练数 |
|---|---|---:|---:|
| single_task | embedding、embedding_capacity | 5 | 2×5×5＝50 |
| mt_main_only | embedding、embedding_capacity | 5 | 2×5×5＝50 |
| multi_task | embedding、embedding_capacity、embedding_aux、embedding_aux_shuffle、embedding_hidden | 5 | 5×5×5＝125 |
| 合计 | 另评估各上游 native | 15 | 225 |

258 个单元＝1 prepare＋15 upstream＋15 cache＋225 refine＋1 evaluate＋1 analyze；最终 evaluation 应有 15 个 native 和 225 个读出，共 **240 条指标记录**。prepare 单元合并 download 与 prepare，集群不使用本地七阶段的 stage_state。

## 2. 模型、预算与统计

按用户确认缩小作者架构：宽度 **64、4 Transformer 层、4 heads、FFN 128**，dropout 0.1；ST 中间线性宽度保留 128。下游 MLP 隐藏层为 **128/64/32**，逐原子投影为 64→128，池化后与 g、native Eg、一个额外标量拼接，残差 MLP 为 194→128→64→32→1；所有读出使用相同架构。按当前实现层形状静态计算，ST 总参数 **145,697**，MT/同架构主任务总参数 **145,698**，其中 main-only 接受主损失梯度的参数为 141,473；下游总参数 **43,649**。参数统计包含逐原子投影、MLP 和 bias，不含 target mean/std buffers；没有初始化或运行模型。

此次将配置从原 256/8/8/512、下游 64/32 更新为上述较小上游和三层读出，实验身份改为 v2；20k 分区、5×5 seeds、训练预算与全部对照保持原配置。v2 是本项目缩小架构的实验，不是作者默认规模的精确复现。原 v1 配置也未执行，不能沿用其身份或旧完成标记。

模型和样本预算对照：

| 项目 | 历史 CPU smoke v3 | 当前集群 v2 |
|---|---:|---:|
| 上游宽度/层数/heads/FFN | 32/2/4/64 | 64/4/4/128 |
| ST / MT 总参数 | 23,121 / 20,946 | 145,697 / 145,698 |
| 下游隐藏层 / 总参数 | 64/32 / 10,561 | 128/64/32 / 43,649 |
| A_train / A_val | 512 / 64 | 11,200 / 1,000 |
| B_train / B_val | 256 / 64 | 2,800 / 1,000 |
| Y | 128 | 4,000 |
| 证据 | 历史真实数据 CPU 闭环 | 静态计划，未执行模型或训练 |

当前 MT 上游约为 Jet tagging [122,077 参数设置](../../../configs/parallel_refine/experiments/experiment1_ex/experiment1_ex_p122k_n1m_a080_b020.json) 的 **1.19 倍**；原未运行 v1 的 4,362,370 参数约为 Jet 的 35.7 倍。缩小模型用于控制容量和计算预算，不能仅凭参数接近证明两个领域有同样的样本效率。20k 是五个分区的总量，真正训练上游/下游的分别为 11,200/2,800 个材料；多 seeds、epochs 与晶胞中的原子都不能直接计为新增独立材料。下游采用缓存，因此优化的只有 43,649 个读出参数，上游 Transformer 不参与 B 的反向传播。数据是否足够、验证差距和微小增益是否稳定，仍需后续真实运行证据，不按 Y 调整规模或架构。

下游参数可拆为逐原子 64→128 投影的 8,320，加上 194→128→64→32→1 残差 MLP 的 35,329，共 43,649；所有 recipe 都保留一个额外标量输入位置，参数与输入宽度相同。上游/下游每 epoch 分别为 88/22 个训练 batch，按最大 epochs 计算完整矩阵最多 660,000/1,485,000 次参数更新（不含验证、缓存、评估和重试），实际会受早停影响；这些次数不能直接换算为壁钟时间。

| 设置 | 上游 | 下游 |
|---|---|---|
| 优化器 | 通用 AdamW | 通用 AdamW |
| 初始 LR / weight decay | 1e-3 / 1e-5 | 1e-3 / 1e-5 |
| batch | 128 | 128 |
| 最大 epochs | 500 | 300 |
| early stopping patience | 40 | 20 |
| ReduceLROnPlateau patience | 15 | 8 |
| scheduler factor / min LR ratio | 0.5 / 0.01 | 0.5 / 0.01 |
| scheduler threshold | 1e-4，绝对阈值 | 1e-4，绝对阈值 |
| 选择指标 | A_val Eg MAE (eV) | B_val Eg MAE (eV)，含 epoch 0 |

该配置复用现有通用 fitter，**不是作者 SGD＋StepLR 的精确训练复现**；没有临时修改公共 optimizer 接口。所有对照使用相同预算、scheduler 和选择规则。保持 FP32 张量及 deterministic 设置，没有新增 AMP；本轮未核验 CUDA kernel、TF32 实际配置或显存峰值。

每个读出共同访问完整 H、g、native Eg，auxiliary 组只增加冻结预测 Ef；容量匹配组额外复制一个 g 通道，shuffle 独立按分区打乱并重新训练，hidden 为一个 Ef 隐藏通道诊断。完整上游和任务头被冻结，B/Y 缓存没有辅助真值。没有新增主辅任务、坐标增强、周期邻居或对称性模型。历史 smoke 已观察到排序/平移/旋转敏感；本次规模化不会自动消除这些限制。

新 analysis 会检查完整 seed 矩阵，拒绝缺失和重复记录；**先在每个上游 seed 内平均 5 个下游结果，再在 5 个上游 seed 间报告均值与样本 SD（ddof=1）**。各读出对比配对同一上游/下游 seed，ST/MT native 配对上游 seed。25 个下游组合不是 25 个独立上游重复；summary 保存逐 seed 方向和全部配对差，不计算显著性或置信区间，也不自动选出“有效案例”。

### 从头训练与样本充分性的潜在问题

三种上游均随机初始化，不加载预训练权重。约 145.7k 上游参数仅使用 11,200 个 A_train 材料，43,649 参数的下游仅使用 2,800 个 B_train 材料；目前没有证明其足以稳定学习表征和读出。数据边界严格、参数接近 Jet 122k 或早停设置都不能替代样本充分性证据。辅助预测弱时的下游负结果，以及小样本正则化带来的上游正结果，都需要限制解释范围。

预训练可能改善样本效率，但须核验 MP* 与 B/Y 的材料及标签重叠、共同初始条件和架构兼容性；共同多任务预训练后的 main-only 不等于完全没有辅助监督的 ST。扩大正式数据或使用无重叠预训练均是后续选择，当前 20k 配置没有自动改变。详见 [从头训练、样本量与预训练边界](training_data_risks_zh.md)。

## 3. 文件部署、环境与源数据

在远程已确认的项目目录中部署本次 `experiments/sci_mp_crystran/` 和 `docs/sci_mp_crystran/`，并保留匹配的公共 `pipeline/`、`scripts/`、`experiments/__init__.py` 和统一 `requirements.txt`。远程不必访问 GitHub：作者模型、MIT 许可及参考文件已包含在实验包。不要把另一个实验的输出、缓存或旧状态复制进本实验目录。

公共 pipeline/scripts 本轮没有修改。MP data/refine 已兼容 `run_unit` 的 variant、upstream_seed、recipe、downstream_seed 过滤，避免一个 GPU 子进程执行其他 seeds。只需定向部署本实验包；若远程公共代码版本不匹配，先明确版本差异和同步范围，不覆盖正在运行的共享代码。

在目标 GPU 节点使用 cross 专属环境，安装命令供远程操作者执行：

```bash
cd /path/to/SerialFlavour-cross
source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate gn2_study_cross
python -m pip install -r cross-domain/requirements.txt
```

PyTorch 须是与集群驱动匹配的 CUDA 版本。MP 增量依赖已在统一 requirements：pymatgen 2026.9.24、ijson 3.5.0；本轮没有修改依赖文件或安装包。conda/项目绝对路径根据已确认部署位置替换，不把本地 `/mnt/d` 当作集群路径。

默认配置使用相对目录，均由工作树根目录解析：

```text
cross-domain/local_data/sci_mp_crystran/                      # 源数据
cross-domain/results/sci_mp_crystran/mp_crystran_cluster_20k_v2/ # 数据分区、模型、缓存、结果
cross-domain/logs/sci_mp_crystran/mp_crystran_cluster_20k_v2/    # 队列及 unit 日志、完成标记
```

可在启动前把 data_root/output_root/log_root 改为集群存储绝对路径，并使用新的 experiment 名称。不要在同一个 experiment 下修改数据、配置、代码或精度后强行续跑。

原始压缩文件仍是 `mp_all_20181018.json.gz`，184,077,884 bytes；input 单文件上限 500 MB，不解压落盘全数据库。集群能访问 Figshare 时 prepare 自动下载并校验；不能联网时，可由操作者将已校验原始文件放进配置的 data_root，再启动完整矩阵。源文件 SHA256：

```text
1f3de2dc7c68959647240921b841293fce918ea1708e6f3760ba35a9cdfe0500
```

数据来源和许可见 [THIRD-PARTY.md](../../experiments/sci_mp_crystran/THIRD-PARTY.md)。本轮没有执行传输。代码会兼容旧 JSON 的裸 NaN，缺失值保留为空而非补零。

## 4. 完整矩阵启动

只读计划命令只读配置和源代码，不初始化 output、不下载、不启动 subprocess，也不配置或访问 CUDA：

```bash
GPU_POOL="0 1 2 3" bash cross-domain/experiments/sci_mp_crystran/scripts/run_cluster.sh --dry-run
```

在已经分配的单节点 GPU 资源上启动完整矩阵：

```bash
GPU_POOL="0 1 2 3" CONDA_ENV=gn2_study_cross \
  bash cross-domain/experiments/sci_mp_crystran/scripts/run_cluster.sh
```

默认使用 GPU_POOL=0，即一个 GPU 顺序完成全部矩阵；选择更多 GPU 只改变调度并发，不改变配置 seed 矩阵。**已有 CUDA_VISIBLE_DEVICES 时，GPU_POOL 是该分配列表的索引**，例如 CUDA_VISIBLE_DEVICES=4,6 配合 GPU_POOL="0 1" 分别绑定设备 4 和 6；没有该环境变量时，GPU_POOL 是主机 CUDA 设备索引。UUID/MIG 标识可来自已有分配列表。该映射只按环境变量静态解析，集群驱动的实际枚举尚未验证；请在资源调度器分配的节点内运行。

每张选中 GPU 同时最多一个 `(variant, upstream_seed)` worker。一个 worker 依次执行上游训练、一次三分区缓存、该上游下全部 recipe×downstream_seed；每次实际计算调用公共 `scripts/run_unit.py`。所有 seed 成功并核验产物后，才统一执行 evaluate/analyze。

可选环境变量：`CONFIG`（默认 cluster_20k.json）、`PYTHON`（默认 python）、`CONDA_ENV`（可选）、`GPU_POOL`（默认 0）、`RETRIES`（默认 1，即首次之外最多再试一次）。例如已经激活环境时：

```bash
CONFIG=cross-domain/experiments/sci_mp_crystran/config/cluster_20k.json \
GPU_POOL="0 1 2 3" RETRIES=1 PYTHON=python \
  bash cross-domain/experiments/sci_mp_crystran/scripts/run_cluster.sh
```

调度器的 --variant/--seed 是内部 worker 参数，不作为缩小实验矩阵的使用入口。不要使用本项目公共 run_pool.py 代替该专用 wrapper。上述命令是供远程操作者运行的说明，本轮没有执行实际启动命令。

## 5. 资源估计、日志与中断后续跑

H-only 静态缓存大小：`(2800+1000+4000)×64×64×4 = 127,795,200 bytes`，每个上游约 121.875 MiB；15 个上游合计 **1,916,928,000 bytes，约 1.785 GiB**，为原宽度 256 配置的四分之一。最大单份 Y 的 H 为 62.5 MiB。g、Ef hidden、mask、标签、ID、prepared JSON、模型/读出权重、历史/临时文件还需额外空间；建议留出至少 30 GiB 独立可写磁盘空间，避免与其他实验缓存混放。

data prepare 使用候选池和结构匹配；当前训练会将 prepared JSON 和选中缓存加载到主机内存，cache 拼接有临时内存开销。多卡并发时每个进程都有数据和缓存开销，建议先按分配并发数预留主机内存（例如 4 workers 仍可按 32–64 GiB 保守预留；H 缓存缩小，但 40k 候选结构匹配的内存需求没有实测）。这些是静态规划值；本轮没有实际测量 RAM、显存、吞吐、壁钟时间或 20k 数据处理成功率。不承诺特定 GPU 型号、显存容量或完成时长。

日志根目录中：

- `single_task__seed1.pool.log` 等：每个上游生命周期进展。
- `unit_output/<unit>.attempt0.log` 等：完整子进程输出，包含 epoch/loss/validation/LR。
- `units/<unit>.json`：状态、身份、耗时和实际产物路径、大小、SHA256。
- `cluster.lock`：阻止两个 pool 同时运行同一配置实验；锁文件存在不等于进程仍运行。
- `cluster_complete.json`：全部完成后写出；仍须结合 258 个 unit markers、240 条指标记录和实际产物判断完成。

重启同一启动命令会先逐文件核验已完成单元并跳过；**这是工作单元级续跑，不是 epoch/optimizer/scheduler/RNG 的中断点恢复**。中途中断的训练单元将从头训练，不能仅凭 best.pt 判断完成。上游或缓存失效而已有依赖标记时会拒绝重建，需使用新实验保留旧证据；不会自动删除或混用旧下游模型。任何 worker 失败后，其他已派发任务可完成，但整体非零退出且不会进入 Y 评估。

SIGTERM/Ctrl-C 时 pool 清理已派发 worker 的进程组；kill -9、节点失联或作业系统强制终止的清理行为未运行验证，不要把 `.lock` 文件自行删除当作完成或恢复。已完成 Y 评估的再次启动只校验并跳过；如果其前置证据变化则拒绝沿用。

## 6. 最终结果和验证边界

最终 results 目录应有 `run_manifest.json`、`resolved_config.json`、`split_manifest.json`、`target_normalization.json`、`upstream/`、`cache/`、`refine/`、`evaluation.json`、`predictions.npz`、`summary.json`、`results_zh.md`。查看 summary 的 12 个 `(variant, recipe)` 分组和配对比较，保留负结果及逐 seed 方向；不要只看 MT 对弱 native 的改善。

集群走 unit markers，**没有 stage_state.json**；本地 smoke 的 `scripts/verify_run.py` 是历史单 seed/七阶段核验工具，不能用于此次集群完成判定。集群 wrapper 在最终评分前检查全部前置单元，evaluate 检查 Y 身份/顺序、读出 seed/recipe/variant 和 B 缓存身份，analysis 拒绝不完整矩阵。

本轮适配改变了 MP Python 模块和专用脚本的代码指纹；历史 v3 产物不覆写，旧代码与原 v3 配置才能对应其身份。当前 smoke 配置在三层读出兼容改动后改名 v5，仅避免误续跑历史产物；保留原小规模 32/2/4/64 与下游 64/32 设置，没有运行 v4 或 v5。

本轮已做：Python AST、JSON、Bash 语法、只读完整矩阵 dry-run、路径/文档链接和修改范围检查。最新 v2 dry-run 对模型构造、CUDA 查询/初始化、子进程与 Context.initialize 设置禁止调用保护，静态计划含三层读出宽度和参数量；检查前后均没有创建实验输出或日志目录。完整 258 单元静态计划与模拟分配列表映射保存在 [cluster_plan_static.json](cluster_plan_static.json)，其中设备 4/6 仅用于环境映射检查，不代表本机有这些 GPU 或已在这些 GPU 执行。已扩展 native epoch 0/容量匹配测试以覆盖两层与三层读出，并保留分层统计/配对/矩阵完整性测试代码，但**没有运行 pytest，也没有启动任何模型 forward、数据准备或训练**。因此 GPU 队列、恢复、中断、20k 去重、所有新增兼容代码和多 seed 数值结果仍需远程真实运行验证。本次提交仅归档代码、配置和文档，不包含集群运行证据；未执行 push 或 SSH/数据同步。