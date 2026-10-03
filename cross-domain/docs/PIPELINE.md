# 通用 Cross-domain Pipeline

下文代码路径以 `cross-domain/` 为基准；CLI 会自动加载 `cross-domain/`，无需手动设置 PYTHONPATH。

入口是 `scripts/run.py`，仅负责配置、阶段依赖和调度，不包含 QM9 判断。每个领域的处理代码集中在 `experiments/<dataset>/`，包含 `data.py`、`model.py`、`training.py`、`refine.py`、`evaluate.py`、`analysis.py`，配置中的 `dataset` 决定模块加载。

## 实验目录组织

QM9 和 NYUv2 MTAN 的处理实现分别集中到 `experiments/qm9/` 与 `experiments/cv_nyu_mtan/`。每个目录包含六个处理模块（data、model、training、refine、evaluate、analysis）、配置、专用启动器与测试。公共 `pipeline/` 和通用 `scripts/` 保留。研究文档继续存放在 `docs/`，数据、缓存、结果与日志位置保持原有规则。

通用入口根据配置中的 `dataset` 加载 `experiments.<dataset>.<kind>`。各 JSON 的样本量、seed、训练预算和输出路径在此次目录迁移中保持不变。所有 cross-domain 实验的增量依赖统一位于 `requirements.txt`。

## 阶段与模块契约

| 阶段 | 调用领域模块 | 职责 |
|---|---|---|
| download | data.download(context) | 校验下载、记录来源 |
| prepare | data.prepare(context) | 标签、分组隔离划分、预处理 |
| train | training.train(context) | A 训练、A-val 选择上游 |
| cache | refine.cache(context) | 冻结上游并发布特征缓存 |
| refine | refine.train(context) | B 主任务监督、B-val 选择读出 |
| evaluate | evaluate.evaluate(context) | 锁定 Y-test 评估 |
| analyze | analysis.analyze(context) | 配对对照与聚合 |

每个阶段接收同一个 `Context`，返回实际生成的文件路径列表。`Context` 提供配置、工作树根目录、数据根目录、独立实验输出目录和模块加载；数据/输出相对路径均相对工作树根目录解析。CLI 的 `--config` 按调用者传入的文件路径解析，其他目录调用时使用绝对配置路径。

`--stage all` 按表格顺序执行；`--stage <阶段>` 要求全部前置阶段已有完整且校验通过的产物。阶段状态、耗时和错误保存在 `stage_state.json`。跳过完成阶段前逐文件检查大小和 SHA256；重建前置阶段会使后续完成记录失效。已有实验的代码/配置身份发生改变时，必须更换 experiment 名称，避免覆盖证据。当前 runner 每个实验使用一个进程，不提供并发运行同一实验的调度功能。

## 共享训练循环

`pipeline/fit.py` 提供优化、训练历史、验证集选择和 checkpoint 保存。领域适配器提供：

- `loss(model, batch) -> (loss_tensor, sample_count)`：定义自己的目标和样本粒度。
- `score(model, val_loader, device) -> float`：定义自己的验证指标和单位。
- 训练配置、模型与 loader、checkpoint 元数据；`selection_mode` 为 `min` 或 `max`。

标签形状、输入结构、回归/分类目标及指标解释由领域模块定义。`pipeline/readout.py` 提供可复用的表格 MLP 和训练集标准化，`pipeline/metrics.py` 提供回归指标；不强制其他领域使用相同模型或损失。

### 可选验证指标调度

训练配置可指定 `scheduler`：`type=reduce_on_plateau`、`factor`、`patience`、`min_lr_ratio`、`threshold`。
未提供该字段时保持固定 LR。调度继承 `selection_mode`，每轮验证后消费同一个 `score`；
绝对阈值默认继承 `min_delta`，最低 LR 按当前训练实际初始 LR 计算，cooldown=0。
有早停时调度 patience 必须小于早停 patience。降低 LR 不重置早停计数，也不改变最佳 checkpoint 选择。
存在 epoch 0 验证时先用它建立调度基线。history 保存本轮 LR、验证后的下一轮 LR 和是否衰减；
manifest 保存解析后的调度参数、衰减次数与最终 LR。当前 checkpoint 用于模型选择及推理，
没有增加 optimizer/scheduler 状态的中途训练恢复功能。

## 环境与 QM9 入口

统一使用 WSL conda 环境 `gn2_study_cross`，首次由 `gn2_study` 克隆。所有实验的新增依赖统一写入 `cross-domain/requirements.txt`，不再创建数据集或实验专用 requirements 文件，只安装到 cross 环境；实验 manifest 记录实际运行版本。不要把不同领域所需包加入原 Jet 环境。

```bash
source /home/yuyang/miniconda3/etc/profile.d/conda.sh
conda activate gn2_study_cross
cd /mnt/d/hep_analysis/gn2_study/SerialFlavour-cross
python cross-domain/scripts/run.py --config cross-domain/experiments/qm9/config/qm9_gap_charge_bond.json --stage all
python -m pytest cross-domain/tests cross-domain/experiments/qm9/tests cross-domain/experiments/cv_nyu_mtan/tests -q
```

新机器统一安装 cross-domain 增量依赖：`python -m pip install -r cross-domain/requirements.txt`。配置默认是 WSL 本地数据路径；其他主机修改 data_root，并更换实验名称。大型运行产物位于 `results/<dataset>/<experiment>/`，由本目录 `.gitignore` 排除；原始数据与处理缓存位于配置指定的数据根目录。

NYUv2 使用 `experiments/cv_nyu_mtan/scripts/run_cluster.sh` 复用通用 `run_unit.py`，提供带 SHA256 校验的 seed 队列续跑与逐图像磁盘缓存；完整配置和运行边界见 [集群 agent 操作说明](cv_nyu_mtan/cluster_agent_handoff_zh.md)。集群直接运行完整矩阵；当前仅静态检查与单元枚举通过，GPU 执行尚未验证。

## 单节点多卡集群调度

`scripts/run.py` 是本地单进程阶段运行器。集群（单节点多卡、无批处理）使用通用单元调度层：

- `pipeline/units.py`：从配置枚举工作单元（`prepare`、`upstream:<variant>:<seed>`、`cache:<variant>:<seed>`、`refine:<variant>:<us>:<recipe>:<ds>`、`evaluate`、`analyze`），`unit_filters` 把单元映射为运行时过滤条件。
- `scripts/run_unit.py --config <c> --unit <u>`：执行单个单元，写 `cross-domain/logs/<dataset>/<experiment>/units/<unit>.json`（状态、耗时、产物 SHA256）。**不使用 `stage_state.json`**，可安全并发；过滤条件不影响 `context.identity`。
- `scripts/run_seed.py --config <c> --variant <v> --seed <s> --gpu <n>`：一个 `(variant, seed)` 的完整生命周期（上游 → 缓存 → 全部 `recipe × downstream_seed`），设置 `CUDA_VISIBLE_DEVICES` 绑定单卡，已完成的 refine 目录跳过以便续跑，成功后写 seed marker。
- `scripts/run_pool.py --config <c> --gpus "0 1 2 3"`：先跑 `prepare`（有 marker 则跳过），再把 seed 单元派发到空闲 GPU，最后跑 `evaluate` 与 `analyze`；失败 seed 记录后继续，结束时返回非零。
- 便捷入口：`GPU_POOL="0 1 2 3" bash cross-domain/experiments/qm9/scripts/run_full.sh`（支持 `CONFIG`、`RETRIES`、`PYTHON`、`CONDA_ENV`）。

续跑：seed marker `status == complete` 时跳过；单 seed 内部已存在的 `best.pt`/缓存也会跳过。`GPU_POOL` 决定并发数；不为同一实验并发运行两个 pool 进程。

集群直接运行完整矩阵，不设置 pilot 或子矩阵前置流程。NYUv2 完整矩阵入口：

```bash
GPU_POOL="0 1 2 3" bash cross-domain/experiments/cv_nyu_mtan/scripts/run_cluster.sh
```

## 指纹与同步边界

实验代码指纹包含公共 pipeline、通用 Python CLI、`experiments/__init__.py`、当前实验包的 Python 处理模块和专用 Python/Bash 脚本。其他实验的处理代码与启动器、测试和文档不会改变本实验代码身份。配置内容单独计算指纹，因此改变配置仍需新实验身份。

远程无须访问 GitHub。通过现有文件同步方式定向更新 `cross-domain/experiments/<dataset>/` 即可部署该实验的代码、配置和入口。同步前先确认正在运行的是哪个实验；例如 QM9 正在运行时，可更新 NYUv2 目录，而不更新公共 pipeline、公共入口或 QM9 目录。本次没有执行任何远程同步。

公共模块仍是共享依赖，修改它们会影响所有实验。目录隔离也没有实现运行中代码快照：同一实验正在运行时，仍不可直接覆盖其代码或正在使用的配置。新进程会读取磁盘文件，代码指纹也会重新计算。应等该实验结束再更新，或把新版本放到另一个本地目录后启动；这个操作同样不需要 GitHub。

## 历史实验与迁移

此次模块路径与指纹范围变化会改变代码身份。历史 checkpoint、缓存、指标、完成标记与证据文档继续保留，不修改原有 manifest 来强行续跑。新代码应使用新的 `experiment` 名称；旧实验若需续跑，应使用与其 manifest 对应的旧代码和配置。不要在远程旧任务运行期间直接应用这次目录迁移。

旧的功能目录与专用入口已迁移，不保留重复实现或兼容入口。NYUv2 的完整矩阵仍为 173 个单元，其中 150 个 refine 单元；本次仅验证组织与依赖隔离，不执行集群配置。

## 本地验证记录

2026-10-03 在 WSL conda `gn2_study_cross` 环境运行公共与两套实验的离线回归测试：**37 passed in 16.57s**；随后移除 `src/` 层级后的复验为 **37 passed in 15.07s**。新增测试确认公共代码和本实验代码／脚本会改变指纹，而另一实验代码／脚本及测试文件不会改变指纹，并确认十二个处理模块从各自实验包加载。

Python AST 解析、两套 Bash 启动器语法检查、通用 CLI 参数加载及 NYUv2 wrapper 只读 dry-run 均通过。七份 JSON 配置与迁移前 Git HEAD 逐字节一致；用户的 `cross_domain_case_studies_zh.md` 保持原样。没有下载完整数据、启动集群配置、执行远程同步或验证 GPU；既有离线测试也不覆盖完整数据的磁盘加载／缓存执行。
