# 通用 Cross-domain Pipeline

入口是 `pipeline/run.py`，仅负责配置、阶段依赖和调度，不包含 QM9 判断。每个领域使用自己的 `data/<dataset>.py`、`model/<dataset>.py`、`training/<dataset>.py`、`refine/<dataset>.py`、`evaluate/<dataset>.py`、`analysis/<dataset>.py`，配置中的 `dataset` 决定模块加载。

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

## 环境与 QM9 入口

统一使用 WSL conda 环境 `gn2_study_cross`，首次由 `gn2_study` 克隆。领域新增依赖写入对应 requirements 文件，只安装到 cross 环境；实验 manifest 记录实际运行版本。不要把不同领域所需包加入原 Jet 环境。

```bash
source /home/yuyang/miniconda3/etc/profile.d/conda.sh
conda activate gn2_study_cross
cd /mnt/d/hep_analysis/gn2_study/SerialFlavour-cross
python cross-domain/pipeline/run.py --config cross-domain/config/qm9_smoke.json --stage all
python -m pytest cross-domain/tests -q
```

新机器的 QM9 增量依赖：`python -m pip install -r cross-domain/requirements-qm9.txt`。配置默认是 WSL 本地数据路径；其他主机修改 data_root，并更换实验名称。大型运行产物位于 `results/<dataset>/<experiment>/`，由本目录 `.gitignore` 排除；原始数据与处理缓存位于配置指定的数据根目录。

## 单节点多卡集群调度

`pipeline/run.py` 是本地单进程阶段运行器。集群（单节点多卡、无批处理）使用通用单元调度层：

- `pipeline/units.py`：从配置枚举工作单元（`prepare`、`upstream:<variant>:<seed>`、`cache:<variant>:<seed>`、`refine:<variant>:<us>:<recipe>:<ds>`、`evaluate`、`analyze`），`unit_filters` 把单元映射为运行时过滤条件。
- `pipeline/run_unit.py --config <c> --unit <u>`：执行单个单元，写 `logs/<dataset>/<experiment>/units/<unit>.json`（状态、耗时、产物 SHA256）。**不使用 `stage_state.json`**，可安全并发；过滤条件不影响 `context.identity`。
- `pipeline/run_seed.py --config <c> --variant <v> --seed <s> --gpu <n>`：一个 `(variant, seed)` 的完整生命周期（上游 → 缓存 → 全部 `recipe × downstream_seed`），设置 `CUDA_VISIBLE_DEVICES` 绑定单卡，已完成的 refine 目录跳过以便续跑，成功后写 seed marker。
- `pipeline/run_pool.py --config <c> --gpus "0 1 2 3"`：先跑 `prepare`（有 marker 则跳过），再把 seed 单元派发到空闲 GPU，最后跑 `evaluate` 与 `analyze`；失败 seed 记录后继续，结束时返回非零。
- 便捷入口：`GPU_POOL="0 1 2 3" bash cross-domain/scripts/run_qm9_full.sh`（支持 `CONFIG`、`RETRIES`、`PYTHON`、`CONDA_ENV`）。

续跑：seed marker `status == complete` 时跳过；单 seed 内部已存在的 `best.pt`/缓存也会跳过。`GPU_POOL` 决定并发数；不为同一实验并发运行两个 pool 进程。
