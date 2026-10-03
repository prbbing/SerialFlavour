# Cross-domain

在 jet tagging 之外的领域复现 SerialFlavour 的 frozen post-refinement 方法：在 A 上训练并选择上游，冻结上游，在独立 B 上仅用主任务监督训练读出，最后在 Y 上比较 native head、强 embedding-only 与辅助增强读出。多任务上游的 embedding-only 不能替代真正的 single-task 上游基线。

工作约定见 [AGENTS.md](AGENTS.md)，研究背景见 [相关工作](docs/related_work.md)，通用接口、代码指纹及远程更新规则见 [PIPELINE.md](docs/PIPELINE.md)。

## 目录结构

```text
cross-domain/
├── AGENTS.md          # 本路线的工作约定
├── requirements.txt   # 所有实验的统一增量依赖
├── pipeline/          # 公共调度、训练循环、缓存与读出工具
├── scripts/           # 通用入口：run / run_unit / run_seed / run_pool
├── experiments/
│   ├── qm9/           # QM9 实现、配置、专用入口与测试
│   └── cv_nyu_mtan/   # NYUv2 MTAN 实现、配置、专用入口与测试
├── tests/             # 公共模块与实验隔离测试
├── docs/              # 实验设计、运行说明与历史结果
├── results/           # 实验产物
└── logs/              # 运行时生成的调度日志与完成标记
```

每个实验目录包含 `data.py`、`model.py`、`training.py`、`refine.py`、`evaluate.py`、`analysis.py`，以及 `config/`、`scripts/` 和 `tests/`。配置的 `dataset` 决定加载 `experiments.<dataset>.<kind>`；公共 pipeline 组织流程，实验包处理领域差异。

## 已实现实验

| 实验 | 配置与入口 | 说明与验证范围 |
|---|---|---|
| QM9 | 配置在 `experiments/qm9/config/`；集群入口为 `experiments/qm9/scripts/run_full.sh` | 已实现完整闭环；见 [实验设计](docs/qm9/qm9_experiment_zh.md)、[本地结果](docs/qm9/qm9_smoke_test_results_zh.md)、[集群说明](docs/qm9/cluster_handoff_qm9_full_zh.md) 与 [历史完整实验结果](docs/qm9/qm9_gap_charge_bond_full_results_zh.md) |
| NYUv2＋MTAN | `experiments/cv_nyu_mtan/config/smoke.json`、`cluster_full.json`；集群入口为 `experiments/cv_nyu_mtan/scripts/run_cluster.sh` | 真实数据 CPU 小规模闭环已完成；见 [本地说明](docs/cv_nyu_mtan/nyuv2_mtan_smoke_test_zh.md) 与 [集群 agent 说明](docs/cv_nyu_mtan/cluster_agent_handoff_zh.md) |

QM9 的 `qm9_gap_charge_bond.json` 用于本地小规模测试，`qm9_gap_charge_bond_full_100k.json` 对应历史完整方案；包含 native 初始化、同容量图消融及独立验证的对照配置为 `qm9_gap_charge_bond_refine_v2_{20k,50k,100k}.json`。历史方案的修正边界见结果文档。

NYUv2 历史 CPU 结果使用图像不重叠的小子集，不作为方法有效性证据。当前 smoke 配置名为 `smoke_cpu_v3`，尚未执行；完整数据的磁盘加载、缓存与 GPU 训练尚未验证。完整集群配置使用图像不重叠划分；scene 分组接口需要外部可靠身份映射。其他候选领域见 [案例建议](docs/cross_domain_case_studies_zh.md)，rMD17 路线延后。

## 环境与入口

依赖统一维护在 [requirements.txt](requirements.txt)，后续新增依赖也汇总到此文件。该文件是 `gn2_study_cross` 的增量依赖，PyTorch、NumPy 等基础依赖沿用克隆环境。安装与运行命令从工作树根目录执行：

```bash
conda activate gn2_study_cross
python -m pip install -r cross-domain/requirements.txt
```

通用阶段入口为 `python cross-domain/scripts/run.py --config <配置路径> --stage <阶段>`，阶段依次为 `download → prepare → train → cache → refine → evaluate → analyze`；`--stage all` 按顺序执行。详见 [阶段契约](docs/PIPELINE.md#阶段与模块契约)。

集群直接运行用户指定的完整矩阵，不添加 pilot 或子矩阵前置流程。单节点多 GPU 入口：

```bash
# QM9：默认历史 full 配置；其他方案通过 CONFIG 指定
GPU_POOL="0 1 2 3" bash cross-domain/experiments/qm9/scripts/run_full.sh

# NYUv2 MTAN：默认完整数据、5 上游 seeds × 5 下游 seeds
GPU_POOL="0 1 2 3" bash cross-domain/experiments/cv_nyu_mtan/scripts/run_cluster.sh
```

两套入口均支持 `CONFIG`、`GPU_POOL`、`PYTHON`、`CONDA_ENV` 和 `RETRIES`。根据主机环境修改配置中的数据路径；不要将集群配置在本地启动。本地小规模测试仅在明确要求时运行，不作为集群完整矩阵的前置条件。

离线回归测试命令：

```bash
python -m pytest cross-domain/tests cross-domain/experiments/qm9/tests cross-domain/experiments/cv_nyu_mtan/tests -q
```

本轮目录迁移后 37 项测试通过，Bash 入口和完整矩阵只读 dry-run 通过；这些检查不代表完整数据流水线或 GPU 训练已经验证。

## 产物与更新边界

结果按 `results/<dataset>/<experiment>/` 隔离，日志按 `logs/<dataset>/<experiment>/` 隔离；原始数据与处理缓存由配置的 `data_root` 指定，相对路径以工作树根目录为基准。大型数据、缓存、权重和预测不纳入 Git。历史结果及失败记录保留，代码或配置身份改变时使用新的 experiment 名称。

可通过文件同步单独更新 `experiments/<dataset>/`，远程不需要访问 GitHub。其他实验的专用代码不会改变当前实验代码指纹；公共 pipeline 和通用入口仍是共享依赖。同一实验运行期间不要覆盖其代码或配置，更新公共代码也需协调正在运行的实验。详见 [指纹与同步边界](docs/PIPELINE.md#指纹与同步边界)。

本路线代码集中在 `cross-domain/`，外部 Jet tagging 的 `src/`、`scripts/`、`configs/` 仅作协议参考。本地测试数据可放在 `D:\hep_analysis\gn2_study\dataset_ex`，按实验隔离；单个超过 5GB 的本地数据文件须提前获得授权。
