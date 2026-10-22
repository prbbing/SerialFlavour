# QM7-X：HOMO 与逐原子电子结构的 frozen post-refinement

本包实现 `download → prepare → train → cache → refine → evaluate → analyze` 全流程，配置中的 `dataset=sci_qm7x` 直接兼容公共 `Context`、`unit_filters`、`enumerate_units`、`run_unit.py`，默认由公共 `run_lifecycle.py` 按上游 seed 串行编排。实验问题是：A 上真实局部电子结构监督训练出的辅助预测，在上游完全冻结以后，能否帮助独立 B 上的有限容量 HOMO 读出。这里没有预设正迁移，也不把辅助预测当成冻结表征之外的新信息。

**当前状态（2026-10-22）：**代码、50k/100k 完整配置与入口已实现；已做 AST、JSON、Bash 语法、纯计划 dry-run，以及离线合成数据 contract tests。没有下载 QM7-X 数据、安装依赖、运行真实数据 prepare、训练、GPU 前向、完整矩阵或远程操作，也没有提交或推送。后文命令供后续获授权的执行使用。

## 来源、任务与单位

原始数据固定为 Hoja 等的 [QM7-X version 2.0，Zenodo 4288677](https://zenodo.org/records/4288677)。[官方小型 JSON 元数据](https://zenodo.org/api/records/4288677) 在本轮只读核实的值是 `version=2.0`、`license.id=cc-by-4.0`、`access_right=open`；数据为 CC BY 4.0，使用时引用数据 DOI `10.5281/zenodo.4288677` 与论文 DOI `10.1038/s41597-021-00812-2`，并注明本包的筛选、分组和抽样改动。

[原论文](https://www.nature.com/articles/s41597-021-00812-2) 与 [Table 2](https://www.nature.com/articles/s41597-021-00812-2/tables/3?error=cookies_not_supported)、[官方 README.txt](https://zenodo.org/records/4288677/files/README.txt?download=1)、[createDB.py](https://zenodo.org/records/4288677/files/createDB.py?download=1) 共同确认字段与层级。注意 Nature 的 `/tables/2` 实际显示 Table 1；这里链接的 `/tables/3` 才是性质字段表。

| 用途 | 原始 HDF5 字段 | 形状 | 原单位 |
|---|---|---|---|
| 原子种类 | `atNUM` | N | 原子序数 |
| 原子坐标 | `atXYZ` | N×3 | Å |
| 唯一主任务：HOMO 能量 | `eH` | 1 | eV |
| 辅助：Hirshfeld 电荷 | `hCHG` | N | e |
| 辅助：各向同性原子极化率 | `atPOL` | N | bohr³ |

每条结构通过 `file[molecule_id][geometry_id][field]` 读取。仅访问上述五个原始字段；不读 `eL`、`HLgap`、`KSE`，不通过 LUMO 与 gap 重建 HOMO。缺字段、形状错误、不支持元素、非有限值或非正极化率均明确失败，不伪造标签或自动填零。原子 padding 只用于批处理，损失和池化均由有效原子 mask 排除 padding。

本轮核验了官方存储说明与字段表，**尚未实际打开原始分片审计完整 schema、缺失率及分布**；离线测试使用同层级的合成 HDF5。解析器同时接受 README 的 `-50` 和作者 createDB 脚本提到的 `-d50` 位移后缀；偏离 `Geom-mr-is-ct-u` 或 u 不在 opt/1…100 范围的 ID 会报错。

## 数据规模、母分子分组与去重

约 4.2M 是平衡和非平衡**构型**数量，不能解释为 4.2M 个独立分子。正式配置中的 50k、100k 也都是总构型预算。每个 `r` 下所有立体异构体 `s`、平衡构象 `t` 及其 opt/扰动结构保持同组；A_train、A_val、B_train、B_val、Y 之间禁止任何母分子组重叠。

[官方 DupMols.dat](https://zenodo.org/records/4288677/files/DupMols.dat?download=1) 的原始 MD5 为 `5d886ccac38877c8cb26c07704dd1034`。它是 1900 行重复 opt ID 的**排除表**，没有提供“重复 ID → 代表 ID”的成对映射。本包将同 r 的剩余各家族归为一个母分子组，并排除列表中每个完整 `Geom-mr-is-ct` 家族的 opt 与全部扰动结构；不从该表臆造跨 r 代表映射。同一家族若出现在不同文件/根组则报错，阻止重复分片混用。这个规则控制已公布重复与同 r 相关性，不能声称发现所有未公布的化学等价结构或完成 scaffold-disjoint 划分。

抽样不读取标签值。先扫描 HDF5 组名并作去重、筛选，按固定 `split_seed` 对母分子排序划入五区；组数量按预定构型预算比例分配。每个母分子最多保留固定 `sampling_seed` 哈希顺序的 32 个构型，再在各自已隔离的分区内逐组轮流抽取构型，达到预定预算。容量不足会报错，绝不偷偷放宽 cap、跨组补样或缩小预算。扫描时只保存每组 cap 个候选身份，之后才读取选中构型的标签。

| 总构型预算 | A_train | A_val | B_train | B_val | Y |
|---|---:|---:|---:|---:|---:|
| 50,000 | 28,000 | 4,000 | 7,000 | 1,000 | 10,000 |
| 100,000 | 56,000 | 8,000 | 14,000 | 2,000 | 20,000 |

`equilibrium_only` 可显式设为 true，`max_conformations_per_parent`、`sizes`、`sampling_seed`、`split_seed` 均可配置。仅平衡结构时不能沿用超过可用容量的预算，需更改配置与 experiment 名称。正式配置不暗加平衡结构 pilot。

`split_manifest.json` 报告每区已分配母分子数、实际抽到的母分子数、构型数、完整样本 ID、indices、重复排除数与两个 seed；这些母分子数必须在真实 prepare 后计算，本轮未猜测。源码/配置 identity、原始每个 HDF5 的 SHA256/大小/路径、重复表哈希、split manifest 哈希、prepared 张量哈希与 normalization 哈希全部保存在 manifest；再次使用时核验。split indices 必须完整且唯一地覆盖所有 prepared 行，行 ID、解析出的 r、tensor 内 parents 与各区 groups 必须相符。

## 模型、来源与许可边界

`model.py` 是**本项目包内独立实现的 SchNet 式小模型**，思路参考 [Schütt 等，NeurIPS 2017](https://arxiv.org/abs/1706.08566)。它不导入其他实验包，不复制第三方仓库源码，不加载预训练权重，因此依赖源码均纳入当前实验的 `Context.code_hash`。没有引入新的第三方代码许可；本包项目代码的许可由仓库权利人决定，本实现不擅自为整个仓库指定许可证。SchNet 论文方法引用与 QM7-X 数据的 CC BY 4.0 是不同事项。

这里的多头组合、均值全局池化、A/B/Y 协议及下游残差读出都是项目适配，**不是已发表 QM7-X 同组合多任务架构的忠实复现**，也不调用作者 createDB.py 或要求安装 SchNetPack、ASE、PyG。

正式骨干宽度 96、4 个连续距离过滤 interaction、32 个 Gaussian radial basis、5 Å cutoff；HOMO 头隐藏宽度 64。原子辅助头使用 `[H_i, g] → 64 → 2`，输出 hCHG 与 atPOL。距离输入提供平移/旋转不变性，原子表示与局部头对排列等变；没有周期边界、显式角度高阶张量、力导数监督或严格物理电荷守恒约束。它是成本受控的分子几何模型，不能据此宣称达到原数据集或现代等变模型的最佳精度。

ST 与 MT 使用完全相同的模型结构和初始化顺序。ST 不读取局部标签、不计算辅助损失，辅助头保留为闲置模块；这样主干与主头容量严格匹配。ST/MT 总参数均为 **181,763**。有效优化路径仍不同，MT 的辅助监督更新共享骨干与局部头，这正是实验变量。

A_train 拟合 HOMO 均值/总体标准差以及有效原子的两项局部标签均值/总体标准差；训练目标为标准化 HOMO MSE，加上 MT 中 `0.5×charge MSE + 0.5×polarizability MSE`。主任务按构型平均，局部任务按有效原子平均。A_val 仅按 HOMO MAE（eV）选上游 checkpoint；B/Y 不参与上游选择与目标标准化。没有把训练期间联合目标改善写成 post-refinement 收益。

## 冻结边界、读出与完整矩阵

缓存时上游所有参数 `requires_grad=False` 且处于 eval；缓存前后核对上游 checkpoint SHA256 不变。默认仅 B_train、B_val 的缓存包含完整 H、g、native HOMO、预测局部物性、完整辅助 hidden、coords、mask、主标签和样本身份；Y 在固定 batch 内通过冻结上游计算，全部已选择读出共享 H/g/native，批次评分后释放，**没有辅助真值**。缓存字段白名单、各 split SHA256、checkpoint identity 与样本顺序都要核验；B loader 禁止请求辅助真值。

每个下游模型都有完整 H 的 learned attention、mean、max 池化、g、native HOMO、原子数及相同的距离 radial 几何访问，避免让辅助对照独享坐标或局部层级。所有配方使用相同的 64 维 extra 槽位、`128/64/32` 读出、参数总数 **110,466** 与预算：

| 配方 | extra 输入 | 上游 |
|---|---|---|
| `embedding` | 全零 extra；强 full-H 读出 | ST、MT |
| `embedding_capacity` | H 前 64 维的确定性副本；容量/额外输入对照 | ST、MT |
| `embedding_aux` | 两个预测局部物性，其余槽位补零 | MT |
| `embedding_hidden` | 完整 64 维辅助头隐藏表示 | MT |
| `embedding_aux_shuffle` | 在各自分子内共同打乱两项预测的原子对应；padding 不动 | MT |

分子内 shuffle 保留该分子两种局部量的成对值与总体分布，只破坏它们相对 H/坐标的局部对应；不会把其他分子的预测移入当前分子。B_train、B_val、Y 使用预先固定的不同 seed offset，并对打乱配方重新训练读出。所有输入标准化仅在相应 B_train 有效原子/分子上拟合，然后固定应用到 B_val/Y；native HOMO 保留原单位用于残差起点。

所有配方最终残差层以零权重、零偏置初始化，使 epoch 0 输出逐项等于缓存 native HOMO；将 epoch 0 保留为 B_val 可选 checkpoint。下游只用 B_train HOMO 监督和 B_val HOMO MAE 早停，绝不重新优化上游。下游模型只包含普通 MLP/池化，无挂接的上游模块、BatchNorm 或 dropout 状态，因此公共 fit 的 `model.train()` 不改变冻结上游行为。

默认完整矩阵为 MT 上游 seeds `[11,22,33,44,55]` × 下游 seeds `[101,102,103,104,105]`：**5 个 A 训练、125 个 B 训练、138 个公共阶段单元、130 行 Y 指标**。5 个配方均保留，不改变 50k/100k 数据预算、任务、几何/容量对照、FP32、200/120 epoch 上限与早停设置。`cluster_50k_st_mt_ablation.json` / `cluster_100k_st_mt_ablation.json` 可显式加入结构匹配 ST 消融，对应 10 个 A 训练、175 个 B 训练、198 单元与185行。默认 MT-only 只能研究已训练 MT 表征的冻结读出差异，不能据此估计 MT 相对 ST 的辅助监督作用。完整清单由纯 dry-run 输出。

每个 seed 的 Y 评估前，先核验该 seed 全部必需 checkpoint、B cache、recipe×downstream seed 完成标记及其 artifact 哈希；缺失、重复、不一致立即报错。只读取 A_val/B_val 已选 checkpoint 与各自 B_train normalization，不用 Y 选择模型或决定是否继续后续 seed。默认每个 batch 仅一次冻结上游前向，25 个小读出共享同一批 H/g/native；分子内 shuffle 由 sample ID 与原 seed 即时重现。native 每 seed 只计一次，误差用 Float64 累加，不永久存 Y H/aux hidden，`evaluation.save_predictions=false` 默认只存小指标。Y 完成证据由每 seed `evaluation_manifest.json` 和 durable receipt 锁定；全部计划 receipt 完整有效后才汇总，输入或产物变化后拒绝覆盖。`evaluation.auxiliary_diagnostics=true` 可单独从 prepared Y 读取局部真值，输出独立 `auxiliary_diagnostics.json`；默认关闭，它不是 cache 输入，不参与读出训练/选择，也不训练 oracle。

主要输出 MAE、RMSE、bias（均 eV），并给出 native→embedding、容量、hidden、shuffle 与 predicted-aux 的同 seed 配对差。先在每个上游 seed 内平均下游 seeds，再对 5 个上游 seed 报均值与样本 SD（ddof=1）；25 个嵌套单元不是 25 个独立上游重复。正的配对差定义为左侧 MAE 减右侧 MAE，表示右侧改善。统计仅描述固定母分子划分上的初始化差异，不包含划分不确定性，不自动给出显著性或机制结论。

## 数据、环境与运行命令

所有命令从 `SerialFlavour-cross` worktree 根目录执行。依赖使用唯一公共 `cross-domain/requirements.txt`；本包新增需求为 `h5py>=3.10,<4`，其余使用既有 PyTorch、NumPy 与标准库。依赖只在 cross 专用环境安装，本轮仅登记而未安装。离线 tests 额外依赖 pytest。

```bash
conda activate gn2_study_cross
# 首次配置环境时按公共清单安装；本轮没有执行安装
python -m pip install -r cross-domain/requirements.txt

# 只读：不创建数据/输出目录、不导入模型或CUDA、不启动子任务
python cross-domain/experiments/sci_qm7x/scripts/run_cluster.py \
  --config cross-domain/experiments/sci_qm7x/config/cluster_50k.json --dry-run
python cross-domain/experiments/sci_qm7x/scripts/run_cluster.py \
  --config cross-domain/experiments/sci_qm7x/config/cluster_100k.json --dry-run
```

默认 `data_root=/mnt/d/hep_analysis/gn2_study/dataset_ex/qm7x`，期望其中已有 `1000.hdf5`…`8000.hdf5` 和原始字节的 `DupMols.dat`。`data.sources` 可改为绝对本地路径及可选 `sha256` 锁；未配置官方解压后 SHA256 时，首次读取逐文件记录本地 SHA256，后续必须一致。部署主机修改路径后需使用新的 experiment 名称。Windows 的 WSL 路径不能直接解释为当前 Windows 数据路径；正式执行使用 WSL/Linux 配置。

默认 `allow_download=false` 且 `download_files=[]`，download 阶段只校验已有本地输入。可选流式下载必须同时显式设置 `allow_download=true`、文件名、官方 URL 与 MD5；仅限 Zenodo 固定 record，不自动下载全量或解压。例如以下是配置片段，**不是本轮已执行下载**：

```json
{
  "allow_download": true,
  "download_files": [
    {"name": "8000.xz", "url": "https://zenodo.org/records/4288677/files/8000.xz?download=1", "md5": "c893ae88b8f5c32541c3f024fc1daa45"},
    {"name": "DupMols.dat", "url": "https://zenodo.org/records/4288677/files/DupMols.dat?download=1", "md5": "5d886ccac38877c8cb26c07704dd1034"}
  ]
}
```

download 单独阶段可返回已下载归档和 `pending_local_hdf5`，然后停止；prepare 要求本地已解压 HDF5 齐备。八个 XZ 合计约 9.6 GB，并不等于本次训练结构预算或解压后体积。每个自动下载和生成的单文件都受不超过 5,000,000,000 bytes 的限制；预先读取 Content-Length，并在流式传输中再次计数。不会自动生成巨大合并数据库或把大分片解压成本隐藏在 prepare 中。已有大 HDF5 可只读使用；需要生成超过 5GB 的本地单文件时须另行取得用户授权，本包不会自动执行。

本包数据接口与单进程入口：

```bash
# 默认使用本地现有输入；不需要网络
python cross-domain/scripts/run.py --config cross-domain/experiments/sci_qm7x/config/cluster_50k.json --stage download
python cross-domain/scripts/run.py --config cross-domain/experiments/sci_qm7x/config/cluster_50k.json --stage prepare
# resource 配置的完整流程须使用下面生命周期入口；无筛选 train/cache/refine 会明确拒绝
```

完整 GPU 矩阵使用包内 wrapper，它调用公共 `pipeline.lifecycle.main` / `scripts/run_lifecycle.py`。每次启动先核验 source/data manifest 与原始输入 SHA256，然后依次执行一个 `(variant, upstream_seed)` 的 upstream → B cache → 全部 recipe×下游 seed → 独立进程流式 Y → durable receipt → 退役可再生成 B cache，再处理下一 seed。支持 `CONFIG`、`GPU_POOL`、`PYTHON`、可选 `CONDA_ENV`、`CONDA_SH`、`RETRIES`：

```bash
GPU_POOL="0" bash cross-domain/experiments/sci_qm7x/scripts/run_cluster.sh
CONFIG=cross-domain/experiments/sci_qm7x/config/cluster_100k.json \
  GPU_POOL="0" bash cross-domain/experiments/sci_qm7x/scripts/run_cluster.sh
```

若继承 `CUDA_VISIBLE_DEVICES`，GPU_POOL 是其中的逻辑下标，wrapper 映射到实际分配的 GPU/UUID。默认 `max_active_seeds=1`，多个 GPU 仅按 seed 轮换，仍只允许一个活动 seed；独占锁防止同一实验重复启动。receipt 永久封存所有 upstream/readout checkpoint、训练记录、input normalization、小 cache manifest、seed metrics 和来源证据的哈希。续跑先核验 durable receipt，主动退役 B cache 不触发重新训练；未封存 seed 仅跳过 artifact 哈希有效的已完成单元。失败保留缓存，不含 optimizer/scheduler/epoch 级训练恢复。删除仅限 receipt 列明且落在该 seed 允许根目录的 B tensor，保留 raw、prepared、source、checkpoint、小 manifest 和未知文件。

默认可再生成 B cache 的静态 FP32 峰值约 **124.76 MB（50k）/249.52 MB（100k）**，只占一个 seed；完成封存后释放。原 ST+MT 全量 B/Y cache 估算分别 2.807/5.614 GB，MT-only 全量 B/Y 则为1.404/2.807 GB；这是调度/缓存体积变化，不能解释为精度收益。checkpoint、prepared、IDs、序列化与 JSON 另占空间，`release_cache=false` 会累计各 seed B cache。详见 [资源预算与生命周期](resources_zh.md)。loader 固定 `num_workers=0`；未实测吞吐、峰值内存/显存、完整时间和真实磁盘需求。

输出在 `cross-domain/results/sci_qm7x/<experiment>/`，包括来源与划分 manifest、`upstream/`、保留小 manifest 的 `cache/`、`refine/`、`evaluation/seeds/`、`lifecycle/` receipts、`analysis/summary.json` 与中文 `analysis/report_zh.md`；调度日志在 `cross-domain/logs/sci_qm7x/<experiment>/`。大型产物不应纳入 Git。

## 验证记录与未验证边界

2026-10-22 已完成：

- v1 历史实现：Python AST、2 份 JSON 与 Bash 语法检查通过；以下 21 tests 是本轮资源改造前的历史记录。
- `bash -n cross-domain/experiments/sci_qm7x/scripts/run_cluster.sh` 通过。
- v1 历史 50k/100k dry-run 均为198单元、10上游、175 refine。v2 默认调整为138单元、5上游、125 refine；需以本轮验证记录为准。
- 根 agent 在 WSL `gn2_study_cross` 专用环境运行 v1 离线 contract tests：**21 passed in 6.64s**，含新增 opt-in 归档与手动解压分离检查；未执行优化器或真实数据训练。

可复现的离线命令（只用合成 HDF5，不访问网络、不执行优化器或真实数据训练）：

```bash
PYTHONDONTWRITEBYTECODE=1 python -B -m pytest -p no:cacheprovider \
  --basetemp /tmp/cross_qm7x_contract_20261022 \
  cross-domain/experiments/sci_qm7x/tests/test_contracts.py -q
```

测试覆盖官方字段/单位、母分子隔离与重复家族、构型上限/容量失败、A_train 标签标准化、B 禁止局部真值与 Y 显式诊断开关、split/tensor 身份损坏拒绝、ST/MT 结构与初始化匹配、padding/旋转/平移/排列行为、native 零残差、同容量槽位、分子内局部 shuffle、B_train mask 标准化、层次配对 ddof=1/缺单元失败、完整矩阵枚举，以及 dry-run 无 tensor/CUDA 导入和目录副作用。下载分支测试由 monkeypatch 写入合成字节，不发起实际网络请求。

**尚未验证：**真实 QM7-X HDF5 完整 schema/覆盖率与读取、完整 parent split 数量、真实数据训练或 cache/refine 全闭环、GPU 确定性与资源预算、多 GPU 调度故障恢复、5×5 科学结果及辅助任务是否产生正收益。离线 contract、文献核查和静态计划均不是科学有效性证据。

## v2 相对于 v1 的变动

默认 `cluster_50k.json` / `cluster_100k.json` 使用新身份 `qm7x_homo_local_v2_mt_resource_*`；新增显式 ST+MT v2 消融配置。旧配置原样保存在 `cluster_*_v1_st_mt_allcache.json`，旧全缓存入口保存在 `scripts/run_cluster_legacy.py`，它不承担资源峰值承诺。新资源配置使用共享生命周期入口，不能用旧 pool 或无筛选 `run.py --stage all`。样本规模、seed 数、训练预算、标签/任务、读出结构和 FP32 均未缩小。改变的是默认上游范围、缓存保留范围、执行顺序与默认预测保存策略。


### v2 资源改造验证

- WSL `gn2_study_cross`：本包离线合同测试 **26 passed in 47.00s**。全部使用合成HDF5/张量/手工checkpoint与完成标记，不执行fit、optimizer、真实训练或下载。
- 当前12个Python文件AST、6份JSON、`run_cluster.sh` Bash语法检查通过。
- 6份配置纯计划通过并断言无torch/h5py导入：默认50k/100k为138单元，显式ST消融与v1保留配置为198单元；两类入口分开使用。
- 新测试覆盖流式Y评分与同tensor缓存评分一致、B-only cache/明确拒绝Y缓存、Y不改B normalization、缺当前seed任一refiner marker阻止Y、合成完整seed封存后退役cache仍可验证/汇总、未知文件保留、少receipt拒绝总汇总、永久normalization损坏使receipt失效、MT-only不强行生成ST配对及无筛选stage拒绝。

以上是工程协议证据，不表示真实QM7-X读取、GPU运行、吞吐/峰值显存/磁盘测量或科学结果已经完成。

四份v2资源配置显式保留1 GiB输出磁盘余量（`workflow.free_space_reserve_bytes=1073741824`），共享driver在prepare前检查计划tensor峰值加余量；v1归档配置不变。余量用于估算外开销，实际资源峰值仍未测量。
