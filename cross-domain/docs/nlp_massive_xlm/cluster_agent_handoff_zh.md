# MASSIVE＋官方 XLM-R Base：远程集群使用说明

更新：2026-10-03。本文件配套 `cross-domain/experiments/nlp_massive_xlm/config/cluster_full.json` 与 `scripts/run_cluster.sh`。本轮只编写代码、配置和说明，并完成只读静态检查；**没有连接远程、同步文件、下载 Base 权重或执行数据准备、训练、缓存及评估**。

下文命令供远程操作者执行。默认直接运行完整矩阵，没有 pilot、子矩阵或预先试跑流程。此前 [CPU smoke 结果](massive_xlm_smoke_test_zh.md) 使用随机初始化的小 encoder，不能作为预训练 Base 的运行验收或收益证据。

## 实验定位与已确认硬件边界

本实验是预训练迁移学习条件下的单语言 post-refinement 验证，不以复现 MASSIVE 原论文的多语言成绩、128 次搜索或 8×V100 预算为目标。每个上游 seed 生命周期限定 **单张 NVIDIA A10、24 GB 显存**；GPU_POOL 中多张卡仅并行不同 seed，不使用 DDP。

用户已确认 **5×5 矩阵保持**，本轮仅补充说明：当前配置仍为 microbatch 8、累积 4、有效 batch 32、FP32、gradient checkpointing；4×8 只是尚未采纳的建议。当前 8×4 是否满足 A10 峰值显存、Base 加载与长度 128 是否覆盖全量数据，均尚未实测。详细参数量、与 Jet 122k 的差异、样本限制和核心增量判据见 [实验特殊条件与结论边界](experiment_scope_zh.md)。本轮文档更新不改变源码/配置指纹，不需要另换实验身份。

## 1. 配置与规模

| 项目 | 默认配置 |
|---|---|
| 数据 | MASSIVE 1.0，en-US；完整官方 train/dev 经去重后分 A/B，完整官方 test 为 Y |
| 上游 | 作者原版 `XLMRIntentClassSlotFill`，公共基础预训练 XLM-R Base encoder；12 层、hidden 768、12 attention heads、FFN 3072 |
| pooling / heads | masked mean intent pooling；作者已有 Linear–GELU＋Linear 双头；60 intent / 56 slot 类型含 Other |
| ST | 只优化 intent CE，slot head 冻结；共享 encoder 在 A 微调 |
| MT | intent CE＋first-subword slot CE，slot 权重固定 1；共享 encoder 在 A 微调 |
| A/B train | 官方 train 去重后按源语句/重复文本组、意图分层，约 75% / 25% |
| A/B val | 官方 dev 去重后约 50% / 50% |
| Y | 完整官方 test **2,974 条**，不按类别均衡抽样、不缩小测试集 |
| 上游 seeds | ST/MT 各 `[17,23,41,59,73]`，共 10 个上游模型 |
| 下游 seeds | 每个上游使用 `[29,31,43,61,79]` |
| 下游读出 | ST：embedding、embedding_matched；MT：embedding、embedding_matched、aux_prediction |
| 训练总量 | **10 上游＋125 下游**；10 次 cache；最终 10 native＋125 readout 共 135 行 Y 结果 |
| work units | prepare 1＋upstream 10＋cache 10＋refine 125＋evaluate 1＋analyze 1＝**148** |
| experiment | `xlmr_base_en_us_full_v1`，与历史 `smoke_cpu_v1` 隔离 |
| split seed | 20261003，所有训练 seed 共享同一划分 |

默认数据路径为 `../dataset_ex/nlp_massive_xlm`，相对**仓库根目录**解析，避免携带本机 `/mnt/d/` 路径。结果与日志分别位于 `cross-domain/results/nlp_massive_xlm/<experiment>/`、`cross-domain/logs/nlp_massive_xlm/<experiment>/`。实际服务器路径由操作者确认；脚本从自身位置定位仓库。

如果目录位置、训练设置或源代码改变，复制配置、改 `experiment` 为新名字再运行。不要改原 manifest 来绕过身份校验，也不要在同一实验运行期间覆盖专用代码或配置。

## 2. 实验协议与实现适配

数据去重沿用 CPU 案例：官方源 ID 与 NFKC/大小写折叠/空白归一化后的文本组成绑定组，跨官方分区按 test > dev > train 保留。保护对象是完整官方 Y；完全重复的开发池记录被剔除，同组不得跨 A/B/Y。意图冲突重复组也整体绑定，不拆开。完整模式按每个意图组的记录数比例划分，容许整组导致轻微比例偏离；全部去重后 train/dev 都被使用，没有先抽样成均衡小池。

此前全 en-US 审计发现 88 个重复组、7 个意图冲突组并剔除 38 条跨分区记录；这是历史数据审计。新的全量 A/B 实际数量、类别分布与源 ID 以本轮 `split_manifest.json` 为准，本次未运行全量 prepare，不补写假定数量。

上游默认 **AdamW 2e-5**、weight decay 0.01（bias / LayerNorm weight 为 0），最多 20 epochs、patience 5、clip 1、10% optimizer update warmup 后线性下降。microbatch 8、累积 4 次，名义有效 batch 32；末尾不足一个窗口时按实际样本数归约。precision 为 float32，开启 encoder gradient checkpointing，未实现 AMP。按 A_val intent accuracy 选择，accuracy 同分时比较较低 intent NLL，两者都同分保留较早 checkpoint。

这是本项目的预训练微调配置，不是论文的 128 次超参数搜索或论文最佳配置。原建议中的 Adam 由 AdamW 实现替代，此处明确记录；学习率、warmup 等也属于本项目建议。累计窗口内 slot CE 是按样本数加权的各 microbatch 有效词 CE 均值，不声称与一次完整 batch 的按所有有效词加权 CE 数值严格相等。

NLP 专用 `finetune.py` 补充公共 fit 当前没有的梯度累积、warmup/linear decay 与 intent NLL 同分选择。阶段调度、通用 work unit、身份和工件记录仍复用公共 pipeline；B 读出继续使用公共 `pipeline.fit.fit`。公共 pipeline、CV/QM9 和 Jet tagging 代码没有为本案例改变。

上游全部冻结后缓存 B/Y 完整 token H、native intent logits、mask、主任务标签；MT 额外缓存 slot logits。禁止把 `annot_utt`、真实槽位、scenario 等当作模型输入。B 优化和早停只读 intent 标签；缓存没有辅助真值。下游 masked mean/max token 读出和原 logits residual 沿用 smoke；epoch 0 精确恢复 native，并允许 B_val 选择。

下游固定 lr 1e-3、batch 64、width 32、最多 100 epochs、patience 15，AdamW wd 0.01、clip 1。本配置**没有学习率搜索**；不能根据 Y 在 1e-3 / 3e-4 间挑选。B 同分当前保留最早 checkpoint，没有 NLL 同分打破。H-only 输入 768 维，matched H / H＋slot 输入同为 824 维，两组参数量和训练预算相同；额外通道使用相同文本词首 mask。

sequence 上限 128，保留首子词槽位监督、忽略续子词/特殊 token/padding。超长语句会明确失败，**不会静默截断 Y 或删掉部分 Y**。若实际遇到超长数据，应修改新配置的长度并使用新实验名，再重新准备；本次未执行 tokenizer 全量长度检查。

## 3. 部署文件与环境

首次部署需同版本公共 `cross-domain/pipeline/`、`cross-domain/scripts/`、`cross-domain/experiments/__init__.py`、统一 `requirements.txt`，以及完整 `cross-domain/experiments/nlp_massive_xlm/`。作者 `official_xlmr.py`、LICENSE/NOTICE/THIRD-PARTY 一并保留。说明可同步 `cross-domain/docs/nlp_massive_xlm/`。

如果服务器已有当前公共 pipeline，仅定向更新 NLP 实验包、对应说明和所需统一依赖即可；不要覆盖正在运行的其他实验、公共模块或其数据目录。运行身份会纳入 NLP 包内所有根级 Python 和专用 Python/Bash 脚本；新增 `finetune.py` 与集群入口也属于指纹。同步前由操作者确认目标和正在运行的实验，本次没有代为同步。

在远程 Linux 使用专用 `gn2_study_cross` 环境。需要 Python 3.11、与远程 CUDA/驱动匹配的 PyTorch（参考历史环境 2.5.1）、NumPy，以及统一增量依赖中的 `transformers==4.44.2`、`sentencepiece==0.2.1`、requests；safetensors 随 Transformers 安装。**统一 requirements 不安装 PyTorch 或决定其 CUDA build**，应使用集群已有匹配的基础环境；不改 Jet 环境 `gn2_study`。

```bash
# 先将两个路径替换成远程机器实际路径。
REPO_DIR=/path/to/SerialFlavour
CONDA_ROOT=/path/to/miniconda3
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate gn2_study_cross
cd "$REPO_DIR"
python -m pip install -r cross-domain/requirements.txt
```

## 4. 基础模型与数据资产：联网机器准备，计算节点离线加载

计算节点不必访问 GitHub/Hugging Face。`allow_download=false`：缺失资产直接报错，不能退回随机 encoder；已有文件逐一核对固定 SHA256。基础模型固定为 [FacebookAI/xlm-roberta-base](https://huggingface.co/FacebookAI/xlm-roberta-base) revision `e73636d4f797dec63c3081bb6ed5c7b0bb3f2089`，不是完整 MASSIVE 微调 checkpoint。

| 资产 | 大小 / 校验 |
|---|---|
| MASSIVE 1.0 归档 | 39,500,415 bytes；SHA256 `7df623fd2d300a4d235d6ee5bd396c9a28258d3a0ccb29abdb054506eba153f8` |
| `model.safetensors` | **1,115,567,652 bytes**，约 1.12 GB；SHA256 `6fd4797bc397c3b8b55d6bb5740366b57e6a3ce91c04c77f22aafc0c128e6feb` |
| Base `config.json` | SHA256 `d66ed8cd4f2a93b358c245e50736fa389ed4f35c0bae7aad0b32abb20c62b579` |
| tokenizer JSON / SentencePiece | 约 9.10 MB / 5.07 MB；固定校验值已写入 data.py |

权重大小与 SHA256 来自 [固定 revision 的官方文件元数据](https://huggingface.co/api/models/FacebookAI/xlm-roberta-base/tree/e73636d4f797dec63c3081bb6ed5c7b0bb3f2089?expand=true)，本轮只读取了该元数据，没有下载 1.12 GB 文件。资产准备器限制权重文件最多 2 GB，其他文件最多 100 MB（Base config 更小），都低于本地单文件 5 GB 限制。

在有网络且能保存资产的机器，进入同版本仓库和 cross 环境后执行以下命令；`--data-root` 指向计划传输的目录。它只准备原始归档、tokenizer 与预训练文件，不初始化实验、不训练。

```bash
python cross-domain/experiments/nlp_massive_xlm/scripts/prepare_assets.py \
  --config cross-domain/experiments/nlp_massive_xlm/config/cluster_full.json \
  --data-root /path/to/staging/nlp_massive_xlm
```

将准备完成的 `raw/` 与 `pretrained/` 两个目录传至远程解析后的 data_root，结构为：

```text
nlp_massive_xlm/
├── raw/
│   ├── amazon-massive-dataset-1.0.tar.gz
│   └── tokenizer/
│       ├── config.json
│       ├── tokenizer.json
│       ├── tokenizer_config.json
│       └── sentencepiece.bpe.model
└── pretrained/xlm_roberta_base/
    ├── config.json
    └── model.safetensors
```

远程 prepare 从已校验的归档只提取 en-US，不依赖数据下载库执行远程脚本。`prepared_assets` 不要求预先带 `en-US.jsonl`，后者会由 prepare 单元的 download 子阶段生成。

远程只读验证资产可执行：

```bash
python cross-domain/experiments/nlp_massive_xlm/scripts/prepare_assets.py \
  --config cross-domain/experiments/nlp_massive_xlm/config/cluster_full.json --verify-only
```

模型加载只读取 safetensors 中 `roberta.*` 的基础 encoder 参数，不加载 MLM head、不用任何 MASSIVE 意图/槽位任务权重。加载前检查隐藏维度、层数、head 数、FFN、词表、LayerNorm epsilon（**1e-5**）、位置/类型 embedding 和 token ID 等与固定基础 config 一致，不忽略尺寸错误。基础 MLM 没有 sentence pooler 权重；作者双头只用 token H，该未使用 pooler 保持随机并冻结。encoder 必须全部匹配，仅允许这个未使用 pooler 的两项参数缺失。旧版本可能保存 position_ids/token_type_ids 两个派生 buffer；兼容层先核对其值与 4.44.2 重建值完全相等再去除，不忽略任何已学习 encoder 参数。

tokenizer 与公共 encoder 的预训练语料可能接触 B/Y 文本，不能宣称已经排除语料重叠；本方案控制的是任务 fine-tuning、模型选择与二阶段标签泄漏。

## 5. 完整矩阵运行入口

默认是**单节点多 GPU seed 队列**，每张卡同一时刻一个完整 `(variant, upstream seed)` 生命周期，不是 DDP，也不把单个模型跨卡切分。多节点或 Slurm 分区参数依具体集群另配，本脚本没有假定分区名或生成 batch job。

在已获分配的 GPU 上运行，只读列出完整矩阵：

```bash
GPU_POOL="0 1 2 3" \
  bash cross-domain/experiments/nlp_massive_xlm/scripts/run_cluster.sh --dry-run
```

dry-run 不验证实际 CUDA、读取 Base 权重、创建输出/日志目录、下载或训练；它只读取配置、源指纹与枚举 148 单元。它不是必须先跑的 pilot。

正式前台运行：

```bash
GPU_POOL="0 1 2 3" RETRIES=1 \
  bash cross-domain/experiments/nlp_massive_xlm/scripts/run_cluster.sh
```

后台执行示例（操作目录已有写权限）：

```bash
mkdir -p cross-domain/logs/nlp_massive_xlm
nohup env GPU_POOL="0 1 2 3" RETRIES=1 \
  bash cross-domain/experiments/nlp_massive_xlm/scripts/run_cluster.sh \
  > cross-domain/logs/nlp_massive_xlm/xlmr_base_en_us_full_v1.launch.log 2>&1 &
```

操作者用已分配给任务的实际 GPU ID 替换示例。入口支持 `CONFIG`（配置路径）、`GPU_POOL`（空格分隔数字 GPU ID）、`PYTHON`（Python 可执行文件）、`CONDA_ENV`（可选激活环境）、`RETRIES`（额外重试次数）。例：

```bash
CONFIG=cross-domain/experiments/nlp_massive_xlm/config/cluster_full.json \
PYTHON=/path/to/miniconda3/envs/gn2_study_cross/bin/python \
GPU_POOL="0 1" RETRIES=1 \
  bash cross-domain/experiments/nlp_massive_xlm/scripts/run_cluster.sh
```

若已有数字形式的 `CUDA_VISIBLE_DEVICES`，未指定 `GPU_POOL` 时自动沿用它；显式请求超出已有分配的 GPU 会报错，不覆盖分配边界。当前 physical-ID 队列不支持 UUID/MIG 格式的 GPU 列表。不要在同一 experiment 同时启动两个 pool；文件锁会拒绝第二个。一个完整 seed 一张卡、一个 pool 一台节点。

流程是：串行 `prepare` → 在空闲 GPU 分配完整 ST/MT seed 生命周期（上游、cache、该 seed 所有 recipe×下游 seed）→ 所有生命周期成功后执行一次 `evaluate` 和 `analyze`。先跑全部模型再看 Y，不根据早到的 Y 修改其他 seed。

## 6. 续跑、失败与查看结果

相同代码/配置再次运行同一命令，会验证 identity 与每个已完成单元工件 SHA256 后跳过。不会仅因 `best.pt` 存在就认为完成。任意 seed 失败时，其余生命周期可完成，但最终返回非零，**不执行 Y evaluation**。`RETRIES=1` 表示最多 2 次尝试。

上游、cache、refine 单元的失败重试会从该单元起点重跑；**没有 optimizer/scheduler 的 epoch 中间恢复**。不能把 validation best checkpoint 当成优化器续训状态。若上游或 cache 必须重建、但其下游 marker 已存在，入口要求换 experiment 保存旧证据；prepare 工件坏了且训练 marker 已存在也同样拒绝混用。

已成功 evaluate 的矩阵视为不可变：续跑先复核所有前置单元和 evaluation，必要时只补 analyze，不重复计算 Y。如果这些前置工件变了，则拒绝续跑，要求新实验身份。

结果文件：

```text
cross-domain/results/nlp_massive_xlm/xlmr_base_en_us_full_v1/
├── run_manifest.json / resolved_config.json / download_manifest.json
├── upstream/{single_task,multi_task}/seed*/
│   └── best.pt / training_manifest.json / history.json / history.csv
├── cache/{single_task,multi_task}/seed*/
│   └── b_train.pt / b_val.pt / y_test.pt / manifest.json
├── refine/<variant>/seed<upstream>/<recipe>/seed<downstream>/
├── evaluation.json
└── summary.json
```

划分与处理输入在 `<data_root>/processed/<experiment>/`，其中 `split_manifest.json` 记录完整分组和实际数目。日志在 `cross-domain/logs/nlp_massive_xlm/<experiment>/`：`units/*.json` 保存状态和 artifact 校验值，`unit_output/*.attempt*.log` 保存单元输出，`<variant>__seed<seed>.pool.log` 保存生命周期日志。**集群入口使用 unit markers，不使用本地 run.py 的 stage_state.json**；不要用后者判断此任务是否完成。

查看故障与完成状态：

```bash
tail -n 80 cross-domain/logs/nlp_massive_xlm/xlmr_base_en_us_full_v1.launch.log
cat cross-domain/logs/nlp_massive_xlm/xlmr_base_en_us_full_v1/units/evaluate.json
cat cross-domain/logs/nlp_massive_xlm/xlmr_base_en_us_full_v1/units/analyze.json
```

验收需同时有 pool 正常退出、148 个单元的完整校验记录、135 行完整 Y 结果，以及 summary。评价 accuracy / macro-F1-60 / intent NLL，MT native 补 slot span micro-F1 与 semantic-frame accuracy。总结 MT-native−ST-native、MT-aux−同容量 MT embedding_matched；完整 Y 的差值不能和历史均衡小 Y 的差值混算。下游 seed 在每个上游 seed 内先平均，再对 5 个上游均值报样本 SD（ddof=1）；25 个组合不是 25 个独立上游重复。

## 7. 资源预算与未验证范围

Base＋作者双头约 2.79 亿参数（架构计数），远大于本地 1612 万的小模型。已确认每个并发 seed 只能使用一张 **A10 24 GB 显存 GPU**，不跨卡切分。若同时分配 4 张卡，主机 RAM 建议 **64 GB**，这属于估算而非已确认资源；**没有 A10 峰值显存实测或墙钟 ETA，不能保证当前配置已经适配通过**。FP32 MT 参数、梯度与两个 AdamW moment 约 4.15 GiB，另有激活、缓存及 allocator；实际峰值取决于硬件和 PyTorch。现有代码未自动采集 CUDA 峰值显存，正式执行时另行记录。

缓存使用 FP32、固定 padding 到 128：H 每条约 `128×768×4 = 393,216 bytes`，Y 的 H 约 1.17 GB；MT slot logits 每条另约 28.7 KB。按推荐 A/B 比例，10 个上游的 B/Y cache 总计量级约 28 GB，10 份上游 best checkpoint 约 11 GB，另有预训练资产、临时 checkpoint 和其他产物。建议为本实验预留 **100 GB 可用磁盘**；逐项估算不是已生成文件的统计。单个 Y cache 预计约 1.25 GB，未默认生成大于 5 GB 的单文件。

当前 cache 在主机内存中先收集再拼接，B 读出也将完整 split 载入主机内存，且续跑会重算 SHA256；多卡并发会放大 RAM 与共享存储 I/O。若内存紧张，减少 `GPU_POOL` 并发卡数；这不改变 seed 矩阵。若 GPU OOM，应在**新配置、新 experiment**中统一调整 ST/MT microbatch 与累积次数（如 4×8，仍有效 batch 32），不能在原身份下静默改预算。

本轮验证为 Python AST 解析、JSON 配置契约、完整矩阵计数、Bash 语法与入口只读 dry-run；记录见 [cluster_static_checks.json](cluster_static_checks.json)。**未执行本轮 pytest、Base 加载、全量 prepare、NLP 微调循环、CUDA、GPU 调度或远程运行**。历史 46 tests passed 属于 CPU smoke 代码的验证，不能直接算作本轮新增路径已运行通过。当前 source 与专用脚本改变了指纹，原 smoke_cpu_v1 证据完整保留；当前 smoke.json 已换为 smoke_cpu_v2，尚未执行。

正式来源与模型 API： [MASSIVE ACL 2023](https://aclanthology.org/2023.acl-long.235/)、[固定作者模型实现](https://github.com/alexa/massive/blob/f966f21846043aabef9b0f974fa7970027f43738/src/massive/models/xlmr_ic_sf.py)、[Transformers 4.44.2 模型加载文档](https://huggingface.co/docs/transformers/v4.44.2/en/main_classes/model)。代码、权重、数据与配置的来源分别记录，公共预训练不冒充任务微调复现。
