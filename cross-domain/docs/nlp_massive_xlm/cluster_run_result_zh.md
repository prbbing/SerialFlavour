# MASSIVE＋官方 XLM-R Base 集群运行结果（xlmr_base_en_us_full_v2）

更新：2026-10-05。本文记录 `xlmr_base_en_us_full_v2` 在本集群上的**完整实际运行**：prepare、10 次预训练微调上游、10 次缓存、125 次冻结读出、Y 评估与分层分析，共 148 个 work unit。此前 [CPU smoke](massive_xlm_smoke_test_zh.md) 使用随机初始化小 encoder，不能作为本轮预训练 Base 的证据。

## 1. 运行身份与产物

| 项目 | 值 |
|---|---|
| experiment | `xlmr_base_en_us_full_v2` |
| identity | `ecf84fc4425091bcbc543ed034cb8afdd3c3ac15a68c280e5bb01ab6afdd9a3f` |
| config SHA256 | `630a11e0a063c683d419b8cc85d8a76cfcd2ee86ec39f21607ac3f5770925a5f` |
| code SHA256 | `95c749d99cdc9c5b40d53b996f9c05560561d5f6988c4cc781c6d3acf368a073` |
| 结果目录 | `/data/yuyang/SerialFlavour/results/nlp_massive_xlm/xlmr_base_en_us_full_v2`（约 37 GB） |
| 日志目录 | `/data/yuyang/SerialFlavour/logs/nlp_massive_xlm/xlmr_base_en_us_full_v2` |
| 数据目录 | `/data/yuyang/SerialFlavour/local_data/nlp_massive_xlm` |
| 完成标记 | 148/148 unit `complete`，`cluster pool complete` |

关键产物：`run_manifest.json`、`resolved_config.json`、`download_manifest.json`、`evaluation.json`（135 行：10 native＋125 读出）、`summary.json`。

## 2. 环境与硬件

- 4× NVIDIA A10（23 GB，驱动 565.57／CUDA 12.7）；Linux 6.8.0。
- conda 环境 `gn2_study_cross`；Python 3.11.16，PyTorch 2.2.1+cu121，transformers 4.44.2，sentencepiece 0.2.1，NumPy 1.26.4，`cuda_available: true`。
- 上游：作者原版 `XLMRIntentClassSlotFill`＋公开基础 XLM-R Base（12 层/768/12 heads/FFN 3072）；AdamW 2e-5、microbatch 8×累积 4（有效 32）、FP32、encoder gradient checkpointing、最多 20 epochs。
- 实测显存约 6 GB/卡，**未出现 OOM**；A10 24 GB 余量充足。

## 3. 数据与权重获取（HF 不可达的替代）

本集群无法访问 HuggingFace（DNS 被污染，直连与常见镜像均超时）。改用可达来源并逐项核对固定 SHA256：

| 资产 | 来源 | 校验 |
|---|---|---|
| `model.safetensors`（1,115,567,652 B） | ModelScope `AI-ModelScope/xlm-roberta-base` | SHA256 `6fd4797b…` ✓ |
| `config.json` / `tokenizer.json` / `sentencepiece.bpe.model` | ModelScope | 与 pinned SHA256 一致 ✓ |
| `tokenizer_config.json` | `aifasthub.com` | SHA256 `994f4675…` ✓ |
| MASSIVE 1.0 归档（39,500,415 B） | 官方 S3（可达但限速，32 线程分块下载） | SHA256 `7df623fd…` ✓ |

放置到 `raw/` 与 `pretrained/xlm_roberta_base/` 后，`prepare_assets.py --verify-only` 全部通过。

划分（`official_grouped_full`，seed 20261003）：官方 en-US 原计 train 11514／dev 2033／test 2974；去重审计剔除 38 条跨分区记录（88 个重复组、7 个意图冲突组）。实际 A_train=8621、A_val=1029、B_train=2860、B_val=999、Y=2974（完整官方 test，未缩小）。

## 4. 运行前修复（重要）

首次以 `xlmr_base_en_us_full_v1` 运行时，所有上游在权重加载即失败：`ValueError: base encoder weight mismatch: [], []`（`model.py:74`）。

- 定位：代码假设官方 Base MLM 检查点**不含** sentence pooler，要求 `missing == {pooler.dense.weight, pooler.dense.bias}`。实测官方 `model.safetensors` 含 `roberta.pooler.dense.{weight,bias}`（204 个键），加载后 `missing=[]`，于是断言无条件失败。
- 修复：把判据改为“缺失项必须是 pooler 子集”即 `set(missing) ⊆ {pooler.dense.weight, pooler.dense.bias}` 且 `unexpected` 为空，兼容 pooler 存在或缺失两种检查点。pooler 不参与作者双头且被冻结（`requires_grad_(False)`），不影响读出。
- 因代码身份改变，新建实验名 `xlmr_base_en_us_full_v2`；v1 失败记录保留。

## 5. 结果

先在每个上游 seed 内平均 5 个下游 seed，再对 5 个上游均值报样本 SD（`ddof=1`）。Y 集 intent accuracy：

| 上游 | 读出 | Y accuracy mean | upstream SD |
|---|---|---:|---:|
| single_task | native | 0.877808 | 0.003060 |
| single_task | embedding | 0.877485 | 0.002902 |
| single_task | embedding_matched | 0.877606 | 0.003488 |
| multi_task | native | 0.880161 | 0.003983 |
| multi_task | embedding | 0.878386 | 0.003473 |
| multi_task | embedding_matched | 0.878467 | 0.003021 |
| multi_task | aux_prediction | 0.879193 | 0.003217 |

配对差值：

| 对比 | mean Δ | SD | 为正 |
|---|---:|---:|---:|
| MT aux_prediction − MT embedding_matched | +0.000726 | 0.002375 | 13/25 |
| MT native − ST native（按上游） | +0.002354 | 0.005958 | 3/5 |

- 上游最优 epoch：ST 约 10–17（早停于 15–20），MT 约 13–20（跑满 18–20）；A_val accuracy 平台约 0.89–0.90。
- `aux_prediction` 相对同容量 `embedding_matched` 平均高约 0.07 个百分点，但 seed 间符号不一致（13/25 为正，个别 −0.6 个百分点），量级与噪声同阶。
- MT native 平均略优于 ST native（+0.24 个百分点），但 5 个上游中仅 3 个为正，离散度（SD≈0.60）大于均值，方向不稳健。

## 6. 解读与限制

- 这是**预训练迁移条件下的二阶段读出**验证，不是 MASSIVE 原论文多语言成绩、128 次超参搜索或 8×V100 预算的复现。
- 未发现辅助预测相对强 embedding-only 的一致增益；极小差异不足以宣称机制性收益。MT 上游相较 ST 的均值优势不稳健。
- 辅助通道是冻结上游的确定性输出，**不是相对完整 H 的新增 Shannon 信息**。
- 统计口径：25 个下游组合不是 25 个独立上游重复；仅报告上游 seed 间均值与样本 SD，不做显著性检验。
- tokenizer 与公共 encoder 预训练语料可能接触 B/Y 文本，**未排除语料重叠**；本方案只控制任务微调、模型选择与二阶段标签泄漏。
- 全量 tokenizer 长度检查未单独执行；超长语句会明确失败而非静默截断。

## 7. 运行成本

| 阶段 | unit 数 | 合计时间 | 单 unit 最大 |
|---|---:|---:|---:|
| prepare | 1 | 0.2 min | 0.2 min |
| upstream | 10 | 450.3 min | 49.5 min |
| cache | 10 | 8.7 min | 0.9 min |
| refine | 125 | 83.5 min | 1.1 min |
| evaluate | 1 | 18.8 min | 18.8 min |
| analyze | 1 | ~0 | ~0 |

全部 unit 完成，0 失败；聚合 unit 时间约 9.36 h。unit 时间窗口 2026-10-04T18:51:13Z → 21:55:47Z，墙钟约 3 h 04 min（4×A10）。

## 8. 复现命令

```bash
cd /home/yuyang/SerialFlavour
PYTHON=/data/yuyang/miniconda3/envs/gn2_study_cross/bin/python \
GPU_POOL="0 1 2 3" RETRIES=1 \
  bash cross-domain/experiments/nlp_massive_xlm/scripts/run_cluster.sh
```

首次需先按第 3 节预置 `raw/`、`pretrained/xlm_roberta_base/` 并用 `prepare_assets.py --verify-only` 复核；同一命令重启会逐文件校验并跳过已完成单元（工作单元级续跑，非 optimizer/epoch 状态恢复）。配置与规模说明见 [cluster_agent_handoff_zh.md](cluster_agent_handoff_zh.md)。
