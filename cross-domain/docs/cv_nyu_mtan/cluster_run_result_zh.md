# NYUv2＋MTAN 集群运行结果（cluster_full_image_v2）

更新：2026-10-05。本文记录 `cluster_full_image_v2` 在本集群上的**完整实际运行**：prepare、10 次上游训练、10 次缓存、150 次冻结读出、Y 评估与分层分析，共 173 个 work unit。与 [集群 agent 操作说明](cluster_agent_handoff_zh.md) 的静态计划相比，本轮为真实 GPU 多 seed 执行证据；历史 CPU smoke 只对应各自的小规模配置。

## 1. 运行身份与产物

| 项目 | 值 |
|---|---|
| experiment | `cluster_full_image_v2` |
| identity | `77cc4e4c12805b7dd365a4339b355667538ac9415a32536b032937195ab8aae1` |
| config SHA256 | `42f15334362d352cf9363e080dda77388043c1627fad6dd96dd5861c352eab6e` |
| code SHA256 | `3767627634a372c4b2cb64307dfe68df3b7d13939c694297ddc27453111c7c88` |
| 结果目录 | `/data/yuyang/SerialFlavour/results/cv_nyu_mtan/cluster_full_image_v2`（约 45 GB） |
| 日志目录 | `/data/yuyang/SerialFlavour/logs/cv_nyu_mtan/cluster_full_image_v2` |
| 数据目录 | `/data/yuyang/SerialFlavour/local_data/cv_nyu_mtan` |
| 完成标记 | 173/173 unit `complete`，`cluster pool complete` |

关键产物：`run_manifest.json`、`resolved_config.json`、`evaluation.json`（160 行：10 native＋150 读出）、`summary.json`、`upstream/*/seed*/best.pt`、`cache/*/seed*/`、`refine/*/seed*/<recipe>/seed*/`。

## 2. 环境与硬件

- 4× NVIDIA A10（23 GB，驱动 565.57／CUDA 12.7）；Linux 6.8.0。
- conda 环境 `gn2_study_cross`；Python 3.11.16，PyTorch 2.2.1+cu121，NumPy 1.26.4，`cuda_available: true`。
- 调度：`GPU_POOL="0 1 2 3"`，每卡一个 `(variant, upstream_seed)` 完整生命周期。
- 缩小版 MTAN：宽度 `[16,32,64,128,128]`，128×160，无预训练、无增强；因 MaxUnpool 限制关闭强制 deterministic。

## 3. 数据获取与划分

- 数据来自固定 revision 的 `tanganke/nyuv2`（`b367b8b53c4dcefbcb4d9310b74976a63cc7f306`）。本集群无法访问 HuggingFace（DNS 被污染），改用可访问镜像 `aifasthub.com` 下载 14 个 parquet（8 train＋6 val，约 2.8 GB），逐个核对 pinned `lfs.oid`；shard 索引取自 `hf-mirror.net` API。
- 划分（`image_disjoint_exploratory`，锁定 seed 20261003）：A_train=477、A_val=80、B_train=159、B_val=79、Y=654。镜像无 scene 身份，**不能据此声称 scene-disjoint 正式结果**。
- 上游 seeds `17/23/31/43/59`，下游 seeds `29/37/47/61/71`；ST 只枚举 `embedding`、`embedding_matched`，MT 枚举 `embedding`、`embedding_matched`、`aux_prediction`、`aux_hidden`。

## 4. 运行前修复（重要）

首次以 `cluster_full_image_v1` 运行时，`refine:single_task:23:embedding_matched:29` 失败：`AssertionError: native initialization mismatch 0.00499`。

- 定位：`refine.py` 在 GPU 上校验“读出在 epoch 0 精确复现 native 预测”，判据 `max|Δ|≤1e-5`。CPU 上该复现为精确 0，但 GPU 上 64 通道卷积所选算法与 native（16 通道）不同，误差约 `4e-4`（合成测试），实际数据约 `5e-3`，超出阈值。这是**数值实现差异而非逻辑错误**，且会命中所有 64 通道 recipe（`embedding_matched` 及 MT 的辅助 recipe）。
- 修复：把 epoch-0 初始化的等价性校验改在 **CPU** 上完成（模型先在 CPU 构造并初始化，校验通过后再 `.to(device)` 训练）。CPU 上 32/64 通道复现误差均为 0，阈值保持 `1e-5` 不变。
- 因代码与配置身份改变，按规范新建实验名 `cluster_full_image_v2`；v1 的部分标记与产物保留作为失败记录。

## 5. 结果

先在每个上游 seed 内平均 5 个下游 seed，再对 5 个上游均值报样本 SD（`ddof=1`）。Y 集 mIoU：

| 上游 | 读出 | Y mIoU mean | upstream SD |
|---|---|---:|---:|
| single_task | native | 0.139112 | 0.008446 |
| single_task | embedding | 0.139364 | 0.008164 |
| single_task | embedding_matched | 0.139255 | 0.008213 |
| multi_task | native | 0.132604 | 0.005680 |
| multi_task | embedding | 0.132347 | 0.005635 |
| multi_task | embedding_matched | 0.132624 | 0.005691 |
| multi_task | aux_prediction | 0.132624 | 0.005691 |
| multi_task | aux_hidden | 0.132624 | 0.005691 |

- 冻结读出相对 native 的差异量级 ≤ 约 2.5e-4，方向不一致；MT 的 `aux_prediction`/`aux_hidden`/`embedding_matched` 在 5 个上游上给出**完全相同**的 Y mIoU（选择退化为同一解），未观察到辅助读出增量。
- 本缩小配置下 MT 上游 native（0.1326）**低于** ST（0.1391），与 MP 实验方向相反；上游最优 epoch 约 51–151（ST）／62–100（MT）。

## 6. 解读与限制

- 总体水平（Y mIoU≈0.13–0.14）很低，说明该缩小、无预训练的 MTAN 设置下语义分割能力有限；此结果**不能**外推到原论文的全宽度、预训练或增强配置。
- 未发现“冻结 embedding-only 或辅助读出带来稳定增益”的证据；即使存在差异也极小且在 seed 间不一致。
- 统计口径：25 个下游组合不是 25 个独立重复；只报告上游 seed 间均值与样本 SD，不做显著性检验。比较保持同 seed 配对。
- 协议为材料/图像级随机划分，镜像无 scene 身份；未核验数据的法向、深度物理尺度与许可字段。结论限定在本锁定划分与 5×5 seed 设置。

## 7. 运行成本

| 阶段 | unit 数 | 合计时间 | 单 unit 最大 |
|---|---:|---:|---:|
| prepare | 1 | 12.1 min | 12.1 min |
| upstream | 10 | 168.0 min | 24.1 min |
| cache | 10 | 7.3 min | 1.0 min |
| refine | 150 | 92.2 min | 1.6 min |
| evaluate | 1 | 23.2 min | 23.2 min |
| analyze | 1 | ~0 | ~0 |

全部 unit 完成，0 失败；聚合 unit 时间约 5.05 h。unit 时间窗口 2026-10-04T17:00:17Z → 18:48:48Z，墙钟约 1 h 49 min（4×A10）。

## 8. 复现命令

```bash
cd /home/yuyang/SerialFlavour
PYTHON=/data/yuyang/miniconda3/envs/gn2_study_cross/bin/python \
GPU_POOL="0 1 2 3" RETRIES=1 \
  bash cross-domain/experiments/cv_nyu_mtan/scripts/run_cluster.sh
```

同一命令重启会逐文件校验并跳过已完成单元（工作单元级续跑，非 optimizer/epoch 状态恢复）。首次需先按第 3 节用镜像预置 `raw/` 数据（含 `shard_index_<revision>.json`），再运行。静态计划与参数矩阵见 [cluster_agent_handoff_zh.md](cluster_agent_handoff_zh.md)。

## 9. 本地代码审查补充（2026-10-05）

本次检查对应 `refine.py` 将 native 初始化校验移到 CPU、显式按读出设备复制 native 参数的修改。审查发现一项尚未修复的 P2 风险：`initial_logits` 在 CPU 上计算，但比较目标 `validation['logits']` 仍是缓存阶段在运行设备上生成的预测；当缓存由 CUDA 生成时，该检查仍混用 CPU 与 GPU 的卷积结果。跨设备数值差异可能超过固定的 `1e-5` 阈值，从而误拒绝数学上等价的初始化。第 4 节记录的集群运行成功不能证明该判据在其他设备、后端或 checkpoint 上均可靠。

建议后续使用同一份缓存 `embedding` 中前 `width` 个通道（`semantic_hidden`），在 CPU 上用复制到 CPU 的 native head 重新计算参考 logits，与 CPU 读出结果比较；GPU 缓存 logits 的差异另行记录，区分初始化等价性与缓存跨设备数值一致性。本次仅记录建议，没有实施该修复，也没有更改历史实验身份或结果。

本地验证在 WSL `gn2_study_cross` 中执行两套相关离线协议测试：

```bash
python -B -m pytest -p no:cacheprovider --basetemp /tmp/cross_review_20261005 \
  cross-domain/experiments/cv_nyu_mtan/tests \
  cross-domain/experiments/nlp_massive_xlm/tests -q
```

结果为 **13 passed，2 个 SWIG 弃用警告，5.86 s**，`git diff --check` 通过。该环境为 PyTorch `2.5.1+cu124`，`torch.cuda.is_available() == False`；现有 NYUv2 初始化测试在 CPU 上生成参考预测，未覆盖 CUDA 缓存与 CPU 读出比较的路径。本次未下载数据、执行训练、连接集群或复核远程运行产物，因此此处属于本地代码审查与离线测试证据，不是新的 GPU 实验结果。
