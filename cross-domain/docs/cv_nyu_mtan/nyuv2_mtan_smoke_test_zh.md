# NYUv2＋MTAN：本地小规模 frozen post-refinement 测试

测试日期：2026-10-03。工作树：`D:\hep_analysis\gn2_study\SerialFlavour-cross`，分支 `feat/cross-domain`。

已经下载真实 NYUv2 分片，并在 WSL conda `gn2_study_cross` 的 CPU 上跑通 **download → prepare → train → cache → refine → evaluate → analyze** 七阶段。STAN、MTAN 和六个下游读出均实际完成训练与独立 Y 评价。该记录是工程闭环测试，不是论文成绩复现或方法有效性证明。

## 1. 实现位置与复用范围

配置入口：[smoke.json](../../config/cv_nyu_mtan/smoke.json)。环境增量依赖：[requirements-cv_nyu_mtan.txt](../../requirements-cv_nyu_mtan.txt)。

| 职责 | 模块 |
|---|---|
| 下载、标签读取、缩放、A/B/Y 划分 | `src/data/cv_nyu_mtan.py` |
| 五级 SegNet＋任务注意力、STAN/MTAN | `src/model/cv_nyu_mtan.py` |
| A 上游训练、A_val 选择 | `src/training/cv_nyu_mtan.py` |
| 冻结空间缓存、B 主任务读出训练 | `src/refine/cv_nyu_mtan.py` |
| 全数据集混淆矩阵、Y 最终评价 | `src/evaluate/cv_nyu_mtan.py` |
| 配对差值和 seed 层级汇总 | `src/analysis/cv_nyu_mtan.py` |
| 作者代码数值等价与协议测试 | `tests/cv_nyu_mtan/test_protocol.py` |

直接复用 `scripts/run.py`、通用阶段依赖/产物校验、`pipeline.fit.fit`、配置/代码身份、随机种子和运行环境工具。没有改动通用 pipeline、原 Jet tagging 代码或 QM9 模块。数据集专用模块也实现调度层的 variant/seed/recipe 过滤和 `applicable_recipes`；本轮实际验证的是单进程 `run.py`，尚未验证多 GPU pool。

配置、文档、测试参考代码的子目录均使用 `cv_nyu_mtan`；各领域模块按现有仓库约定使用同名 Python 文件。

## 2. 数据来源、下载规模与标签

[NYU Depth V2 官方主页](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html)；[MTAN 作者仓库](https://github.com/lorenmt/mtan)。作者提供 RGB、13 类分割、深度、预计算法向的预处理版本，但原预处理程序未发布。

本次检查作者 Dropbox 目录整包下载响应：`Content-Length = 8,975,242,476` 字节，且 Range 请求仍返回完整包。超过仓库单文件 5GB 限制，未保存整包。改用 [tanganke/nyuv2 分片镜像](https://huggingface.co/datasets/tanganke/nyuv2)，固定 revision：

```text
b367b8b53c4dcefbcb4d9310b74976a63cc7f306
```

镜像数据卡说明来源为 ForkMerge／Tsinghua Cloud，提供 795 train、654 val（官方 test）的三任务数据。这里只读取两个分片，未将 mirror `noise` 字段作为任务、输入或标签。

| 文件 | 字节数 | SHA256 |
|---|---:|---|
| `train-00000-of-00008.parquet` | 208,564,689 | `2c340d96fa2d9d225e39721805fa302f93cf62d4506bc60d438f3aa766d30bd7` |
| `val-00000-of-00006.parquet` | 224,448,496 | `74366586abc14e81dc1bdd759dd655be0af9c123d8fa3112181cb6f03bd0db14` |

总下载 433,013,185 字节，约 413MiB；每个文件均实际通过预先固定的大小和 SHA256 校验。原始文件、来源记录和处理数据位于：

```text
D:\hep_analysis\gn2_study\dataset_ex\cv_nyu_mtan\raw\
D:\hep_analysis\gn2_study\dataset_ex\cv_nyu_mtan\processed\smoke_cpu_v2\
```

原分片图像尺寸为 288×384。取 train 第一个分片前 40 行、val 第一个分片前 8 行，不根据模型分数选择样本。CPU 测试缩放到 64×96：RGB 双线性插值，分割／深度／法向最近邻插值，以避免把几何无效零值混入有效监督；有效法向再次单位化。不做增强或训练集统计归一化。

处理后的实测范围：RGB `[0,1]`；语义 `-1..12`，`-1` 为 ignore；深度 `[0,9.9152]`；法向分量约 `[-0.999985,0.999985]`。模型只输入 RGB。深度有效掩码为 `depth > 0`；法向有效掩码为非零向量。主／辅助损失分别按各自有效像素归约，无辅助有效标签时该任务贡献零损失。

镜像未提供 scene ID、法向生成版本或明确的 dataset license 字段；未完成与作者 Dropbox 的逐文件比对及深度物理尺度独立校准。记录镜像来源并遵循原 NYUv2 使用条款；不要把模型代码的 MIT 许可直接当作数据许可。正式发布前须进一步核查这些字段。

## 3. A／B／Y 协议

固定 split seed `20261003`。随机排列选定的 40 张 train 图像，划分为：

| 分区 | 图像数 | 实际用途 |
|---|---:|---|
| A_train | 24 | ST：分割；MT：分割＋深度＋法向 |
| A_val | 4 | 最大数据集级分割 mIoU 选择上游 |
| B_train | 8 | 冻结后仅分割监督训练 CNN 读出 |
| B_val | 4 | 最大分割 mIoU 选择读出，包括 epoch 0 native 候选 |
| Y_test | 8 | 选定所有模型后最终评价；来自官方测试池的 val 分片 |

保存 row ID、原 RGB SHA256、split IDs、数据与划分 SHA256。实际检查 48 张图像的 RGB 内容无完全重复，所有分区索引互斥。这里实现的是 **image-disjoint smoke**：同场景不同帧的相关性未受控，不能称为 scene-disjoint 正式实验。代码拒绝使用其他 split mode 冒充场景划分。

ST 与 MT 使用同一 A、种子、训练预算和验证主指标。不使用预训练。B 的 dataset batch 只提供语义标签；缓存保存 shared／semantic hidden、native logits 和冻结辅助预测／辅助 hidden，**不保存深度或法向真值**。Y 标签只用于最终评分，没有参与拟合、checkpoint 选择或归一化。

## 4. 上游架构与训练

参考 [MTAN（CVPR 2019）](https://openaccess.thecvf.com/content_CVPR_2019/html/Liu_End-To-End_Multi-Task_Learning_With_Attention_CVPR_2019_paper.html) 及作者 `model_segnet_mtan.py`／`model_segnet_stan.py`，固定参考 commit：

```text
c36c30baa18968dec74fe9039abcfd4f132edfa1
```

保留五级 SegNet 编码／解码、MaxPool 索引和 MaxUnpool、共享 attention transition、各任务 attention mask，以及 `3×3 Conv → 1×1 Conv` 预测头；原宽度 `[64,128,256,512,512]` 缩小为 `[8,16,32,32,32]`。分割输出 13 类 logits，深度一通道，法向三通道单位向量。

ST 使用作者 STAN 的单任务 attention 结构，只实例化语义注意力和预测头，读取分割监督。MT 实例化三个已发表任务；等权目标为分割 CE＋深度有效像素 L1＋法向有效像素 `1−cos`。不用新的上游辅助任务。法向归一化增加 epsilon，以防零范数数值错误；分割返回 logits，由 CE 内部做 log-softmax。

| 上游 | 实测参数量 | epoch 上限 | A_val 最佳 epoch | A_val mIoU（%） |
|---|---:|---:|---:|---:|
| STAN | 230,421 | 3 | 3 | 3.7109 |
| MTAN | 269,673 | 3 | 3 | 2.1933 |

上游 seed `17`，batch 2，学习率 `1e-4`，weight decay 0，梯度裁剪 5。通用 fit 使用 AdamW；在 weight decay 为零时采用其无权重衰减设置。保留等 epoch 预算，但 MT 参数和计算量较大，不称为同计算量对照。

本轮没有复现作者 200 epochs、原始宽度和分辨率；也没有启用 uncertainty weighting、DWA 或超参数搜索。作者原表格与现行数据集混淆矩阵评价不同，不能直接比较其 mIoU。

## 5. 冻结空间特征与下游对照

输出均保持 64×96，避免辅助空间图只与全局池化 MLP 比较。基本 embedding 为 `semantic attention hidden（8通道）＋shared decoder hidden（8通道）`。

| recipe | 输入 | 通道 | 读出参数 | ST/MT |
|---|---|---:|---:|---|
| `embedding` | 完整局部 embedding | 16 | 2,541 | 两者 |
| `embedding_matched` | 重复完整 embedding 至固定通道预算 | 32 | 9,677 | 两者 |
| `aux_prediction` | embedding＋冻结深度＋冻结法向＋embedding 通道补齐 | 32 | 9,677 | MT |
| `aux_hidden` | embedding＋深度／法向任务 hidden | 32 | 9,677 | MT |

重复／补齐使用真实 embedding 通道，避免仅靠零输入通道填充参数。重复不增加信息；这是固定深度、空间层级和名义参数数量的容量对照。所有读出均使用作者卷积预测头类型，无全局池化、无几何真值输入；native logits 保存作诊断但不作为读出输入。

将 native 主头映射进扩大后的 CNN，使额外输入初始不影响输出。六组在 B_val 的初始化 logits 与对应 frozen native logits **最大绝对误差均为 0.0**；epoch 0 作为可选 checkpoint 保留。额外 hidden 通道仍有可训练卷积，不在训练中强制零化。

下游 seed `29`，batch 2，最多 4 epochs，学习率 `1e-3`，weight decay 0，梯度裁剪 5；所有组使用相同 B 和 B_val 选择规则。上游 `eval()`、`requires_grad_(False)`，缓存使用 inference mode；实际逐 tensor 检查包括 BatchNorm buffers 的 state_dict 完全不变，并核验训练 B 前后上游 checkpoint SHA256 不变。加载缓存／读出时核对代码、配置、数据、checkpoint 和 cache manifest 身份。

## 6. Y 上的实际结果

mIoU 由全部 Y 图像的有效语义像素累积 **13×13 混淆矩阵**计算，对 union 非零的类取平均；union 为零的类排除。Pixel Accuracy 也按全部有效像素计算。本次共有 42,500 个有效语义像素；具体 `classes_with_union` 随模型预测变化，逐模型保存在 evaluation 中。

| 上游 | 读出 | Y mIoU（%） | Y Pixel Accuracy（%） |
|---|---|---:|---:|
| STAN | native | 3.4536 | 18.6988 |
| STAN | embedding | 3.6573 | 26.6235 |
| STAN | embedding_matched | 3.8973 | 30.9012 |
| MTAN | native | 2.7826 | 9.6541 |
| MTAN | embedding | 3.6203 | 39.7388 |
| MTAN | embedding_matched | 3.6159 | 39.7741 |
| MTAN | aux_prediction | 3.6194 | 39.7506 |
| MTAN | aux_hidden | 3.6164 | 39.7671 |

MT-native 比 ST-native 低约 **0.6710 个 mIoU 百分点**；本轮未观察到上游辅助监督正迁移。MT 的 aux_prediction 相对 embedding_matched 仅高 **0.00348 个百分点**，aux_hidden 仅高 **0.00049 个百分点**。在 24 张 A_train、3 epochs 和单次 seed 条件下，这种差异不足以支持辅助预测带来有效增量的结论。部分读出高于 native，只能记录为此次读出／优化行为。

MT-native 辅助诊断：深度 AbsErr `2.37010`（保留镜像单位，不宣称独立校准后的米）、RelErr `0.80499`；法向平均角误差 `84.8003°`、中位角误差 `83.9001°`。两类几何标签各有 43,562 个有效像素。辅助任务远未收敛，极低分辨率和首次分片子集均限制了结果解释。

这里只运行 **1 个上游 seed × 1 个下游 seed**，不报告 SD、置信区间或显著性。分析接口在多 seed 时先在同一 upstream seed 内平均 downstream 结果，再对 upstream means 计算 `ddof=1` 样本 SD；本次 SD 为 null，不把一次运行当作零方差。

## 7. 运行环境、验证与产物

实测环境：WSL2 Linux 6.6.87.2；Python 3.11.6；PyTorch 2.5.1+cu124；NumPy 2.4.6；PyArrow 23.0.1；requests 2.34.2。`torch.cuda.is_available()` 为 false；CPU threads 4。

首次 `smoke_cpu_v1` 在 train 开始时报错：PyTorch MaxUnpool2d 没有强制 deterministic 对应实现。未完成训练的 v1 记录保留。最终配置改为 `deterministic=false`、保留所有随机种子，使用新实验名 `smoke_cpu_v2`，没有覆盖失败证据。固定种子不等于严格位级确定性。

实际验证：

- 新增 4 项离线测试通过：缩小相同宽度后，STAN/MTAN 与固定作者模型逐层复制参数，输出数值等价；native 初始化、同容量、冻结参数／buffers；混淆矩阵和 ignore 像素语义。
- 全部 `python -m pytest cross-domain/tests -q`：**32 passed in 11.38s**，含既有 QM9／通用 pipeline 测试。
- 七阶段全部 complete，checkpoint、缓存、评价及 summary 实际存在并被 SHA256 记录。
- 再次执行相同 `--stage all`，七阶段均通过产物校验并 skip，没有重新选择或评分 Y。

| 阶段 | pipeline 实测耗时（秒） |
|---|---:|
| download（下载已预取，大小与 SHA256 再核验） | 5.17 |
| prepare | 12.96 |
| train | 6.62 |
| cache | 5.54 |
| refine | 8.69 |
| evaluate | 3.04 |
| analyze | 0.13 |

合计约 42.15 秒，不含首次网络下载、PyArrow 安装、单元测试和首次失败尝试。v2 实验输出约 42.95MB（十进制），数据与权重没有纳入 Git。

```text
cross-domain/results/cv_nyu_mtan/smoke_cpu_v2/
  run_manifest.json / resolved_config.json / stage_state.json
  upstream/{single_task,multi_task}/seed17/
  cache/{single_task,multi_task}/seed17/
  refine/<variant>/seed17/<recipe>/seed29/
  evaluation.json / summary.json
```

每个训练目录包含 `best.pt`、history JSON/CSV、training manifest。缓存每个 B/Y 分区保存空间 tensors、row IDs 和 manifest。小型证据快照：[smoke_evidence.json](smoke_evidence.json)，不包含图像、权重或空间特征。

最终 run identity：`9290733f2a8be86770190bbf6417ec9e3558d8aab4799482f365ce49bfb199ef`。

## 8. 重跑命令与下一阶段

```bash
source /home/yuyang/miniconda3/etc/profile.d/conda.sh
conda activate gn2_study_cross
cd /mnt/d/hep_analysis/gn2_study/SerialFlavour-cross
python -m pip install -r cross-domain/requirements-cv_nyu_mtan.txt
python cross-domain/scripts/run.py \
  --config cross-domain/config/cv_nyu_mtan/smoke.json --stage all
python -m pytest cross-domain/tests -q
```

只运行某阶段可改 `--stage`，其前置阶段必须已完成且产物身份有效。更改源码／配置／运行预算时须用新 experiment 名称；通用 runner 拒绝覆盖不同身份的已有实验。本次 downloader 固定下载两个分片，prepare 样本请求不能超过已下载分片数量；它不是完整 795/654 数据的正式规模下载器。

正式实验应先取得包含 scene 身份、明确法向生成与类别映射的完整版本，扩展下载／manifest，并实现 scene 分组，而非将本次 image split 简单放大。恢复更合理分辨率、训练预算和辅助任务质量，统一 ST/MT 初始化与选择预算，再做 5 个上游 × 5 个下游 seeds 的配对实验。只在 A_val／B_val 选择方案，锁定 Y；同时保留无正迁移和下游无增量的结果。

本轮没有测试 GPU 运行、正式 scene split、完整数据、收敛后的论文架构成绩、shuffle/oracle 诊断或多 seed 效应。已有用户候选说明文档保持原内容，新增实现和实测记录独立放在本目录。
