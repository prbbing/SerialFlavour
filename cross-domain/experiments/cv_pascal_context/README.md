# PASCAL-Context 21 类 × HRNet-W18 / MTI-Net 的冻结 post-refinement

本包实现 `download → prepare → train → cache → refine → evaluate → analyze`，默认由共享 `pipeline.lifecycle` 按上游seed顺序调度，训练单元仍通过 `run_unit.py`。目标是检验作者多尺度任务交互之后，冻结辅助预测是否仍能改善独立 B 上的语义读出。本包实现并不构成实测收益，不能据此宣称新架构、正迁移、显著性或机制。

## 任务、来源与版本

主任务为包含背景的 **21 类 VOC 语义分割**，void=255。采用作者 `PASCAL_MT` 数据包；不是 59 类 PASCAL-Context 版本。四个辅助任务是 7 类人体部件（6 部件＋背景）、单通道 edge、单通道 saliency 和 3 通道 surface normals。辅助任务真值仅供 A_train 的 MT 损失。主任务、类别数、路径及掩码规则已对照作者实际代码核查：

- [ECCV 2020 MTI-Net 论文](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123490511.pdf)。
- [作者仓库，固定 commit ed3d9ac1c35493d8d22c4e33574919c326d77b6d](https://github.com/SimonVandenhende/Multi-Task-Learning-PyTorch/tree/ed3d9ac1c35493d8d22c4e33574919c326d77b6d)。
- [作者 PASCAL / hrnet18 / mti_net.yml](https://github.com/SimonVandenhende/Multi-Task-Learning-PyTorch/blob/ed3d9ac1c35493d8d22c4e33574919c326d77b6d/configs/pascal/hrnet18/mti_net.yml)。
- [作者 PASCAL 数据加载器](https://github.com/SimonVandenhende/Multi-Task-Learning-PyTorch/blob/ed3d9ac1c35493d8d22c4e33574919c326d77b6d/data/pascal_context.py)。
- [PASCAL_MT.tgz 原始来源](https://data.vision.ee.ethz.ch/kmaninis/share/MTL/PASCAL_MT.tgz)。

`vendor/author/` 保存上述固定提交的 17 个小型原样源码、配置、类别字典和许可证文件。`source_lock.json` 逐文件记录 URL、大小及 SHA256；根 `source.py` 又硬编码该 lock 的 SHA256，纳入公共代码指纹，运行及跳过完成单元之前均校验。网络代码取自固定提交，导入包、构造模型和只读 dry-run 都不会联网。作者源码采用 CC BY-NC 4.0，HRNet 文件含 Microsoft MIT 标注，ResNet 文件注明 Torchvision 来源；保留所有原始头注释与完整仓库 LICENSE。数据、原图和 ImageNet 权重各自的使用条款需要单独遵守，仓库许可不自动覆盖这些资源。

`source.py` 在独立模块命名空间执行原样源码，避免污染其他实验的 `models` / `data` 模块。只在内存中改写内部绝对 import、NumPy 旧标量别名和 Torchvision 迁移后的 URL-loader import；`PROJECT_ROOT_DIR` 路径服务指向 vendor；缺少可选 `termcolor` 时仅关闭彩色文本。这些改动没有替换网络层、宽度或数值前向。原样 vendor 文件未被修改。包内 `.gitattributes` 禁止 Git 对 vendor 与 source lock 做行尾转换，避免 Windows/Linux checkout 破坏逐字节校验。

saliency 与 normals 是作者 bundle 的 **teacher 蒸馏伪标签**，不是人工直接测量的两个独立标签。saliency 使用作者 `>0.5` 阈值；normals 从 RGB 编码映射到 [-1,1]，作者用 Context/NYU 共有类别加 `tvmonitor` 决定有效区域。边缘来自 Context `LabelMap` 的 Laplacian＋thin，因此与语义标注相关。当前 bundle 没有本项目核实过的 teacher checkpoint/hash 或完整训练暴露清单；不能承诺教师没有见过 Y。ImageNet 预训练也不构成排除图像内容暴露的证据。

## 数据与 A/B/Y

在本地提供解压后的作者数据，默认路径为工作树下 `cross-domain/local_data/cv_pascal_context/PASCAL_MT/`：

```text
PASCAL_MT/
  JPEGImages/<id>.jpg
  ImageSets/Context/{train,val}.txt
  semseg/VOC12/<id>.png                 # 若存在，优先使用
  semseg/pascal-context/<id>.png        # VOC12 不存在时使用
  pascal-context/trainval/<id>.mat      # LabelMap，edge / normals 有效区域
  human_parts/<id>.mat                 # anno，人类类别 15，作者6部件映射
  normals_distill/<id>.png
  sal_distill/<id>.png
```

默认严格核对官方 train=4,998、val=5,105 个 ID。4,998 个 train 内部按固定 seed=2718、分组隔离分为 A_train=3,000、B_train=1,000、共享 val=500，另498张记录为unused、不参与训练/选模/评估；`data.shared_validation=true`，A_val 与 B_val 使用完全相同、顺序一致的这500个ID。两个名称分别表示上游与下游选模用途，不是两份独立样本，计算唯一样本数时只计一次。官方 val 全部锁定为 Y=5,105。数量不同、ID 重复、标签不属于 0..20/255 或官方 train/val 内容重复时失败；不静默截断或混入另一版本。若来源实际 ID 数不同，先核查来源，随后显式改配置并更换 experiment 名。

prepare 审计每个文件 SHA256、JPEG 字节重复与解码 RGB 重复，并把重复图片合为分组；可提供覆盖全部 ID 的 `data.group_map` JSON，把可靠的来源/序列关系一起并入 union groups。train/Y 分组交叉直接拒绝；train 内分组保持完整，A_train、B_train、共享val之间不交叉，只有A_val/B_val按设计完全重合；精确数量无法满足则拒绝。默认审计不能排除视觉近重复或没有元数据的场景关联，不能称为已认证的场景级隔离。prepare 会保存实际 ID、分组、重复原因、标签来源和所有输入哈希；读取样本时核查所用文件。

大文件默认本地提供，`allow_download=false`，`allow_extract=false`。`download` 阶段首先校验本地已解压 schema，不自动获取数据或权重。若经审核手动启用 archive 路径，下载只允许固定作者 URL，并执行文件大小上限、可选预期哈希、流式下载和逐成员安全解压；拒绝 `..`、绝对/Windows 路径、symlink/hardlink/device、覆盖文件、成员数及解压体积超限。默认单 archive 和单成员上限均为 5 GB，expanded 总量上限为 40 GB；真实包若超限，应先按仓库约定取得大文件授权后调整配置。本轮没有下载该包。

## 上游与公平对照

所有模型是作者原宽度 HRNet-W18-small-v2，四尺度通道 `[18,36,72,144]`。**必须本地提供 ImageNet 预训练 `hrnet_w18_small_model_v2.pth`，并填写已审核的 `model.pretrained.sha256`。** 初始配置的 hash=null 是需要用户提供本地资源身份的显式占位；缺文件、未填 hash、hash 不匹配或编码器权重覆盖不完整都会清楚失败，不退回随机权重，不在运行时隐式下载。正式实验不能使用已在 PASCAL 上微调的发布多任务 checkpoint。

默认仅 `multi_task`；`single_task` 和 `main_only_matched` 是单独可选扩展，完整三variant配置保留在 `config/cluster_three_variant.json`，使用独立experiment身份：

| variant | 作者网络 | A_train 监督 |
|---|---|---|
| single_task | 作者 SingleTaskModel：HRNet＋语义 HighResolutionHead | 仅最终语义 CE |
| main_only_matched | 完整五分支 MTINet，保留交互及所有输出 | 最终与四尺度语义 CE；辅助损失关闭 |
| multi_task | 完整五分支 MTINet | 作者最终＋四尺度全部任务损失 |

同一个 upstream seed 下先构造相同 backbone 和语义 head，再构造其他层，因此 ST/MT 的共同初始权重一致；main_only_matched / multi_task 的全网络初始状态一致。manifest 保存相应初始张量摘要。ST 与 MTI 架构不同，二者差异不能单独归因为辅助监督；`multi_task - main_only_matched` 是同结构主对照。main_only_matched 的未直接监督辅助输出只是机制/容量诊断，不能解释成已经学会辅助任务。

A 训练采用作者 Adam、1e-4、weight decay 1e-4、poly exponent=0.9、最多 100 epochs，含 epoch 0；配置默认 batch=8、不早停，每 epoch 仅 A_val 的 **21 类 dataset mIoU** 选模。多任务权重沿作者配置：semseg=1、human_parts=2、sal=5、normals=10、edge=50；含四尺度 deep supervision。人体部件无标注样本全255；全 void 任务安全贡献0。normals 使用作者 normalized L1，有效掩码只用于 A 的标签损失；缩放前的原始有效区域按 nearest resize，避免 cubic 将无标签边界误当有效像素。

本项目相对作者原训练流程的明确变动：严格 A/B/Y 数据拆分、仅主任务 A_val 选择、逐 epoch 验证和 native epoch0；512×512 固定 resize＋A_train 水平翻转，省略作者随机旋转/随机尺度。保留 ImageNet RGB normalization。这里是作者架构上的本项目协议实现，不声称逐字复现论文训练结果。

## 冻结缓存与 B 读出

选定上游后 `requires_grad_(False)`、`eval()`，缓存前后逐张量摘要相等，包括 BN buffers；B 不持有可训练上游。默认仅B_train/B_val逐图保存，特征保持float32；标签无损uint8，读取恢复long；Y采用流式评估，GPU 只持有一个 cache batch，不把全数据加载 RAM。每图记录 ID、split、SHA256 与字节数：

- 完整四尺度 backbone＋主任务 distillation 空间 H；没有全局池化或挑单层。
- 主任务原始 FP32、上采样前 native logits（512 输入时为128×128）、RGB输出尺寸和主真值。推理时在当前设备用作者相同 bilinear／align_corners=False 精确重建512×512 native；不是降低输入分辨率或量化。cache 发布时检查重建与原作者输出逐元素相等。
- MTI 四尺度全部辅助分支 distilled hidden。
- 四尺度 initial 辅助预测以及相同尺度的 final 辅助预测表示（parts softmax、sal/edge sigmoid、normal unit vectors），不保存下游未使用的原分辨率 final 辅助输出、final semseg 副本或 raw deep-supervision 张量；四尺度读出输入完整保留。

B/Y loader 完全不读取辅助真值；cache 不含辅助真值和 truth-derived mask。尤其不能用 GT-derived normals 有效类别或人体部件有效标记给辅助预测置零、裁通道或选择样本。上游输出以原模型前向产生，所有辅助输入只来自 RGB。

所有 B 读出是相同四尺度卷积 residual：每尺度 1×1→3×3，再3×3 fusion→21输出；各尺度均访问完整 H，native logits 作为固定加法基底。末层零初始化，使 **所有 recipes epoch0 精确等于同一上游 native logits**。native 与 residual 组在当前设备调用同一个重建函数；相同后端的重建与作者前向一致已在离线检查验证，GPU路径尚未验证。训练默认 AdamW、lr=1e-4、最多100 epochs、patience=15、batch=4；reduce-on-plateau patience=5；只用 B_train 主语义 CE，含 epoch0，以 B_val mIoU 选择。native 及上游 BN 均在缓存阶段锁定，不会因通用 fit 的 `model.train()` 更新。

| recipe | 额外输入 | 解释 |
|---|---|---|
| embedding_matched | 重复真实完整 H，填满相同通道预算 | 强 embedding-only、同参数量/层级/空间卷积范围基线 |
| aux_prediction | initial/final 辅助预测 | 检验显式冻结预测增量 |
| aux_hidden | 全辅助 distilled hidden | 检验预测表示是否优于任务 hidden 访问 |
| aux_shuffle | 同 partition 其他图像的完整空间辅助预测 | 固定 derangement，无自配对，不跨 B_train/B_val/Y |

所有 recipe 的输入 slot 数、trainable 参数、native 基底和训练预算完全相同；不足的通道通过重复真实输入填满，不用无效全零容量。ST 只适用 embedding_matched；两个 MTI variants 使用全部4种。shuffle seed 对每 upstream/downstream/split 固定，完整图像交换保留各任务空间分布，但切断与本图像关系；它不是只打乱像素的局部平滑对照。

## 完整矩阵、统计与资源

默认完整配置是 `config/cluster_full.json`，仅multi_task，上游 seeds=`[11,22,33,44,55]`、下游 seeds=`[101,202,303,404,505]`。四recipes、5×5 seeds、100 epochs与原patience均保留：**5上游训练、5 B-cache、100下游训练、105 Y行、113公共单元**。完整三variant扩展为15上游、15cache、225下游、240 Y行、258公共单元。两配置有独立v4_sharedval500身份，不复用旧v1/v2/v3标记，也不增加pilot。

driver先prepare，逐上游seed训练、仅B缓存、完成全部recipe×downstream seed，然后独立进程评估该seed锁定Y。触Y之前验证完整本seed的B单元及checkpoint/cache provenance。第一遍每batch上游一次前向，native和全部非shuffle读出共享瞬时完整H/hidden/辅助预测；第二遍每batch上游一次，只评估shuffle。native每上游只计一次；读出留在CPU，每次只将当前一个读出移入GPU，不按每读出重复上游前向。

首遍只spool每图四尺度辅助预测，原始tensor约10.66GB/MTI上游；不写Y完整H、hidden或主任务高分辨率logits。第二遍严格复用旧路线 `shuffle_seed+upstream_seed*1009+downstream_seed*10007+3` 的整图Y donor derangement，各次读取校验hash与ID。donor不跨split，不使用辅助真值或truth mask。

21类dataset confusion matrix含背景，void255忽略；永久保存mIoU、pixel accuracy、class IoU及有效像素。默认 `evaluation.save_predictions=false`，开启时仍用每32图uint8压缩NPZ，全部预测分片及manifest纳入永久artifact。固定512×512评估不等同作者原图尺寸/多尺度推理指标。

driver核验全部永久checkpoint/history/training_manifest、小cache/spool manifest、seed评估与可选预测的hash后，封存 `lifecycle/<variant>/seed<seed>/receipt.json`；随后仅删除receipt列出的可重建B-cache tensor与Y donor spool，再运行下一seed。保留checkpoint、manifest、receipt与lineage；未知文件不递归删除。清理中断可按receipt续做；有效receipt的seed跳过Y和训练。

**早期seed的Y结果不得用于修改后续seed、recipe、训练预算或停止规则。** 完整计划预先固定；最终evaluate只聚合全部有效receipt的seed结果，不要求已清理tensor存在。缺失/重复/陈旧seed阻止最终汇总。analysis先在每个上游seed内平均下游，再对上游均值报告mean与sample SD(ddof=1)，native每上游只计一次。MT-only不生成ST/main-only对比；三variant扩展才报告这些对比。不把25单元当25个独立上游重复，不自动声明显著性或因果机制。

只读 `workflow.plan(context)` 不导入torch，不创建目录。默认MT每seed B-cache约**26.77GB**、Y donor spool含5%序列化额度约**11.20GB**，同时驻留峰值约**37.97GB**，另加20GB reserve、永久checkpoint/logs、原始输入和权重。五seed B-cache累计写入约133.86GB是总I/O，不是驻留峰值；关闭 `workflow.release_cache` 时plan按全部cache/spool累积计算峰值。开启预测时额外计入完整矩阵未压缩uint8上界，不预先假设压缩收益。保留FP32、512分辨率、full H与完整训练矩阵。默认只有1个active上游，多GPU仅轮换，不自动并发；真实GPU显存/吞吐/时间与存储实测尚未验证。

旧全cache评估仍保留为可选代码路径：需独立experiment、关闭workflow seed_lifecycle模式、将cache.splits显式设为B_train/B_val/Y，再使用旧通用阶段入口。默认launcher使用新生命周期。

## 环境与命令

从 `D:/hep_analysis/gn2_study/SerialFlavour-cross` 或 Linux 对应工作树根执行，环境用独立 `gn2_study_cross`。依赖统一位于 `cross-domain/requirements.txt`，不另建实验 requirements；需要 torch/numpy、Pillow、scipy、PyYAML、opencv-python-headless、scikit-image、six，不需要 torchvision/easydict/yacs，termcolor是可选文本着色。以下是后续用户主动运行的命令，本轮没有安装依赖：

```bash
conda activate gn2_study_cross
python -m pip install -r cross-domain/requirements.txt
# 先提供本地PASCAL_MT与ImageNet权重；在配置中填实际路径和预训练SHA256。
# 配置改变后须使用新的experiment名称，不覆盖已有证据。
sha256sum /path/to/hrnet_w18_small_model_v2.pth

# 只读计划：不创建产物目录、不导入模型、不配置CUDA、不触发下载。
python cross-domain/experiments/cv_pascal_context/scripts/run_cluster.py \
  --config cross-domain/experiments/cv_pascal_context/config/cluster_full.json \
  --gpus 0 1 --dry-run

# 用户授权GPU完整执行后：不需要先跑pilot。
GPU_POOL="0 1" bash cross-domain/experiments/cv_pascal_context/scripts/run_cluster.sh

# 单unit用于显式恢复/排查；完整默认实验使用上述生命周期launcher。
python cross-domain/scripts/run_unit.py \
  --config cross-domain/experiments/cv_pascal_context/config/cluster_full.json \
  --unit refine:multi_task:11:aux_prediction:101

# 离线工程测试，无真实数据/权重获取或训练。
python -m pytest cross-domain/experiments/cv_pascal_context/tests -q
```

wrapper直接转发共享 `pipeline.lifecycle.main`，不再monkeypatch旧pool；各训练单元仍通过公共run_unit。`CONFIG`、`GPU_POOL`、`PYTHON`、`CONDA_ENV`、`RETRIES`受支持。只读dry-run跨平台，无模型/CUDA/目录创建。旧run_seed入口显式拒绝单seed调度，提示使用完整lifecycle，防止跳过封存与安全清理。

续跑按每unit identity＋文件大小/SHA256校验，完整单元跳过；前置无效且已有依赖证据时要求新experiment，不擦除历史。失败单元重跑不是optimizer/scheduler状态恢复。完成过的Y只验证并跳过，不重新用Y选方案。包根代码、配置、公共pipeline和固定vendor来源改变都会通过identity或显式source lock检查被拦截。运行同一个实验期间不要覆盖其代码。

主要产物为 `prepared/{acquisition,manifest}.json`、各upstream的best/history/training_manifest、小cache manifest、每seed评估与lifecycle receipt（tensor已按receipt清理）、各readout的best/history/training_manifest、可选分片Y预测、`evaluation.json`、`summary.json/csv`，均在独立 `results/cv_pascal_context/<experiment>/`；unit markers/logs在 `logs/cv_pascal_context/<experiment>/`。不纳入源码提交。

## 验证边界（本次交付）

已进行固定源码/论文/数据schema联网核查，小型作者源码vendor及逐文件hash记录；包内Python AST、JSON和只读完整矩阵dry-run验证。2026-10-22 在 WSL `gn2_study_cross` 运行离线 contract tests，缓存优化后的19项检查在5.94秒通过；其中真实作者ST和两个MTI变体的小tensor前向，以及低分辨率native重建逐元素相等检查均通过。随后加入磁盘缓存→读出→指标集成检查与存储预算检查，最终 **21 passed in 6.79s**。离线 tests 覆盖完整矩阵、真实作者网络随机权重小tensor前向、ST/MT共同初始化、native零残差、容量、shuffle、tar安全、合成数据schema与分组、59类拒绝和Y完整性守卫；未把测试文件存在当作通过；实际命令为 `PYTHONDONTWRITEBYTECODE=1 python -B -m pytest -p no:cacheprovider --basetemp /tmp/cross_pascal_contract_20261022 cross-domain/experiments/cv_pascal_context/tests/test_contracts.py -q`。

尚未下载真实PASCAL或预训练权重、未安装依赖、未执行数据加载/完整权重导入、未训练任何模型、未运行GPU/完整矩阵、未评估真实Y、未远程运行、未提交或推送。合成/随机权重前向仅是离线工程证据，不能证明预训练加载成功、真实标签MAT兼容、缓存预算实测、GPU可承载、方法收益或论文可复现性能。

### v2资源改动离线验证（历史：当时为分开的A_val/B_val）

新增合成检查覆盖MT默认/三variant扩展矩阵、plan无torch导入、流式features与旧cache逐元素一致、两遍Y全部9个合成方法行与旧cached mIoU/confusion/class metrics完全一致、各ds donor derangement、预测开关、B-only缓存与uint8标签，以及峰值/累计空间区别。没有真实数据/权重下载、真实训练、GPU运行或真实Y指标。2026-10-22在WSL `gn2_study_cross` 的最终本包验证为 **31 passed in 8.52s**；Windows共享生命周期dry-run通过，输出113单元、105行Y及37.934GB峰值。


每个seed的receipt还封存稳定的全局谱系：`run_manifest.json`、`resolved_config.json`、`prepared/manifest.json`（原始输入hash/精确划分/分组）、`prepared/acquisition.json` 与 `prepared/source.json`（固定源码来源及完整source lock快照）。这些文件不参与cache清理；会随seed/最终汇总变化的evaluation、summary及lifecycle complete不放入单seed永久清单。


GPU参数使用已有 `CUDA_VISIBLE_DEVICES` 分配中的**逻辑编号**。例如 `CUDA_VISIBLE_DEVICES=4,5` 时默认 `GPU_POOL=0` 只使用物理GPU4；显式 `GPU_POOL="0 1"` 才在物理GPU4/5间轮换，仍只有一个active seed。缺省不会把物理token当作逻辑编号。

### v3：恢复共享验证集（历史：998张）

按用户要求，A_val/B_val改用同一份998张验证图片，原先A_train=3000、B_train=1000、官方Y=5105及训练/seed矩阵保持不变。两份配置均启用 `data.shared_validation=true`，并使用新的 `v3_sharedval` experiment身份。这里合并的是原先两个验证划分的样本预算；重新按固定seed进行分组划分，不宣称旧A/B逐ID成员保持不变。没有改动其他数据集或共享pipeline。

prepare只分配一次共享val，再将相同有序ID用于A_val、B_val；manifest显式记录共享策略和unique_train_images=4998。恢复/加载时拒绝验证策略变化或A_val/B_val的ID/顺序不一致。辅助真值仍仅用于A_train；共享val两阶段均只用主任务标签选模，Y始终独立且不参与选择。B缓存只保存B_train和B_val，不另存A_val副本。

只读计划确认113公共单元、105行Y保持；B缓存预算从1498张增至1998张，每seed约35.66GB，Y donor spool约11.20GB，临时驻留峰值约46.86GB（含5%序列化额度），另加20GB reserve、永久产物、原始数据和权重。上述为空间估算，不是实测。

2026-10-22在WSL `gn2_study_cross` 对PASCAL本包与共享生命周期合并执行离线检查：**49 passed in 10.71s**（PASCAL38、共享11）。新增检查覆盖共享val同序ID、完整重复组隔离、唯一样本只计一次、错误共享计数拒绝、prepare及两个验证loader角色一致、损坏验证ID阻止恢复，以及两份配置的缓存预算。命令：

```bash
PYTHONDONTWRITEBYTECODE=1 python -B -m pytest cross-domain/experiments/cv_pascal_context/tests cross-domain/tests/test_lifecycle.py -q -p no:cacheprovider --import-mode=importlib --basetemp /tmp/pascal_sharedval_20261022
```

未下载真实数据或权重、未执行真实训练/GPU评估、未运行QM9测试、未提交或推送；旧完成标记和实际运行产物不作改写。

### v4：共享val固定为500张，其余保持

按用户要求，两份配置将共享val缩为500张；A_train仍3000、B_train仍1000、Y仍5105，模型、训练参数、seed矩阵、recipe和资源工作流不变。`unused_train_count=498`显式记录停用预算，manifest保留这498张的具体ID；它们不进入任何训练、选模、缓存或评估loader。实际参与的官方train图片为4500张，另498张停用，原始官方train计数仍严格核对4998。

分组算法先按原A_train=3000、B_train=1000、验证池=998及相同split_seed分配，再只将该验证池按split_seed+1分成500张共享val和498张unused。这样保留v3的A_train/B_train/Y逐ID成员和顺序，验证集是原998张的子集；重复/来源分组在这次子划分中也保持完整，无法满足精确数量则拒绝，不拆开组。manifest的unused IDs与所有活动划分不相交，加载时检查这一约束。

新身份为 `v4_sharedval500`，保留旧完成记录。当前默认只读计划仍为113单元、105行Y，B缓存1500张约26.77GB，临时峰值约37.97GB，另加20GB reserve及原始数据/权重/永久产物；均为空间估算。

2026-10-22在WSL `gn2_study_cross` 对当前PASCAL本包与共享生命周期离线复验：**51 passed in 10.79s**（PASCAL40、共享11）。新增两项覆盖val缩减前后A/B/Y逐ID及顺序保持、验证子集与unused的完整分组隔离、unused manifest记录及loader拒绝。沿用上面的pytest命令，将basetemp改为 `/tmp/pascal_val500_20261022`。未下载、真实训练、GPU/远程运行、提交或推送。
