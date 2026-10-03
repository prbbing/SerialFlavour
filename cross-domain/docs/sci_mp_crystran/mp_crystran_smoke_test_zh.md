# MP＋CrystalTransformer 冻结 post-refinement 本地测试

日期：2026-10-03。工作树：`D:\hep_analysis\gn2_study\SerialFlavour-cross`，分支 `feat/cross-domain`。最终实验：`mp_crystran_smoke_cpu_v3`。

## 1. 完成情况与结论边界

已使用真实 Materials Project 结构和标签，在 WSL `gn2_study_cross` 环境的 CPU 上完成 `download → prepare → train → cache → refine → evaluate → analyze`，入口退出码为 0。12 项本实验科学契约测试和 37 项现有公共 pipeline / QM9 / NYUv2 回归测试通过，共 49 项。

本次只使用 1,024 个材料、一个上游 seed 和一个下游 seed，是工程闭环验证，不是 CrystalTransformer 论文成绩复现，也不是方法有效性的统计检验。相同 MT 架构的主任务监督对照优于 MT-native；辅助预测对同容量冻结读出的改善仅约 0.00018 eV，不能据此宣称辅助监督或 post-refinement 稳定有效。所有对照、预算与选择规则由配置固定，没有按 Y 结果重新调参。

## 2. 真实数据、下载体积与划分

使用 matminer 公开的 `mp_all_20181018` 固定快照，包含 83,989 条 MP 材料记录；主标签为 `gap pbe`（PBE-DFT 带隙，eV），辅助标签为 `e_form`（形成能，eV/atom）。读取松弛后的 `structure`，不用 `initial structure`、磁矩、模量或 e_hull 作为模型输入。[数据字段说明](https://hackingmaterials.lbl.gov/matminer/dataset_summary.html#mp-all-20181018)、[固定 metadata commit](https://github.com/hackingmaterials/matminer/blob/f89a530b76070fb613737b181381833330fb50c2/matminer/datasets/dataset_metadata.json)。

该快照日期不同于 CrystalTransformer 论文的 MP 2018-06-01 和 MP* 2023-06-23。本次不下载论文完整数据包、不使用作者预训练权重，也没有 MP API 密钥要求。原始数据许可/署名按 MP 的 CC BY 4.0 声明处理，引用 Jain et al., APL Materials 1, 011002 (2013)。[MP 数据许可及引用示例](https://doi.org/10.17188/1654373)。

下载位置：`D:\hep_analysis\gn2_study\dataset_ex\sci_mp_crystran\mp_all_20181018.json.gz`；WSL 对应 `/mnt/d/hep_analysis/gn2_study/dataset_ex/sci_mp_crystran/`。源文件 **184,077,884 bytes，约 175.55 MiB**；专用代码限制每个输入文件最多 500 MB，没有解压保存完整数据库。下载 SHA256 必须等于：

```text
1f3de2dc7c68959647240921b841293fce918ea1708e6f3760ba35a9cdfe0500
```

流式扫描全快照，只保留候选小池和最终训练子集。质量筛选要求主/辅标签有限、带隙非负、有序占位、合法原子序数和晶体几何；单个晶胞最多 32 个原子。此次原子数筛选排除 24,426 条，其余所列筛选没有发现需排除的记录。按 `SHA256(2026:mpid)` 取最小的 2,048 个候选，再按 material ID 去重和 pymatgen StructureMatcher 去除等价结构。匹配阈值 `ltol=0.2, stol=0.3, angle_tol=5°`，`primitive_cell=True, scale=True, attempt_supercell=False`，从候选池移除 2 个等价结构；从剩余 2,046 个候选取 1,024 个。该去重带有声明的容差，并非证明全数据库所有重复已消除。

随后用 split seed 2026 随机分区：

| 分区 | 材料数 | 用途 |
|---|---:|---|
| A_train | 512 | ST、同架构主任务对照、MT 上游训练；目标标准化拟合 |
| A_val | 64 | 仅按带隙 MAE 选择上游 checkpoint |
| B_train | 256 | 完整冻结后，仅用带隙标签训练读出；g 和额外通道标准化拟合 |
| B_val | 64 | 仅按带隙 MAE 选择读出；保留 native epoch 0 候选 |
| Y | 128 | 最终带隙评分与配对预测 |

split manifest 保存索引和 material ID，核验五个分区没有重叠。等价结构在分区前去重；本次未采用组成不重叠划分，不能把结果解释为跨组成外推。目标均值/标准差只从 A_train 拟合：Eg 为 `0.87294668 / 1.35062049 eV`，Ef 为 `-1.23023081 / 1.26439907 eV/atom`。任何读出都不使用辅助真值；B/Y 数据加载器拒绝请求辅助标签。

旧 pandas 快照在非必需字段中含裸 `NaN`，严格 ijson 不能直接解析。实验包提供流式 `NonfiniteJSONReader`，只将字符串外的 NaN/Infinity 替换为空值，保留字符串及跨 chunk 边界；不会把缺失标记填成数值零。四种 chunk 大小的兼容测试已通过。

## 3. 作者模型与本地适配

上游以 [CrystalTransformer 论文](https://www.nature.com/articles/s41467-025-56481-x) 和 [作者代码固定 commit](https://github.com/fduabinitio/ct-UAE/tree/0141ff9e09277d2229c9d7a24c1bcc5eac9de78e) 为依据。MIT 许可及未修改的 ST / MT@2p 源文件保留在实验包 `tests/reference/`；出处与适配说明见 [THIRD-PARTY.md](../../experiments/sci_mp_crystran/THIRD-PARTY.md)。

元素采用 100 维 one-hot，分别与笛卡尔坐标经线性层映射到 16 维后拼接。共享 Transformer 宽度 32，2 层、4 attention heads、FFN 宽度 64、dropout 0.1；保留作者默认 post-norm、ReLU 和首原子 token 材料表征 g。论文公开 MT 模型的宽度/层数/heads/FFN 为 256/8/8/512，本次仅缩小规模。导出完整逐原子 H、g 与预测，移除原代码不参与 forward 的 positional / coord_diff 参数，关闭 nested-tensor 加速以简化数值核验；padding 使用显式 mask。

| 上游 | 监督与性质头 | 总参数 |
|---|---|---:|
| `single_task` | 作者 ST 的 Linear(32,128) → Linear(128,1)，没有中间激活；只监督 Eg | 23,121 |
| `mt_main_only` | 与 MT 完全相同的两个 Linear-ReLU-Linear 头；仅 Eg 损失，Ef 头没有梯度 | 20,946 |
| `multi_task` | 作者 MT@2p：head1 为 Ef、head2 为 Eg；标准化 MSE 等权相加 | 20,946 |

`mt_main_only` 是同结构监督消融，不冒充作者 ST；它与 MT 初始化和总参数逐项一致。ST/MT 作者主头不同，单独比较二者不能严格隔离辅助监督效应。离线测试加载作者权重后，两种前向数值一致；另测主任务 loss 不向 Ef 头传梯度。

三种上游都复用公共 AdamW fit，lr `1e-3`、weight decay `1e-5`、batch 32、最多 20 epochs、patience 6、clip_grad 1、seed 1，只按 A_val 的 Eg MAE 选择。**这与作者 SGD＋StepLR 训练策略不同**；使用 AdamW 是本地小规模与现有通用训练循环的适配。此次不启用作者坐标扰动/旋转增强，保留发布晶胞和 site 顺序，ST/MT 使用相同数据与预算。

## 4. 完整冻结与下游输入

冻结整个上游编码器及全部任务头，`eval()` 关闭 dropout，所有参数 `requires_grad=False`。缓存包含 H、mask、g、native Eg、主标签、材料 ID；MT 额外缓存预测 Ef 和 Ef 头隐藏表示，没有辅助真值。缓存身份包括 checkpoint SHA256、数据/代码/配置身份；最终核验读出训练结束后上游 checkpoint 与缓存导出时 hash 相同。

每种读出共同访问完整 H，经可学习逐原子投影和 masked mean pooling，再与 g、native Eg 拼接。共同读取 native 预测，避免将主预测的可访问性差异误作辅助收益。输出为 `native Eg + residual`；残差最后一层初始化为零，native 在 epoch 0 就是可选择候选。仅按 B_val 选择，因此即使训练后的候选退化也可保留 native。

| 读出 | 一个额外标量通道 | 目的 |
|---|---|---|
| `embedding` | 恒为零 | 完整 H/g 的 embedding-only 基线 |
| `embedding_capacity` | g 的第一个通道复制 | 与辅助组相同输入宽度和参数量，且复制量不增加输入信息 |
| `embedding_aux` | 冻结预测 Ef | post-refinement 主比较 |
| `embedding_aux_shuffle` | 分区内固定打乱的预测 Ef | 按相同预算重新训练的诊断，不是仅在测试时破坏输入 |
| `embedding_hidden` | Ef 头隐藏表示的第一个通道 | 单标量容量匹配隐藏特征诊断；未检验完整隐藏向量的最优读出 |

所有读出有 10,561 个参数，具有相同 H 层级访问、宽度 64/32、初始化 seed 和优化预算；ST 与同架构主任务对照只训练前两种读出。g 和额外通道的均值/方差只在 B_train 拟合；H 保留冻结 Transformer 的表示，不再用其他分区拟合统计量。shuffle 的 B_train/B_val/Y 种子分别为 1001/2001/3001，预先固定且分别打乱。

下游使用 AdamW、lr `1e-3`、weight decay `1e-5`、batch 64、最多 40 epochs、patience 10、clip_grad 1、seed 1，仅优化 Eg 标准化 MSE，按 B_val 物理单位 MAE 选择。局部 H 没有逐原子任务真值监督；本例辅助输出仍只是材料级标量，不能宣称局部辅助任务效果。

## 5. 实测结果

最终 Y 指标，MAE/RMSE 单位均为 eV，越小越好。所有预测身份配对相同。

| 上游 | 读出 | MAE | RMSE | R² | 读出选择 epoch |
|---|---|---:|---:|---:|---:|
| ST | native | 0.690945 | 1.062638 | 0.253639 | 上游 17 |
| ST | embedding | 0.685343 | 1.052393 | 0.267961 | 2 |
| ST | embedding_capacity | 0.684906 | 1.051997 | 0.268511 | 2 |
| MT 架构，仅 Eg 监督 | native | 0.637973 | 1.020177 | 0.312093 | 上游 16 |
| MT 架构，仅 Eg 监督 | embedding | 0.637973 | 1.020177 | 0.312093 | 0 |
| MT 架构，仅 Eg 监督 | embedding_capacity | 0.637973 | 1.020177 | 0.312093 | 0 |
| MT | native | 0.689763 | 1.000696 | 0.338114 | 上游 19 |
| MT | embedding | 0.681536 | 0.996733 | 0.343346 | 2 |
| MT | embedding_capacity | 0.681462 | 0.996677 | 0.343420 | 2 |
| MT | embedding_aux | 0.681281 | 0.996680 | 0.343417 | 2 |
| MT | embedding_aux_shuffle | 0.681618 | 0.996621 | 0.343495 | 2 |
| MT | embedding_hidden | 0.681349 | 0.996618 | 0.343498 | 2 |

上游 A_val 最优 MAE：ST `0.625766`、同架构主任务 `0.611459`、MT `0.721418`，三者各训练 20 epochs。ST/MT 下游均在 epoch 2 选择，在 epoch 12 早停；同架构主任务读出选 epoch 0，在 epoch 10 早停。

本次观察：MT-native 对作者 ST-native 的 MAE 差为 `-0.001182 eV`，但对同架构主任务 native 为 **`+0.051791 eV`**，辅助监督没有在本次同结构对照中改善 Eg。MT embedding 对 MT-native 改善 `0.008228 eV`；MT aux 对容量匹配 embedding 仅改善 **`0.000181 eV`，约 0.0265%**，对普通 embedding 改善 `0.000254 eV`。shuffle 和单通道 hidden 的结果也很接近。本次观测主要显示小幅读出调整，不能确认辅助预测有可靠增量。只有单 seed，没有置信区间、SD 或显著性检验；数值差不能外推到完整规模。

辅助预测是冻结网络的确定性计算，不增加原始输入的 Shannon 信息。即使未来出现稳定收益，也应解释为有限样本/有限读出下的可学习表述优势，而非新物理信息。

## 6. 架构限制的真实审计

模型保持作者架构，没有新增周期邻居、距离编码、CLS token 或等变网络。虽然保存晶胞信息用于追溯和结构去重，**模型实际输入没有 lattice，也没有显式周期边界机制**。首原子 token 使材料预测可能依赖排序；原始笛卡尔坐标使平移/旋转不变性也无结构保证。

在方案锁定后，仅对 B_val 的前 16 个材料审计，未将扰动分数用于模型选择。反转原子顺序、平移 `[1,2,3] Å`、绕 z 轴旋转 90°，MT 带隙预测相对原输入的平均绝对变化分别为：

| 扰动 | 平均绝对预测变化 (eV) |
|---|---:|
| 原子顺序反转 | 0.239627 |
| 平移 | 0.328483 |
| 旋转 | 0.518069 |

这些变化明显大于本次 aux 读出的微小差值。它们是原架构在本次缩小模型和未增强设置下的实测限制，不是对论文完整训练的复现判断；对称性审计不用于选择正结果，也不因这一限制擅自换模型。完整规模前需要单独核查作者增强策略及晶胞/排序规范，并明确同一方法的训练和推理约定。

## 7. 运行证据、复现与代码边界

最终七阶段完成且各自 SHA256 校验通过，耗时分别为 download `2.55 s`、prepare `88.19 s`、train `10.50 s`、cache `3.22 s`、refine `6.96 s`、evaluate `2.05 s`、analyze `0.08 s`，总计约 `113.54 s`，不含首次网络下载、安装、离线测试和最终核验。真正的下载、训练、缓存、预测和指标已运行；不仅是静态检查。

环境：WSL2 Linux，Python 3.11.6，PyTorch `2.5.1+cu124`，NumPy `2.4.6`，CUDA 不可用，本次 CPU 4 线程；新增 pymatgen `2026.9.24`、ijson `3.5.0`，实际 pymatgen-core `2026.10.2`、spglib `2.7.0`。只安装到 `gn2_study_cross`，没有改 `gn2_study`。

从 `SerialFlavour-cross` 根目录：

```bash
conda activate gn2_study_cross
python -m pip install -r cross-domain/requirements.txt
bash cross-domain/experiments/sci_mp_crystran/scripts/run_smoke.sh
```

公共入口也可直接运行特定阶段：

```bash
python cross-domain/scripts/run.py \
  --config cross-domain/experiments/sci_mp_crystran/config/smoke.json --stage all
python cross-domain/experiments/sci_mp_crystran/scripts/verify_run.py \
  --config cross-domain/experiments/sci_mp_crystran/config/smoke.json
```

完成阶段重跑时会校验并跳过，不覆盖旧模型。修改代码或配置必须更换实验名。`v1` 保留 NaN 解析失败记录；`v2` 为补充同结构监督消融前的成功闭环；当前配置对应 `v3`。当前脚本与汇总面向单 seed smoke，完整数据/多 seed、`run_pool`、GPU/集群执行和论文完整预算尚未验证。

可检查的产物：

- [最终结果汇总](../../results/sci_mp_crystran/mp_crystran_smoke_cpu_v3/results_zh.md)、[逐对照指标](../../results/sci_mp_crystran/mp_crystran_smoke_cpu_v3/evaluation.json)、[配对预测](../../results/sci_mp_crystran/mp_crystran_smoke_cpu_v3/predictions.npz)。
- [阶段完成与 SHA256](../../results/sci_mp_crystran/mp_crystran_smoke_cpu_v3/stage_state.json)、[运行身份](../../results/sci_mp_crystran/mp_crystran_smoke_cpu_v3/run_manifest.json)、[材料划分](../../results/sci_mp_crystran/mp_crystran_smoke_cpu_v3/split_manifest.json)、[冻结与对称性核验](../../results/sci_mp_crystran/mp_crystran_smoke_cpu_v3/verification.json)。
- [文档证据快照](smoke_evidence.json) 包含最终退出码、测试数量、阶段状态、指标、核验结果、版本及主要产物 hash；原始数据、缓存、权重和预测位于 Git 忽略目录。

所有实现、config、专用 scripts 和 tests 均新增于 `experiments/sci_mp_crystran/`。公共 pipeline 与 scripts 没有改动。按用户明确授权，仅在公共 requirements 补充 pymatgen 和 ijson；按本次文档要求新增 `docs/sci_mp_crystran/`，没有改案例建议正文、QM9/NYUv2/NLP 实验代码，也没有 commit、push 或远程同步。