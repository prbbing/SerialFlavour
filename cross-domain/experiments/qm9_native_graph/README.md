# QM9 gap：native ＋图残差核心框架

本包是新身份 `qm9_native_graph / qm9_gap_native_graph_nested_b_v1`，保留 HOMO–LUMO gap（eV）主任务和 charge／bond 上游辅助监督。**当前仅编写核心实现并做静态语法检查；没有编写或运行测试，没有运行 forward、数据处理、下载或训练。模型效果及数值行为尚未验证，后续继续优化。**

旧 QM9 的图组相对较弱局部读出有收益，但仍明显弱于 native；原生初始化后的辅助摘要也没有稳定提升。因此本包直接检验：保留冻结 native 起点后，完整 H／g 与几何输入上的辅助预测图残差是否还能改善 gap。它不是已有正结果，也不预设辅助输入具有 H 之外的新信息。历史结果与旧实现保持原样。

## 核心模型与公平对照

`model.NativeResidualGraphReadout` 输出 B-normalized gap：

`f_B(g,H)=f_native,B(g)+delta(H,g,z,distances,predicted_aux)`。

native 是上游 gap head 的冻结副本；复制时吸收 g 的 B_train 标准化以及 A-normalized → B-normalized target 转换。编码器与所有辅助头均冻结、eval；优化器仅更新残差模块。delta 的最后一层 weight／bias 都初始化为零。所有 recipe 都允许 B_val 选择 epoch 0；训练代码将初始读出与相同设备上直接重算的 native head 比较，同时单独记录其与历史缓存的误差，避免把 CPU／CUDA舍入差异当作训练收益。这里的等价是数学构造及带容差的运行时检查，**本轮未执行验证**。

全部组读取完整原子 H、全局 g、原子类型和坐标计算的距离 RBF；使用完全相同的节点／边槽宽、层数、参数模块及优化预算。几何组是 full-H geometric baseline，能够自行从几何学习关系。typed bond 保留五类完整概率 `(none,single,double,triple,aromatic)`；不先压成连接率或硬键。charge 为独立可选增量。unused slots 置零保证参数形状相同，但不保证活跃输入维度或优化难度相同。

| recipe | bond 槽 | charge 槽 | frozen auxiliary hidden 槽 | 用途 |
|---|---|---|---|---|
| geometry | 全零 | 全零 | 全零 | ST／MT 强几何基线 |
| bond | 完整预测五类概率 | 全零 | 全零 | 相对 geometry 的预测关系增量 |
| bond_charge | 完整概率 | 物理单位预测电荷 | 全零 | 电荷增量 |
| uniform | 五类各 0.2 | 预测电荷 | 全零 | 常数关系控制 |
| existence | 保留 p(none)，其余四类等分 1−p(none) | 预测电荷 | 全零 | 连接性／键类别控制 |
| shuffle | 分子内部打乱概率行 | 预测电荷 | 全零 | 保留分布，破坏原子对对应 |
| aux_hidden | 全零 | 全零 | charge／bond head 最后输出层之前的真实冻结激活 | 辅助表征可读性控制 |

除 geometry 外仅用于 MT。统一使用对称 pair 特征、双向 sum message aggregation、分子 sum pooling；坐标仅经距离进入，因此结构上支持刚体与原子排列不变性。shuffle 在缓存的固定原子对顺序上按 molecule ID 和固定 seed 确定，排列输入时必须让已打乱的边属性随 pair_index 一起搬运；不能重新按新行号打乱后再要求结果相同。排列不变性尚未做数值测试。

## 固定 A／Y 与嵌套 B

`config/nested_b_v1.json` 固定 A_train=56,000、A_val=5,000、B_pool=14,000、B_val=5,000、Y=20,000，沿用旧 100k 的数据抽样／划分协议；A_val 与 B_val 独立。先对 B_pool 的 molecule IDs 排序，再仅用 `nested_b.order_seed` 生成一次顺序，取前 2,000／7,000／14,000。因此不同 B 的训练集严格嵌套，A、B_val、Y 完全相同。

`data.prepare` 保存 `data/nested_b.json`，包括 pool IDs、固定顺序、所有前缀预算及 dataset／split SHA256。`nested_rows` 载入时核对身份、完整排列与分子顺序。只在当前 B_train 前缀拟合 H／g／target 标准化；不用 B_val／Y 拟合。

`plan.RECIPES`、`plan.all_configs(config)` 和 `plan.build_plan(config)` 是可审计矩阵接口。默认 5 upstream × 5 downstream seeds；10 次 ST／MT 上游训练和10套缓存为全部 B 预算共用，**不为每个 B 重训上游**。读出矩阵为 `5×3×(1 ST+7 MT)×5=600` 次。native 每个上游只有一个真实测量；跨预算引用不增加统计重复。

## 复用边界和 cache contract

复用 `experiments.qm9` 的数据解析、上游模型／训练、原始冻结缓存，不复制整套旧 pipeline。当前只允许在新实验自己的 namespace 内重新准备／训练／缓存；不导入历史模型、缓存或 marker。配置的 `data_root` 可复用只读 raw 输入，但 prepared 路径包含新 experiment；output 使用 `results/qm9_native_graph/<new experiment>/`。若未来要导入旧产物，必须另外实现有明确来源身份与内容校验的导入流程。

公共 `Context.code_hash` 不自动覆盖另一个实验包。为此配置显式记录旧 QM9 七个 Python 文件的逐文件 SHA256，六个阶段入口均核对 `legacy_source_sha256` 并绑定 `adapter_identity.json`。旧源码变化必须审阅后采用新实验身份，不能更新旧 marker 强行续跑。旧 loader 继续严格比较新 context.identity、config、prepared dataset／split、checkpoint 与缓存内容；wrapper 另核对模型参数、variant／seed、A_train 标准化及 A_val 选择身份。

| 来源 | 字段／单位 | 检查 |
|---|---|---|
| 新 namespace 的 legacy cache | ids、embedding(g)、atom_embedding(H)、atom_z、atom_offsets、pair_index、pair_offsets、target、main_prediction | legacy manifest / dataset / splits / checkpoint / IDs / shape / FP32 / checksum |
| MT legacy cache | charge_prediction（e）、bond_probs（完整五类概率） | 禁止辅助真值；charge 已反归一化一次 |
| 本包 supplemental cache | pos（Å）、charge_hidden、bond_hidden | 旧 cache 与 manifest SHA256、源码 pin、checkpoint、ID 顺序、shape / FP32 / finite |

旧 cache **不含坐标或辅助 hidden**；本包从同一 prepared molecule/atom顺序补坐标，并由冻结 head 的倒数输出层前激活补 hidden。不得把 `charge` 真值或 `pair_class` 真值放入读出输入。ST 的辅助 hidden 槽按相同宽度填零。`main_prediction` 与 `target` 缓存均为物理 eV，而 native head 原始输出是 A-normalized；模型返回 B-normalized，`physical()` 才转换回 eV。

## 六模块接口与后续接入

| 文件 | hook／接口 |
|---|---|
| data.py | download(context)、prepare(context)、nested_rows |
| training.py | train(context)、load_upstream(context)；上游 B 预算无关 |
| refine.py | cache(context)、train(context)、load_cache、predict_refiner |
| evaluate.py | evaluate(context)；全部600读出存在后才进入Y评分，严格加载身份 |
| analysis.py | analyze(context)；完整矩阵、无重复、同seed配对 |
| model.py | NativeResidualGraphReadout；冻结native＋零初始化图残差 |
| plan.py / identity.py | 无torch的纯计划接口／跨包源码身份 |

这些 hook 遵守公共单进程阶段接口；未来可交给 `scripts/run.py` 的顺序阶段调用。**现有 `pipeline/units.py` 与集群 pool／seed scheduler 没有 B_budget 维度，不支持直接调度本矩阵**；本包不修改公共枚举、依赖或启动器，也不提供冒充集群可运行的 wrapper。当前 `plan.build_plan` 只规划、不创建目录／模型／GPU任务；调用者应先把配置作为 JSON 读取后传入。

训练 checkpoint 保存 B预算、recipe、上下游 seed、训练分子 hash、标准化、上游 checkpoint hash、B缓存 manifest hash、嵌套计划 hash及source lock。Y评估保存每个读出的预测与checksum；分析先在每个上游seed内平均下游，再报告上游均值和 `ddof=1` SD。增量统一 `MAE(first)−MAE(second)`，正值表示第二组更好；不同 B 预算是依赖设置，不能当独立重复。

## 待优化与验证

尚待：在授权后做真实数据、模型前向和训练验证；检查native等价与排列不变性；完善B维度调度、失败恢复和并发锁；优化缓存／loader内存与构图速度；依据B_val进行预先定义的超参数选择。当前配置继承旧上游预算并给出一致残差默认值，不代表调优完成。辅助hidden对照同参数但可用槽不同，其表现不能单独证明辅助预测拥有新增信息。未验证的核心实现不能作为已有实验结论或集群运行保证。
