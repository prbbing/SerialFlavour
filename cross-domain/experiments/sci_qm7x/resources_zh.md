# QM7-X 资源预算与生命周期（2026-10-22）

默认配置只运行 multi_task，5 个上游 seed × 每个5配方×5下游 seed；ST 是独立可选消融。当前两份规模配置共享5,000个val，实际独立构型数50,000/95,000；A_train/B_train/Y原样本及分组、200/120 epoch 上限、早停、FP32、任务、几何访问与所有容量/shuffle对照保持。默认只活动一个上游 seed，多GPU参数只轮换设备，不增加并发。

| 项目（十进制 MB） | 50k | 100k |
|---|---:|---:|
| 一个 seed 的 B_train+B_val cache tensor | 187.14 | 296.305 |
| Y 永久 H/aux hidden cache | 0 | 0 |
| 假设当前划分ST+MT全10 seeds缓存B/Y（非执行路径） | 3430.90 | 6082.05 |
| 假设当前划分MT全5 seeds缓存B/Y（非执行路径） | 1715.45 | 3041.025 |
| prepared tensor 估算 | 33.55 | 63.745 |
| MT-only 永久上下游模型参数张量估算 | 58.87 | 58.87 |
| 默认上述 tensor 合计估算峰值 | 279.56 | 418.92 |

估算以 max_atoms=23、H宽96、aux hidden宽64和FP32计算；不是测量结果。合计未包括原始HDF5/XZ、样本字符串、序列化、JSON、训练history、checkpoint buffers、可选预测，也未表示CPU RSS/VRAM峰值。`workflow.release_cache=false` 时50k/100k的B缓存会累计到935.7/1481.525 MB（MT5 seeds）；ST+MT消融累计翻倍。保持默认及时退役时，ST消融仍只活动一个seed，B缓存峰值不变，永久checkpoint增多。

顺序固定为 prepare（一次）→ 一个 seed 的 upstream → B cache → 全部recipe×下游seed → 独立进程evaluate-seed → receipt → 删除明确列明的可再生成B tensor → 下一seed → 全矩阵汇总。训练和评估进程退出后释放模型/设备资源。Y按batch读取prepared输入，一次上游前向得到H/g/native，已在B_val选定的25个小readout共享该批次；每个模型使用自己的B_train input normalization，分子内shuffle按样本ID和原seed即时生成。误差用Float64累加，native每seed只评分一次，默认不保存完整预测。Y辅助真值只在显式diagnostics标志下评分，不进入读出。

每个receipt保留当前seed所有永久checkpoint、input normalization、训练历史/manifest、cache manifest、seed指标和评估manifest的实际路径/大小/SHA256，连同prepared/source/split/normalization来源证据。可再生成cache清理清单仅包含该seed cache_manifest已记载的b_train.pt和b_val.pt。清理不递归删除目录，未知文件、raw、prepared与source证据永久保留。完成receipt的hash核验优先于旧cache unit marker，所以已主动删掉B cache的seed不会重训。任何永久产物变化均使receipt失效；不能仅靠status=complete跳过。未完成seed失败时cache保留，支持重跑失败单元，不恢复optimizer/epoch状态。

只在全部计划seed receipts及其永久artifact有效后生成metrics/summary；缺seed、少recipe/downstream seed、损坏哈希均拒绝汇总。默认MT-only不生成ST对比。统计先每上游seed平均下游seed，再对5个上游seed报告均值与样本SD(ddof=1)，保留同seed配对；不把25个嵌套重复作为25个独立上游实验。不根据中途Y指标改配方、筛seed或停止后续seed。

入口：`scripts/run_cluster.py`或公共`scripts/run_lifecycle.py`，均支持`--config ... --gpus 0 --dry-run`只读列计划，不import torch/h5py、不建数据/结果目录。真实运行须去掉dry-run，后续执行需用户授权；本轮仅实现与合成离线合同测试，没有真实数据/训练/下载/远程运行。v1全缓存配置保留供旧入口使用，实验身份与当前v3共享val隔离。

四份当前v3资源配置显式设置 `workflow.free_space_reserve_bytes=1073741824`（1 GiB）。共享driver在prepare前检查输出盘可用空间，要求至少达到计划tensor峰值估算加该余量，覆盖估算未计入的序列化、小型manifest/history等额外开销；这仍是启动预检，不等于真实资源峰值已测量。v1保留配置未改动。
