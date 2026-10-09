# G2APS-ReID P2：PCLHD Baseline 可行性审计

## 1. 审计范围与结论

本报告只进行静态源码分析和理论资源估算。审计过程中未启动训练、未提取全量特征、未运行 Jaccard/DBSCAN，也未修改 PCLHD、D-AGVA 或三域匹配实现。

审计对象为当前工作区中的实际复现代码：

- PCLHD 根目录：`PCLHD_repro_release/`
- 基础双域流程：`_train_lag_engine.py`
- 三域流程：`_train_lag_three_domain_engine.py`
- 统一入口：`train_lag.py` 与 `experiment_runner.py`
- G2APS P1 数据与协议定义：`g2aps_datm/tools/reconstruct_g2aps_p1.py`

选择 LAGPeR 引擎作为主审计对象，是因为它包含完整的 Stage 1、PCLHD Stage 2、ALL-memory、D-AGVA 接口和三域 Stage 2。现有 CARGO 引擎是同一流程的定制分支，其数据适配经验可以复用，但没有消除本报告发现的稠密 Jaccard 内存问题。

### 1.1 总体判断

| 项目 | 判断 | 原因 |
|---|---|---|
| Stage 1 特征提取与训练 | **条件可行** | mini-batch 显存不随训练集总规模增长；但 aerial 域的稠密 Jaccard 单次计算已有约 31.95 GiB 的双矩阵下界，DBSCAN 邻域开销仍可能使 64GB RAM 不稳定。 |
| 原始 PCLHD Stage 2 | **当前实现不可行** | 在计算 ALL Jaccard 时，理论可确认的 CPU 常驻下界约为 99.53 GiB，超过 64GB RAM。 |
| 三域 Stage 2（near ratio 0.5） | **当前实现不可行** | Far、Near、Ground 距离矩阵在 ALL Jaccard 前同时保留；可确认下界约为 91.54 GiB。 |
| RTX 4090 24GB | **不是当前首要阻塞项** | `search_option=3` 使用 CPU FAISS，源码没有在 GPU 上构造全量 $N\times N$ 张量；训练显存主要由 batch、AGW 激活、EMA encoder 和 Cluster Memory 决定。 |
| G2APS 数据接入 | **尚未实现** | Dataset registry 中没有 G2APS；基础引擎硬编码 `lag_ir/lag_rgb` 和路径字符串，评测端也没有读取 P1 固定协议的实现。 |

因此，不能直接用现有命令在 4090 24GB + 64GB RAM 上对 100,871 张 G2APS 训练图像执行完整 PCLHD Stage 2。必须先解决 ALL Jaccard/DBSCAN 的稠密内存路径，并增加独立、可审计的 G2APS Dataset Adapter。

## 2. 实际源码调用链

### 2.1 入口与 Stage 切换

1. `PCLHD_repro_release/train_lag.py:3-7` 调用 `experiment_runner.run_experiment('lag')`。
2. `PCLHD_repro_release/experiment_runner.py:16-41` 将基础引擎和三域引擎分别绑定到 `_train_lag_engine.py` 与 `_train_lag_three_domain_engine.py`。
3. `experiment_runner.py:49-59` 定义 `full/stage1/stage2`、Stage 1 augmentation 和 Stage 2 method。
4. `experiment_runner.py:298-323` 先以 `--stage1-only` 启动基础引擎，再根据配置以 `--stage2-only` 启动基础或三域引擎。
5. 基础引擎在 `_train_lag_engine.py:400-445` 解析模式并调用 `main_worker_stage1()` 或 `main_worker_stage2()`；两个 worker 分别定义于 `_train_lag_engine.py:446` 和 `_train_lag_engine.py:678`。
6. Stage 2 从显式 checkpoint 或默认 `model_best.pth.tar` 初始化，见 `_train_lag_engine.py:701-712`。正式 G2APS 实验必须改用预先冻结的 checkpoint 规则，不能根据正式测试集性能选取 `model_best`。

### 2.2 基础双域 PCLHD 流程

| 环节 | 真实代码位置 | 行为 |
|---|---|---|
| Ground 特征提取 | `_train_lag_engine.py:732-740` | 使用 `mode=1`，分别提取 EMA 与在线 encoder 特征，并按训练列表顺序拼接为 CPU tensor。 |
| Aerial 特征提取 | `_train_lag_engine.py:742-750` | 使用 `mode=2`，同样保留 EMA 与在线特征。 |
| ALL 特征合并 | `_train_lag_engine.py:752-753` | `torch.cat([features_rgb, features_ir])`。在 LAG 命名中 RGB 对应 ground，IR 对应 aerial。 |
| 域内 Jaccard | `_train_lag_engine.py:755-758` | 先后计算 aerial 与 ground 的 k-reciprocal Jaccard，并立即调用预计算距离 DBSCAN。返回的两个距离矩阵仍被变量持有。 |
| ALL Jaccard | `_train_lag_engine.py:759-762` | 对全部样本调用 `compute_modal_invariant_jaccard_distance()`，固定 `k1=40, k2=32`。 |
| DBSCAN | `_train_lag_engine.py:724-730, 755-762` | 三个 `DBSCAN(metric='precomputed', min_samples=4, n_jobs=-1)` 分别生成域内及 ALL 伪标签。 |
| Cluster Memory | `_train_lag_engine.py:823-857` | 两个域使用 `CMhard`，ALL 使用 `CMhybrid`；memory buffers 被复制到 GPU。 |
| PGM/BGM | `_train_lag_engine.py:900-931` | 对两组域内簇原型执行 Hungarian matching，并处理较大一侧的未匹配簇。 |
| 训练 DataLoader | `_train_lag_engine.py:959-983` | 同时建立 domain aerial、domain ground、ALL-aerial 和 ALL-ground 四个 loader。 |
| 两次优化 | `clustercontrast/trainers.py:207-284` | 第一次优化域内/跨域/EMA loss，第二次优化 ALL-memory；每个 iteration 有两次 `optimizer.step()`。 |

特征提取的底层实现位于 `clustercontrast/evaluators.py:21-63`。输入被送入 GPU，输出立即转回 CPU；原图与水平翻转图的特征在 CPU 上平均，并保存在 `OrderedDict` 中。整理后的聚类特征是 CPU `float32` tensor，因此后续 Jaccard 的主要内存压力位于 CPU RAM，而不是 GPU VRAM。

### 2.3 k-reciprocal / Jaccard 的真实实现

`clustercontrast/utils/faiss_rerank.py:30-123` 实现域内 Jaccard，`faiss_rerank.py:126-247` 实现模态平衡的 ALL Jaccard。关键事实如下：

1. 当前调用均使用 `search_option=3`，进入 `faiss_rerank.py:59-62, 159-162` 的 CPU `IndexFlatL2` 精确近邻搜索；没有在 GPU 上生成全量距离矩阵。
2. `use_float16` 默认是 `False`，基础和三域引擎均未覆盖该参数。因此 `V`、`V_qe` 和 `jaccard_dist` 都是 `numpy.float32`。
3. `V` 在 `faiss_rerank.py:71` 或 `:171` 被分配为稠密 $N\times N$ 数组。
4. 当 $k_2\neq1$ 时，`V_qe` 在 `faiss_rerank.py:90` 或 `:191` 再分配一个同尺寸数组。赋值 `V = V_qe` 之前，新旧矩阵同时存在。
5. Jaccard 输出在 `faiss_rerank.py:102` 或 `:226` 再分配一个同尺寸数组；计算阶段它与 `V` 同时存在。
6. 最后的 `jaccard_dist < 0` 还会产生一个 $N\times N$ boolean mask，见 `faiss_rerank.py:118-119, 242-243`。

因此，虽然近邻搜索只返回 $N\times k$ 的索引，Jaccard 主体仍是严格的 $O(N^2)$ 稠密内存实现。

### 2.4 Cluster Memory、Sampler 与 DataLoader

- `clustercontrast/models/cm.py:128-190` 定义 `ClusterMemory`。`CMhard` 保存 $2C\times D$ 的 mean/hard bank 和 $16C\times D$ 的 EMA sample bank；`CMhybrid` 保存 $2C\times D$ bank。
- `clustercontrast/utils/data/sampler.py:46-106` 的 `RandomMultipleGallerySampler` 根据当前伪标签和 camera ID 采样。训练阶段真正进入 sampler 的标签来自 DBSCAN，而不是初始数据集身份标签。
- `_train_lag_engine.py:125-168` 构造 aerial/ground loader；均启用 `pin_memory=True` 和 `drop_last=True`。
- `clustercontrast/utils/data/preprocessor.py:11-39` 延迟读取单张图像并应用可选 `image_process` 和 transform。
- `clustercontrast/utils/data/__init__.py:7-27` 的 `IterLoader` 在数据耗尽后重建 iterator。当前实现用裸 `except` 捕获所有异常，不仅捕获 `StopIteration`；后续移植时应至少将非耗尽异常继续抛出，否则可能掩盖数据错误。该修正不改变算法，但属于健壮性修复。

### 2.5 D-AGVA 调用位置

1. `clustercontrast/utils/data/dynamic_agva.py:12-59` 定义在线增强，按 `apply_slots/3` 的概率在读取 aerial 图像时执行。
2. `_train_lag_engine.py:105-122` 构造 aerial transform；启用 D-AGVA 时不追加 `ChannelAdapGray`。
3. `_train_lag_engine.py:125-142` 将 `get_dynamic_agva_process()` 注入 aerial loss loader。
4. `clustercontrast/utils/data/preprocessor.py:31-37` 先执行 D-AGVA，再执行普通 resize/crop/flip/normalize transform。
5. 实际 mask-guided remap 位于 `clustercontrast/utils/data/aerial_ground_augment.py:39-245`。
6. `experiment_runner.py:298-320` 控制 D-AGVA 只在 Stage 1 或原始 Stage 2 中启用；`experiment_runner.py:235-237` 明确禁止三域 Stage 2 同时启用动态 AGVA。

G2APS 当前没有已验证的 SCHP mask 目录，因此 D-AGVA 不能仅靠 Dataset Adapter 启用。此问题不阻塞关闭 D-AGVA 的 PCLHD baseline，但阻塞后续 D-AGVA 实验。

### 2.6 三域匹配调用位置

1. `_train_lag_three_domain_engine.py:171-188` 加载 view score，并将 aerial 样本拆为 Far/Near。
2. `_train_lag_three_domain_engine.py:423-443` 每轮只提取一次完整 aerial 特征，再通过索引切分 Far/Near。
3. `_train_lag_three_domain_engine.py:445-466` 依次计算 Far、Near、Ground、ALL 四个稠密 Jaccard 矩阵；前三个矩阵在 ALL 计算期间仍常驻。
4. `clustercontrast/three_domain.py:57-100` 构建双向 BGM：Hungarian matching 给出一对一核心，额外簇用最近原型补全。
5. `_train_lag_three_domain_engine.py:492-499` 只建立 Far$\leftrightarrow$Near 与 Near$\leftrightarrow$Ground，不建立 Far$\leftrightarrow$Ground。
6. `clustercontrast/trainers_three_domain.py:39-209` 执行三域主优化和 ALL-memory 第二次优化。

## 3. 复杂度与内存估算

### 3.1 估算条件

- $N_a=65{,}482$，$N_g=35{,}389$，$N_{all}=100{,}871$。
- AGW 输出维度 $D=2048$，见 `clustercontrast/models/agw_lag_stable.py:184`。
- 特征和当前 Jaccard 默认均为 `float32`。
- 域内使用命令行 $k_1=30,k_2=6$；ALL 在源码中固定为 $k_1=40,k_2=32$。
- GiB 按 $2^{30}$ bytes 计算。
- 下列数字是由形状与 dtype 推导的理论值，不是服务器实测峰值。

### 3.2 特征、近邻与稠密距离矩阵

| 集合 | $N$ | $N\times2048$ float32 特征 | $N\times N$ bool | $N\times N$ float16 | $N\times N$ float32 | $N\times N$ float64 |
|---|---:|---:|---:|---:|---:|---:|
| Aerial | 65,482 | 0.500 GiB | 3.993 GiB | 7.987 GiB | 15.974 GiB | 31.947 GiB |
| Ground | 35,389 | 0.270 GiB | 1.166 GiB | 2.333 GiB | 4.666 GiB | 9.331 GiB |
| ALL | 100,871 | 0.770 GiB | 9.476 GiB | 18.952 GiB | 37.905 GiB | 75.809 GiB |

精确 FAISS top-40 的索引若按 `int64` 保存，ALL 只约 0.030 GiB；对应 `float32` 距离约 0.015 GiB。由此可见，kNN 结果本身不是主要问题，稠密 `V/V_qe/jaccard_dist` 才是主要 OOM 来源。

### 3.3 Jaccard 生命周期与 CPU 峰值

对任意集合，当前函数至少出现以下两个稠密双矩阵阶段：

1. query expansion：`V + V_qe`；
2. Jaccard 生成：`V + jaccard_dist`。

所以单次 ALL Jaccard 的确定性下界是：

\[
2\times100{,}871^2\times4\ \text{bytes}=75.81\ \text{GiB}.
\]

这已经超过 64GB RAM，尚未计入特征、FAISS index、Python list、倒排索引、boolean mask、DBSCAN 邻域和进程运行时。

基础 Stage 2 在进入 ALL 计算前还保留 aerial 与 ground 的返回距离矩阵，合计 20.64 GiB。与此同时，在线/EMA 域特征及 `features_all` 约 2.31 GiB，CPU FAISS `IndexFlatL2` 对 ALL 特征的内部副本约 0.77 GiB。因此可由源码直接确认的下界为：

| 同时存活对象 | 内存 |
|---|---:|
| Aerial + Ground 返回 Jaccard | 20.639 GiB |
| ALL 的 `V + V_qe` 或 `V + jaccard_dist` | 75.809 GiB |
| 在线/EMA 域特征及 `features_all` | 2.309 GiB |
| ALL CPU FAISS index（特征副本） | 0.770 GiB |
| **可确认下界** | **99.527 GiB** |

该下界不包含 DBSCAN，因此无法通过减少 DataLoader workers 等外围调整降到 64GB 以内。

三域 Stage 2 在 $q=0.5$ 时，Far 与 Near 各约 32,741 张图像。Far、Near、Ground 三个返回 Jaccard 合计约 12.65 GiB；再加 ALL 双矩阵、特征和 FAISS index，可确认下界约为 **91.54 GiB**。三域划分减小了 aerial 域内矩阵，但没有改变 ALL 的 $N_{all}^2$ 瓶颈。

Stage 1 不计算 ALL。按当前“先 aerial、后 ground、最后统一删除”的生命周期，最大的单函数阶段约为 aerial 的两个 float32 稠密矩阵，即 31.95 GiB；加入两域特征、FAISS index 和随后 DBSCAN 的邻域结构后，64GB RAM 只能判为条件可行，不能在未测峰值前判为安全。

### 3.4 DBSCAN 开销

当前代码将完整稠密 Jaccard 传入 `DBSCAN(metric='precomputed', n_jobs=-1)`。其基础输入已经是一个 $N^2$ 数组；半径邻域索引的附加空间取决于 `eps` 下的边数 $E$，平均复杂度可写为 $O(N+E)$，最坏仍可达到 $O(N^2)$。此外：

- `requirements.txt` 和 `environment.yml` 没有固定 NumPy/scikit-learn 版本；输入校验是否复制、并行后端和邻域表示可能随版本变化。
- 若内部将数据提升为 float64，ALL 单矩阵会从 37.90 GiB 增至 75.81 GiB。
- `n_jobs=-1` 不应被解释为零复制保证；具体并行实现需要在服务器锁定版本后测量 RSS/PSS。

因此，报告不能给出一个伪精确的 DBSCAN 峰值。可以确定的是：它只会增加当前 99.53 GiB 下界，不会使现有实现适配 64GB RAM。

### 3.5 Cluster Memory 与 GPU 显存

设 aerial、ground 和 ALL 的簇数分别为 $C_a,C_g,C_{all}$，$D=2048$，`num_instances=16`。当前 Stage 2 GPU memory bank 为：

\[
M_{CMhard}(C)=18CD\times4\ \text{bytes},\qquad
M_{CMhybrid}(C)=2CD\times4\ \text{bytes}.
\]

即每个 CMhard 簇约 0.140625 MiB，每个 CMhybrid 簇约 0.015625 MiB。总 bank 为：

\[
0.140625(C_a+C_g)+0.015625C_{all}\ \text{MiB}.
\]

若采用极端上界“每个样本都是一个簇”，aerial CMhard、ground CMhard 和 ALL CMhybrid 分别约为 8.99、4.86 和 1.54 GiB，总计 15.39 GiB。实际 DBSCAN 簇数通常远低于样本数，但在尚未运行 G2APS 聚类前不能用未经测量的簇数给出更乐观结论。

CMhard 的 EMA monitor 还会生成 batch$\times16C$ logits。即使 `loss_ema` 不参与反向传播，`clustercontrast/trainers_no_ema_loss.py:79-90` 仍执行 EMA forward 和 CMhard monitor，因此“去掉 EMA loss”不会消除这部分瞬时显存或 `features_ema` bank。

AGW encoder 与 EMA encoder 同时位于 GPU，见 `_train_lag_engine.py:189-205`。从模块结构静态估算，每个 encoder 约 72.6M 参数；两份 float32 参数约 0.54 GiB，训练 encoder 的梯度和 Adam 状态还会增加约 0.26 GiB，未计激活和 CUDA/cuDNN workspace。该模型值是静态结构估算，不是 `torch.cuda.max_memory_allocated()` 实测值。

### 3.6 Batch、DataLoader 与显存尺度

在 $288\times144$、float32 下：

- 64 张 RGB 输入约 30.38 MiB；
- 128 张约 60.75 MiB；
- 192 张约 91.13 MiB。

Stage 2 的 ground loader 为每张图像产生两个增强视图。`batch_size=64` 时，ground 实际送入 128 张，aerial 送入 64 张，联合 forward 共 192 张。ALL-memory 的第二次优化在第一次 `optimizer.step()` 后执行，因此两个 192-image forward 不会作为同一个 autograd graph 同时驻留。三域流程的 Far 32 + Near 32 + Ground 128 仍是 192 张。

原始输入只占显存的一小部分，ResNet 中间激活和 cuDNN workspace 才是训练 batch 的主要成本。数据集从 LAGPeR 增长到 G2APS 不会直接放大这一部分，因此 24GB VRAM 对固定 batch 的可行性需要用单 iteration smoke test确认，而不能仅靠样本总数推断。

CPU 侧需要注意：基础 Stage 2 同时建立四个 DataLoader，每个默认 8 workers、`pin_memory=True`；PyTorch 默认预取因子通常为 2。只按 batch tensor 计算，四个 loader 的理论预取量已约 2.85 GiB，另有 32 个 worker 的解释器、PIL 解码与 transform 开销。三域流程会建立六个 loader，worker 数可达 48。该开销不是当前 99.53 GiB 的主因，但在改造 Jaccard 后必须纳入 64GB 总预算。

### 3.7 评测内存

P1 协议列表的总行数包含 query：

- A$\rightarrow$G+：4,876 query + 37,202 gallery；
- G$\rightarrow$A+：4,876 query + 62,791 gallery。

对应 float32 query-gallery 相似度矩阵约为 0.676 GiB 和 1.141 GiB，可在 64GB RAM 上处理。真正需要避免的是 `_train_lag_engine.py:209-228` 的 `TestData`：它会在构造 Dataset 时解码并缓存所有测试图像。若用于 G2APS，两套协议的原始 uint8 图像缓存分别约为 4.88 GiB 和 7.84 GiB，Python list 转 NumPy array 时还可能出现接近双份的瞬时占用。G2APS 评测 Adapter 应采用与普通 `Preprocessor` 类似的按需读取方式，而不是复用该 eager loader。

## 4. 不改变算法语义的优化候选

### 4.1 可以直接实施的生命周期优化

以下改动不改变近邻、Jaccard 数值或 DBSCAN 目标：

1. 每个域执行“Jaccard $\rightarrow$ DBSCAN $\rightarrow$ 立即删除距离矩阵 $\rightarrow$ `gc.collect()`”，不要在计算 ALL 时保留域内矩阵。
2. 特征字典整理为有序 tensor 后立即释放 `OrderedDict`；在线/EMA 特征可按用途写入预分配 tensor 或 memmap，避免临时 `torch.cat` 副本。
3. FAISS index、`initial_rank`、倒排索引和无用 loader 在阶段结束后显式释放。
4. 将评测图像改为 lazy loading；相似度按 query block 计算并在线累计 CMC/mAP/mINP，避免保存不必要的完整排序中间量。
5. 将 `IterLoader.next()` 的裸 `except` 改为只捕获 `StopIteration`，防止把数据损坏或 worker 异常误判为 epoch 耗尽。

这些改动能够降低域内矩阵和 Python 对象的叠加，但不能解决单次 ALL Jaccard 的 75.81 GiB 双矩阵峰值。

### 4.2 精确、但需要等价性验证的 ALL 路径

优先方案是保留精确 `IndexFlatL2`、相同 $k_1/k_2$ 和 float32 运算，重写存储方式：

1. 将逻辑上稀疏的 k-reciprocal 权重 `V` 和 query-expanded `V` 存为 CSR/分块结构，而不是稠密 $N^2$ 数组。
2. 按行或按块精确计算 Jaccard。对没有共享非零邻域的样本对，其距离为 1；当 DBSCAN 的 `eps<1` 时，无需把这些边写入图。
3. 只保存距离不大于 `eps` 的精确半径邻接图，并以 scipy sparse precomputed matrix 交给 DBSCAN，或实现与 sklearn DBSCAN 完全一致的确定性邻域遍历。
4. 在小规模固定数据上逐项比较 legacy 与新实现：top-k 索引、Jaccard 最大绝对误差、`eps` 边集合、噪声点、簇划分（允许簇编号置换）。全部通过后才能称为语义等价。

该方案在数学上可以保持原始目标，但浮点求和顺序、稀疏矩阵对显式零的解释及 DBSCAN 邻域遍历顺序都需要回归测试，不能未经验证直接宣称 bitwise identical。

如果短期内不重写稀疏逻辑，可采用 float32 `numpy.memmap` 保存 `V/V_qe/jaccard_dist` 并按块计算。这可以把工作集转移到磁盘，但计算量仍为 $O(N^2)$，DBSCAN 仍可能复制或构造巨大邻域，I/O 成本也会很高。它适合作为等价性参考实现，不一定适合作为最终训练实现。

### 4.3 会改变数值路径或算法语义的方案

下列方案不能作为“无语义变化”的默认优化：

- 将 `use_float16=True`：ALL 单矩阵可降至 18.95 GiB，但会改变 Jaccard 数值和 DBSCAN 边界样本；即使生命周期优化后可接近 64GB，也必须单独做复现实验。
- FAISS IVF/HNSW、近似 kNN：可能改变 reciprocal 邻域和伪标签。
- 降低 $k_1/k_2$、关闭 ALL-memory、移除 EMA monitor、降低训练样本数：均改变原始 PCLHD 配置或目标。
- 随机抽取训练子集：不能替代本任务要求的完整训练。

## 5. G2APS 数据接入审计

### 5.1 当前不支持的原因

1. `clustercontrast/datasets/__init__.py:14-36` 只注册 AG-ReID、LAGPeR 和 CARGO 等 Adapter，没有 `g2aps_aerial/g2aps_ground`。
2. `_train_lag_engine.py:462-463, 695-696` 将数据集名称硬编码为 `lag_ir/lag_rgb`。
3. `_train_lag_engine.py:792-797, 895-898` 依赖路径是否包含 `aerial_modify` 或 `ground_modify` 判断域。
4. `clustercontrast/utils/faiss_rerank.py:198-215` 的 ALL query expansion 同样通过 `ir_modify/aerial_modify/rgb_modify/ground_modify` 字符串判断域。
5. P1 的实际目录为 `bounding_box_train/PID/`，文件名采用 `PID_c0v...` 或 `PID_c3v...`，不会命中上述路径判断。
6. 现有 LAG 测试函数只理解 LAGPeR 的目录和协议，不读取 P1 的固定 query/protocol list。

因此，仅在 registry 中新增两个类仍不足以运行；还必须把域标记从路径字符串中解耦出来。

### 5.2 最小 Dataset Adapter 需求

后续实现应新增两个训练视图 Adapter 和一个协议 Adapter：

#### `g2aps_aerial`

- 递归读取 `bounding_box_train/*/*.jpg` 中满足 `c0` 的图像。
- 保留 camera/domain ID `0`。
- 不把目录 PID 或文件名 PID作为训练监督标签；训练 tuple 中使用明确的无监督占位值，DBSCAN 后再由伪标签替换。
- 真实 PID 只允许用于 P1 数据审计，不进入 sampler、loss、聚类约束、checkpoint 或超参数选择。

#### `g2aps_ground`

- 读取同一训练目录中满足 `c3` 的图像。
- 保留 camera/domain ID `3`。
- 采用与 aerial 相同的身份隔离规则。

#### 固定评测协议 Adapter

- 读取 `protocols/query_c0.txt`、`query_c3.txt`、`exp3_A2G+.txt` 和 `exp4_G2A+.txt`。
- 校验 `protocol_metadata.json` 中的 count 与 SHA256。
- A$\rightarrow$G+ 使用 `query_c0` 和 `bounding_box_test_c3 + ground PID 5555 distractors`。
- G$\rightarrow$A+ 使用 `query_c3` 和 `bounding_box_test_c0 + aerial PID 5555 distractors`。
- PID 5555 只能作为负样本，不得成为任何 query 的正样本。
- 不新增或暗示 G$\rightarrow$A+G；P1 明确只定义 A$\rightarrow$G+ 与 G$\rightarrow$A+。
- 使用 lazy image loading，并执行标准 same-PID/same-camera 过滤、无正样本 query 检查和 query/gallery 路径去重。

### 5.3 域元数据改造

最小安全方案不是伪造包含 `aerial_modify` 的目录名，而是让 ALL Jaccard 和原型生成函数显式接收与样本等长的 domain flag。例如，按照稳定排序生成 `is_aerial: bool[N]`，并在以下位置替换路径字符串判断：

- `compute_modal_invariant_jaccard_distance()` 的 query expansion；
- `generate_modal_invariant_cluster_features()`；
- ALL 跨域簇筛选；
- 三域 `modality_is_aerial`。

这属于数据接入和元数据传递改造，不改变 PCLHD 的损失或匹配语义。必须用 AG-ReID/LAGPeR 小规模回归确认改造前后伪标签一致。

### 5.4 D-AGVA 与三域匹配的额外前提

- D-AGVA：需要为 65,482 张 aerial 训练图像生成、校验并冻结 SCHP 前景 mask 映射；缺失 mask 时不能在正式实验中静默改变有效增强比例。
- 三域匹配：需要覆盖全部 aerial 样本的 ground-view resemblance score manifest。G2APS 重建后 c0 是单一域标签，不能把它直接解释为多个独立物理摄像机；若使用 `v` 序列做 OOF 分组，必须先核实其采集语义并证明 train/validation group 隔离。
- View score、near ratio 和训练配置不得通过正式 A$\rightarrow$G+/G$\rightarrow$A+ 测试结果选择。

## 6. 最小改造方案

建议将工作拆成彼此可单测的四个改动集，避免一次性改动训练核心：

1. **G2APS I/O 层**：增加 aerial/ground train Adapter 和固定协议 eval Adapter；不触碰距离、Memory 或 loss。
2. **显式域标记层**：消除路径字符串判断，并在旧数据集上验证输出一致。
3. **精确聚类后端层**：先调整对象生命周期，再实现 exact sparse/blockwise Jaccard + radius graph；保留 legacy dense 实现作为小规模 oracle。
4. **资源与错误保护层**：加入矩阵预算预检、RSS/GPU peak 记录、非有限特征检查、空簇/全噪声检查和固定 checkpoint 规则。

不建议先写训练命令再依靠 Linux swap“试跑”。当前 Stage 2 的理论下界已经超过物理内存，换页只会导致极慢运行或进程被 OOM killer 终止，不能形成可靠复现实验。

## 7. 分阶段验证计划

### Phase 0：纯 CPU 单元测试

- 用小型目录 fixture 验证 c0/c3 拆分、路径排序、训练 PID 隔离和 PID 5555 评测行为。
- 验证协议 list count、SHA256、query/gallery 无非法重叠及每个 query 的有效正样本。
- 验证显式 domain flag 与样本顺序严格对齐。

### Phase 1：距离实现等价性

- 在固定的 500、2,000、5,000 样本子集上同时运行 legacy dense oracle 与候选 exact 后端。
- 比较 exact top-k、Jaccard、`eps` 邻接边、DBSCAN noise mask 和簇划分。
- 子集仅用于工程等价性验证，不作为正式训练或论文结果。

### Phase 2：资源递增测试

- 在 5k、10k、25k、单域全量、ALL 全量的顺序下测量 RSS/PSS、临时磁盘、运行时间和边数 $E$。
- 每个阶段设置硬 RAM 预算与失败即停，不进入下一规模。
- 此阶段只测特征文件或聚类后端，不使用测试身份标签。

### Phase 3：GPU smoke test

- 在聚类后端通过后，只执行一个训练 iteration，记录 `torch.cuda.max_memory_allocated()` 与 `max_memory_reserved()`。
- 分别验证 Stage 1 和 Stage 2 的第一次优化、ALL 第二次优化、EMA monitor 和 memory bank。
- 验证 worker 数和 prefetch 对主机内存的影响，再决定正式 workers 配置。

### Phase 4：完整 baseline

- 使用全部 100,871 张训练图像和预先冻结的 epoch/checkpoint 规则。
- 测试集只在模型与配置冻结后用于最终 A$\rightarrow$G+、G$\rightarrow$A+ 报告。
- baseline 通过后，才分别迁移 D-AGVA 和三域匹配；不得在 baseline 尚未稳定时同时引入两个模块。

## 8. 必须先解决的 BLOCKER

| 优先级 | BLOCKER | 验收条件 |
|---|---|---|
| P0 | ALL Jaccard 稠密双矩阵超过 64GB | exact 后端在小规模与 legacy 等价，并在 $N=100{,}871$ 预算/实测中不超过预定 RAM 上限。 |
| P0 | Stage 2 同时保留域内距离矩阵 | 每个域 DBSCAN 后立即释放距离，内存生命周期测试通过。 |
| P0 | 缺少 G2APS Dataset/Protocol Adapter | 训练 65,482/35,389 计数正确；两套协议 count/SHA256 正确；PID 5555 仅作 distractor。 |
| P0 | 域判断依赖路径字符串 | 改为显式 domain flags，并在旧数据集上验证伪标签/原型结果一致。 |
| P1 | DBSCAN 稀疏输入与 sklearn 版本未冻结 | 固定 NumPy/scikit-learn/FAISS/PyTorch 版本，完成邻接与簇结果回归。 |
| P1 | 24GB GPU 的实际训练峰值未知 | 单 iteration 覆盖两次 optimizer step、EMA monitor 和最坏实测簇数，且保留安全余量。 |
| P1 | 现有 LAG 测试 Dataset eager 解码 | G2APS 评测改为 lazy loading 或流式特征提取。 |
| P2 | D-AGVA masks 与三域 view scores 尚未准备 | 分别完成 mask 覆盖审计和 group-isolated score manifest 后再启用。 |

## 9. 最终可行性结论

在不修改当前稠密 Jaccard/DBSCAN 数据路径的前提下，**G2APS-ReID 完整 PCLHD Stage 2 在 RTX 4090 24GB + 64GB RAM 上不可行**。确定性 CPU 内存下界约 99.53 GiB，已经足以构成停止条件；三域 Stage 2 也仍有约 91.54 GiB 下界。

该任务并非总体不可实现。特征提取、mini-batch 训练、Cluster Memory 和两套固定评测协议均可在现有硬件上通过受控改造实现。可行路线是：先增加无身份泄漏的 G2APS Adapter 与显式域标记，再以 legacy dense 实现为 oracle，开发经过等价性验证的 exact sparse/blockwise Jaccard 与半径邻接 DBSCAN 路径。完成这些 P0 项目前，不应启动正式 G2APS baseline 训练。
