# G2APS-ReID P3-B：显式域元数据与距离矩阵生命周期

## 1. 范围与结论

本阶段只完成两项工程重构：以显式物理域标记替代 G2APS 训练流程中的路径字符串判断，以及缩短 Stage 2 dense Jaccard 输出矩阵的存活时间。没有实现 Sparse Jaccard、近似近邻、半精度距离、训练子集采样或新的聚类策略；DBSCAN 的输入、`eps`、`min_samples`、执行顺序、CMhard、CMhybrid、ALL-memory、PGM/BGM、D-AGVA 和三匹配语义均保持不变。

小规模固定输入回归表明，legacy 路径语义与显式 flags 得到完全相同的 kNN 邻居、Jaccard 数值、DBSCAN 噪声掩码与成员划分、cluster prototype、BGM 映射和 ALL label translation。距离矩阵对象在 DBSCAN 返回后不再被 Stage 2 调用方持有。

该改造仍不足以在 64 GiB RAM 上运行 G2APS 的完整 dense ALL Jaccard。`N_all=100,871` 时，单个 float32 `N x N` 数组为 37.9047 GiB，而当前算法在 ALL Jaccard 内部至少同时需要 `V` 与 `jaccard_dist`，仅二者即约 75.8093 GiB，尚未计入特征、FAISS、倒排索引、DBSCAN 邻域及 Python/NumPy 临时对象。该问题明确保留为 P3-C BLOCKER。

## 2. 修改文件

| 文件 | P3-B 修改 |
| --- | --- |
| `clustercontrast/domain_metadata.py` | 新增物理域/视角子域契约、严格校验、legacy 兼容推断、排序/拼接/筛选/路径替换后的 flags 对齐、显式 prototype 与 ALL 标签翻译。 |
| `clustercontrast/memory_lifecycle.py` | 新增一次性构建、消费并释放 precomputed distance 的局部作用域封装。 |
| `clustercontrast/utils/faiss_rerank.py` | modal-invariant Jaccard 接受显式 `domain_flags`；query expansion 不再直接读取路径；G2APS 禁止静默路径兜底。 |
| `clustercontrast/three_domain.py` | Far/Near 子域与物理 camera/domain 分离；联合 prototype 和跨域簇判断统一使用显式域标记。 |
| `clustercontrast/datasets/g2aps_common.py` | P3-A 的 G2APS 训练行公开整数物理域标记 `0/3`。 |
| `clustercontrast/datasets/g2aps_protocol.py` | 固定协议行公开整数物理域标记 `0/3`。 |
| `train_lag.py` | 在仓库现有 RAHP/CESA 基础引擎中传播显式 flags；Aerial、Ground、ALL 距离依次构建/聚类/释放；prototype、关联分析和标签翻译不再依赖路径。 |
| `_train_lag_three_domain_engine.py` | 三域 Stage 2 传播 physical-domain 与 Far/Near subdomain；四次距离依次释放；prototype、BGM 输入和标签翻译统一对齐。 |
| `tools/validate_g2aps_p3b.py` | 只读真实数据元数据检查，不提取特征、不执行 Jaccard/DBSCAN/训练。 |
| `tests/test_g2aps_p3b_domain_memory.py` | 新增域对齐、legacy 兼容、生命周期和固定小规模算法语义回归。 |

`clustercontrast/trainers.py` 与 `clustercontrast/trainers_three_domain.py` 经过审查但未修改；二者消费上游已构建的 pseudo datasets、memory 和映射，不自行根据路径判断域。

## 3. 域标记契约

### 3.1 物理域

物理域标记与 G2APS camera ID 一致：

- `0`：Aerial；
- `3`：Ground。

`domain_flags[i]` 必须对应第 `i` 个 feature row。布尔值、其他整数、长度不一致、重复路径、元数据缺失都会立即报错。训练域不能由文件名中的真实 PID 推断。

G2APS adapter 以 `domain_metadata` 和 `domain_flags` 提供显式信息。若 G2APS 文件名满足 `PID_c[03]vSEQ_FRAME.jpg` 而调用方没有传入显式元数据，legacy resolver 会拒绝继续执行，不会根据路径目录静默猜测。

### 3.2 Far/Near 子域

三域训练另行维护：

- `A_far`；
- `A_near`；
- `Ground`。

Far/Near 只是 Aerial 的视角子域，二者的物理域标记仍为 `0`，不能伪装为新 camera ID。`validate_subdomain_flags` 强制 Ground 行只能属于 `Ground`，Aerial 行不能属于 `Ground`。

### 3.3 排序、拼接、筛选与路径替换

- 任意重排使用样本路径重新对齐 flags；
- Ground/Aerial 拼接由 `concatenate_domain_rows` 同时拼接样本和 flags；
- 筛选/索引重排由 `filter_domain_rows` 和 `reorder_domain_rows` 同步处理；
- 使用增强图根目录替换路径时，由 `carry_domain_flags_by_basename` 从替换前的显式行迁移 flags；源或目标 basename 重复、缺失或总体数量改变均直接报错，不能依赖增强目录名称。

## 4. 域判断调用链审计

以下行号对应本次交付版本。

| 环节 | 文件与位置 | P3-B 行为 |
| --- | --- | --- |
| G2APS 训练 adapter | `clustercontrast/datasets/g2aps_common.py` | 每行只公开 `pid=-1`、camera/domain metadata；真实 PID 仅用于离线隔离审计。 |
| 基础 Stage 2 排序与拼接 | `train_lag.py` 的 `main_worker_stage2()` | 对 `ir_train/rgb_train` 获取 flags，并与 `all_train` 同序拼接。 |
| 基础 Aerial/Ground/ALL 聚类 | `train_lag.py` 的 Stage 2 聚类块 | 三个 precomputed distance 逐个构建、传入原 DBSCAN、立即释放；ALL Jaccard 显式传 flags。 |
| ALL prototype | `train_lag.py` 的 `generate_modal_invariant_cluster_features()` | `generate_domain_balanced_cluster_features` 按 flags 聚合，不扫描路径。 |
| ALL query expansion | `clustercontrast/utils/faiss_rerank.py:129-151,159-233` | 按显式 flags 平衡 Aerial/Ground 邻居；legacy 数据可经统一 resolver 进入。 |
| ALL association | `train_lag.py` 的 `associated_analysis_for_all()` | `cross_domain_cluster_ids` 判定同时含两域的伪簇。 |
| Stage 2 label translation | `train_lag.py` 的 ALL pseudo-dataset 构建块 | 只使用 DBSCAN label 与 domain flag；忽略训练元组中的原始/占位 PID。 |
| PGM/BGM | `train_lag.py` 的 PGM 块 | 输入为上游显式分域后的 prototype 和伪标签；RAHP/CESA 与原匹配逻辑未删除。 |
| 三域 physical/subdomain 构建 | `_train_lag_three_domain_engine.py:363-400` | 物理域与 Far/Near 子域分别维护；路径替换时迁移 flags。 |
| 三域四次聚类 | `_train_lag_three_domain_engine.py:469-488` | Far、Near、Ground、ALL 依次构建/消费/释放；ALL 使用 physical flags。 |
| 三域 prototype 与 BGM | `_train_lag_three_domain_engine.py:497-519` | ALL prototype 使用 physical flags；链式 BGM 仍为 Far↔Near、Near↔Ground。 |
| 三域 ALL 映射 | `_train_lag_three_domain_engine.py:555-564` | 跨域簇判定使用 physical flags，Far/Near/Ground 行分发使用 subdomain flags。 |
| Trainer | `clustercontrast/trainers.py:197-229`、`clustercontrast/trainers_three_domain.py` | 只消费 `i2r/r2i` 或链式映射及已分组 loader，不执行路径域判断。 |

### Legacy 入口

统一 legacy resolver 只支持已有明确约定：`aerial_modify/ir_modify -> 0`、`ground_modify/rgb_modify -> 3`，以及 CARGO `Cam1..5 -> 0`、`Cam6..13 -> 3`。因此旧 AG-ReID、LAGPeR 和 CARGO 样本仍可进入共享 Jaccard。

现有 `train_agreid.py` 的 association 继续使用明确的 Ground/Aerial 拼接边界，没有被 P3-B 覆盖或替换。它没有作为 G2APS 训练入口改写；若后续让 G2APS 走该引擎，必须先完成与 `train_lag.py` 相同的显式 metadata 接入。CARGO 独立代码也不在本次提交的算法改造范围内。

## 5. Jaccard 生命周期

### 5.1 修改前

基础 Stage 2 在同一 epoch 作用域依次保存 Aerial、Ground、ALL 三个返回距离矩阵，直到全部 DBSCAN 完成后仍保持引用。三域 Stage 2 同理保存 Far、Near、Ground、ALL 四个返回矩阵。构建 ALL Jaccard 时，先前域内输出矩阵与 ALL 内部的 `V`、`jaccard_dist` 同时存在。

### 5.2 修改后

`cluster_precomputed_ephemeral`/`consume_precomputed_ephemeral` 将每个距离矩阵限制在一次 consumer 调用中：

1. 由原函数构建完全相同的 dense float32 距离；
2. 原样传给 DBSCAN 或 AG-ReID 原有 adaptive cluster consumer；
3. consumer 返回 labels 后在 `finally` 中删除最后一个局部引用；
4. 下一域的距离矩阵随后才开始构建。

这不会修改聚类参数或顺序，也没有用 `torch.cuda.empty_cache()` 掩盖 CPU 引用。prototype 和 memory 建立后，已无后续用途的原始/EMA feature tensors 也显式删除；Cluster Memory 自身继续保留。

### 5.3 理论内存变化

按 2048 维 float32 feature 估算：

| 数据块 | N | Feature 矩阵 | 单个 dense N×N | Jaccard 内部 V+输出下限 |
| --- | ---: | ---: | ---: | ---: |
| Aerial | 65,482 | 0.4996 GiB | 15.9736 GiB | 31.9473 GiB |
| Ground | 35,389 | 0.2700 GiB | 4.6655 GiB | 9.3310 GiB |
| ALL | 100,871 | 0.7696 GiB | 37.9047 GiB | 75.8093 GiB |
| Far 或 Near（q=0.5 近似） | 32,741 | 0.2498 GiB | 3.9934 GiB | 7.9868 GiB |

修改前仅三个基础 Stage 2 返回距离矩阵合计 58.5438 GiB；三域在 q=0.5 时四个返回矩阵合计约 50.5570 GiB。修改后调用方最多持有一个返回距离矩阵。构建 ALL 矩阵时不再叠加已经完成聚类的域内返回矩阵。

但 `compute_modal_invariant_jaccard_distance` 本身仍创建 dense `V` 和 dense `jaccard_dist`。DBSCAN precomputed 邻域结构还可能产生与数据分布相关的额外内存，NumPy/FAISS/Python 临时对象也未计入上述下限。因此 4090 24GB + 64GB RAM 仍不能安全执行 G2APS 全量 ALL dense Jaccard。

Cluster Memory 的大小取决于聚类数 C，而非图像数 N。当前 CMhard 约为 `(2+K) * C * D * 4` bytes（`K=16, D=2048`），ALL CMhybrid 约为 `2 * C_all * D * 4` bytes；由于 C 只能在实际聚类后确定，本报告不编造峰值实测结果。DataLoader/模型激活显存也与 batch、AMP 和网络中间激活有关，不属于本次 CPU dense-distance 阻塞的决定因素。

## 6. 算法语义回归

固定随机种子和小规模 synthetic features 下完成以下检查：

| 检查项 | 结果 |
| --- | --- |
| c0/c3 mapping、排序、拼接、筛选、重排 | 通过 |
| 增强根路径替换后的 flags 迁移 | 通过；不使用目录推断 |
| 缺失、非法、布尔 flags | 明确报错 |
| G2APS 缺失显式 metadata | 明确报错，不 legacy fallback |
| AG-ReID/LAGPeR/CARGO legacy 路径 | 与原约定一致 |
| kNN 邻居索引 | 完全相同 |
| query expansion 与 dense Jaccard | `rtol=0, atol=0` 完全相同 |
| DBSCAN noise mask | 完全相同 |
| DBSCAN cluster members | cluster ID 置换无关的成员划分相同 |
| cluster prototype | `rtol=0, atol=0` 完全相同 |
| BGM correspondence | 完全相同 |
| ALL label translation | 完全相同；真实 PID 不参与 |
| 距离对象生命周期 | weak reference 证明下一 builder 启动前前一矩阵已不可达 |

本地执行 `python -m unittest discover -s tests -v`：25 项测试全部通过。该结果属于小规模算法回归，不代表服务器真实规模 Jaccard 已运行。

## 7. 真实 PID 隔离

G2APS 训练 adapter 输出 `(path, -1, camid)`。域标记来自公开 camera/domain metadata；ALL label translation 只读取 DBSCAN 伪标签与 flags。回归测试故意向样本元组注入大于 9000 的假想真实 PID，输出训练伪标签仍只来自 DBSCAN labels，证明 PID 字段不参与 prototype、跨域簇判断或匹配。真实服务器检查脚本还要求 100,871 个训练行的 PID 全部为 `-1`。

## 8. 服务器部署与只读验证

在服务器完整工作树执行：

```bash
cd /data/home/lab338/Jaxon/PCLHD_repro_release

python -m unittest discover -s tests -v

python tools/validate_g2aps_p3b.py \
  --data-dir /data/home/lab338/Jaxon/dataset/G2APS_ReID_P1
```

第二条命令只初始化 adapter 并验证 65,482 Aerial、35,389 Ground、100,871 ALL 行的显式域标记和 `pid=-1`。预期状态为：

```text
G2APS_P3B_DOMAIN_VALIDATED
```

不得在 P3-B 验收中启动训练、全量特征提取、Jaccard 或 DBSCAN。

## 9. P3-C BLOCKER

P3-C 前仍有以下阻塞项：

1. 必须解决 `compute_modal_invariant_jaccard_distance` 的 O(N^2) dense `V` 与 `jaccard_dist`。在保持精确算法语义的前提下，需要单独设计并验证分块/稀疏等价实现；若采用 FAISS 近似近邻、稀疏候选截断或近似 Jaccard，必须明确标记算法语义与原 PCLHD 不再严格等价，并建立独立基线。
2. 当前基础/三域引擎已具备显式 metadata 消费能力，但命令行仍固定实例化 AG-ReID/LAGPeR adapter；G2APS dataset key、固定 A→G+/G→A+ evaluator 以及受限 smoke-run 必须在后续训练接入阶段显式绑定，不能通过伪造旧数据目录名进入。
3. 本地没有挂载服务器正式数据，`tools/validate_g2aps_p3b.py` 的真实 100,871 行检查需要在服务器执行并保存输出；本地 25 项测试仅证明接口和小规模语义。

P3-B 不包含上述改动，也不应据此启动 G2APS 完整训练。
