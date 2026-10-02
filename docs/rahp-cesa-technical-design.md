# RAHP / CESA technical design against this PCLHD tree

## Repository audit and scope

The starting tree contained only SYSU/RegDB. The user supplied `C:/Users/Jaxon/Desktop/fsdownload/PCLHD2` as an implementation reference. Its LAG adapter did not contain the real seven-scene raw layout: the old `prepare_lag.py` converted AG-ReID and randomly split identities. That converter is now deprecated and fails fast. Formal LAGPeR code consumes only an already prepared official scene split through the contract below. The VI scripts remain the PCLHD reference. The two frozen methods share one implementation each across both dataset entrypoints.

## Existing data flow

| Step | Actual source | Frozen behavior / insertion |
| --- | --- | --- |
| Feature extraction and DBSCAN | `train_agreid.py` and `train_lag.py`, both Stage 1 and Stage 2; original analogues `train_sysu.py:458-516,784-906` and `train_regdb.py:386-443,622-740` | Extracted features follow `sorted(dataset_*.train)`; DBSCAN uses reranked/Jaccard distance. Leave clustering geometry and parameters intact. |
| Filtered pseudo-label tables | `train_agreid.py` and `train_lag.py`, both stages; original analogues `train_sysu.py:535-548,942-981` and `train_regdb.py:459-471,774-814` | RAHP tables append only labels other than `-1`, with `q_full[i]` in a parallel reliability array. ALL has its own label space and requires its own `q_full`. The disabled path preserves the reference ALL table handling. |
| Minibatch indexes | `clustercontrast/utils/data/preprocessor.py:11-66` | Both preprocessors return the index into the filtered dataset. An AG trainer gathers `q_filtered[indexes]` and duplicates it when it duplicates RGB labels/features. |
| Stage 1 memory calls | `clustercontrast/trainers.py:48-80` | Domain `ClusterMemory` calls ultimately use `CM_Hybrid` when configured `CMhybrid`. |
| Stage 2 memory calls | `clustercontrast/trainers.py:124-204` | Encoder domain memories call `cm_hybrid`, including the encoder path of `CMhard`; ALL memory uses `CMhybrid`. `encoder_ema` calls `cm_hard` on `features_ema`, which stays unchanged. Cross loss remains at coefficient `0.25`. |
| Dynamic hard proxy | `clustercontrast/models/cm.py:41-83` | Group by pseudo-ID, measure each instance against the old mean slot, update mean slot, select hard index, update hard slot. The disabled path keeps the original `np.argmin(np.array(distances))` and momentum formulas. |
| PGM | `train_agreid.py` and `train_lag.py`, Stage 2; original analogues `train_sysu.py:985-1041` and `train_regdb.py:818-861` | Normalize prototypes, compute raw cosine, exponentiate, reciprocal cost, Hungarian plus original unmatched completion, then `r2i/i2r` label translation. `R` is the final edge set. CESA score calibration belongs strictly between raw cosine and `exp()`. Each entrypoint retains its original unmatched handling. |
| Checkpoints | `train_agreid.py` and `train_lag.py`, both stages | Every epoch writes `checkpoint.pth.tar` for latest/resume. The predetermined last epoch also writes `model_final.pth.tar`. Test metrics never select a checkpoint. Stage 2 always stores model, EMA, optimizer, scheduler, RNG, DBSCAN eps, epoch and optional CESA state. |

## RAHP shared API and algorithm

`clustercontrast/methods/rahp.py` accepts normalized or raw extracted ResNet-50 features plus the pseudo-label vector, normalizes features internally, and returns a full-length `q_full` with zero at outlier positions. It computes exact raw-cosine KNN consistency without self-neighbors using FAISS inner product when available, with a query/reference chunked Torch top-k fallback. It never materializes the full sample-by-sample cosine matrix. Cluster centers and assignment margins are computed independently for each supplied domain/label space. Margin standardization uses median, MAD and clipping to `[-5, 5]`. The single-cluster margin reliability is one. The final reliability is `(q_nbr + eps)^alpha * (q_margin + eps)^(1-alpha)`.

For each batch pseudo-ID, `CM_Hybrid.backward` retains the old-slot similarity measurements before its mean update. With `reliability` supplied, hard candidates are the `max(1, ceil(beta * batch_count))` greatest values of `1 - cosine(feature, old_mean_slot)`. Highest reliability wins; exact ties favor higher hardness, then the smaller batch index. The original mean and hard momentum updates follow. With no reliability, the original `np.argmin(np.array(distances))` line runs. `CM_Hard` and `features_ema` are untouched.

Each AG/LAG trainer builds `q_filtered` in the same loop as each filtered pseudo-label table, then `clustercontrast/trainers.py` gathers it by the preprocessor indexes. A dual RGB view duplicates `q_batch` exactly when it duplicates labels. Domain and ALL tables each compute reliability using their own pseudo-label space; no ground-truth identity enters RAHP. A diagnostics accumulator records per-memory hard replacement, candidate count, reliability, and chosen hardness percentile, and emits one summary per epoch together with the KNN/margin statistics. Stage 1 applies RAHP to its domain memories when `--memorybank CMhybrid` is selected. Stage 2 forces `CMhard` for domain memories but its encoder branch still calls `cm_hybrid`; ALL uses `CMhybrid`. The EMA instance proxy remains outside RAHP.

Exact raw-cosine KNN chooses backends in this order: GPU FAISS `IndexFlatIP`, CPU FAISS `IndexFlatIP`, then chunked Torch exact top-k. All paths remove self-neighbors and preserve the same K, metric and neighbor definition; none creates an N by N matrix. Each reliability space prints `[RAHP-KNN] backend=... N=... K=... time=...`.

## CESA shared API and algorithm

`clustercontrast/methods/cesa.py` stores only previous aerial/ground label vectors, sparse matched-edge persistence and the next Stage 2 epoch index. The overlap contingency contains only observed pairs from aligned previous/current sample labels; `-1` is ignored in overlaps. Jaccard uses independently counted previous/current cluster sizes. A current cluster inherits a previous cluster only if both directions select one another as their unique highest-Jaccard match and Jaccard reaches the threshold. Ambiguous ties reset history, so a split or merge cannot copy one trajectory into multiple clusters.

At PGM's raw-cosine point, surviving previous edge keys are mapped through the two lineage maps into a sparse history boost. `calibrated_score = raw_cosine + min(stage2_epoch / warmup, 1) * eta * H_minus`. The caller then executes its existing `exp()`, reciprocal cost, Hungarian and unmatched logic unchanged. In these two entrypoints PGM is inline and `R` already remains in caller scope, so there is no return API to change. Source PGM rows are ground and columns are aerial; the integration transposes the score for CESA's aerial/ground state and converts each final `R` pair from `(ground, aerial)` to `(aerial, ground)`. `r2i/i2r` and the trainer's cross loss do not change. The first Stage 2 epoch has zero boost. After matching, each edge in final `R` receives `rho * H_prev + (1-rho)` only when its pair of lineage predecessors was a previous matched edge; otherwise it receives `1-rho`. Switching partner therefore restarts persistence. Checkpoint `cesa_state` contains `prev_labels_aerial`, `prev_labels_ground`, `edge_persistence`, and `stage2_epoch`; CESA checkpoints also store model, optimizer, scheduler, RNG and clustering eps for resume.

When `num_cluster_rgb < num_cluster_ir`, the frozen baseline does not execute PGM and retains `R=[]`, `r2i={}`, and `i2r={}`. CESA calls `advance_without_matching`: it records the current aligned aerial and ground pseudo labels, clears sparse edge persistence, and increments `stage2_epoch`. Its diagnostic reports `pgm_executed=False` and zero edges. This makes the next lineage transition compare adjacent epochs and prevents stale edges from crossing a no-match epoch.

## Checkpoint and evaluation protocol

Formal experiment checkpoint: **fixed final epoch**. Stage 1 and Stage 2 write `checkpoint.pth.tar` after every epoch and `model_final.pth.tar` only after the configured final epoch. Stage 2 initializes from the Stage 1 `model_final.pth.tar`. `--eval-during-train=False` is the default, so formal training does not read the test split each epoch. Enabling it is a debug-only logging option; Rank-1, mAP and mINP do not affect saving, resume, initialization, early stopping, or any training branch. Final evaluation is a separate command and loads the Stage 2 `model_final.pth.tar`.

RAHP Stage 1 requires `--memorybank CMhybrid`. An explicit `--use-rahp` with Stage 1 and any other memory bank raises `ValueError`. `--stage2-only --use-rahp` remains valid because it skips Stage 1.

## Architecture and pretrained checkpoint policy

Training and final evaluation both instantiate `models.create('agw', ...)`. Evaluation loads `state_dict` with `strict=True`; missing or unexpected parameters raise `Checkpoint architecture does not match AGW evaluation model.` AG-ReID remains A to G and G to A. LAGPeR uses its three protocols described below.

The unchanged baseline ImageNet file is `resnet50-19c8e357.pth`. Resolution order is `--pretrained-resnet50 PATH`, then `PCLHD_RESNET50_PRETRAINED`, then `examples/pretrained/resnet50-19c8e357.pth` relative to the repository. A missing file fails before model construction. The repository does not download or track this weight file.

## Official LAGPeR prepared-data contract

The repository does not have enough information to convert the raw seven-scene dataset. **BLOCKED: NEED REAL LAGPER DATA TREE** for a converter. `prepare_lag.py` is a deprecated guard and cannot generate formal data.

An externally prepared official tree must contain:

```text
LAGPeR/
  images/...
  meta/
    protocol.json
    train_aerial.txt
    train_ground.txt
    query_aerial.txt
    gallery_aerial.txt
    query_ground.txt
    gallery_ground.txt
```

`protocol.json` must declare dataset `LAGPeR`, protocol `official-scene-split`, four train scenes, three test scenes, 4/8 train aerial/ground cameras and 3/6 test aerial/ground cameras. Every manifest line is `image_path pid camid view`, with a path relative to the dataset root and view equal to `aerial` or `ground`. The loaders never create a split. Train PID is retained only as dataset metadata and is relabeled locally; DBSCAN still supplies every training pseudo label.

Final LAGPeR evaluation constructs:

- A to G: `query_aerial.txt` against `gallery_ground.txt`.
- G to A: `query_ground.txt` against `gallery_aerial.txt`.
- G to A+G: `query_ground.txt` against the concatenated aerial and ground galleries.

PID and real camera ID are read from the same manifest record and concatenated together. Evaluation removes only gallery records for which both PID and real camera match the query. Filenames of the form `0001_c13_000008.jpg` can be audited as PID 1, camera 13; view type is never substituted for camera ID.

## CLI, ablations, and boundaries

The shared CLI helper defines `--use-rahp`, `--rahp-beta` (0.25), `--rahp-knn` (20), `--rahp-alpha` (0.5), `--use-cesa`, `--cesa-rho` (0.8), `--cesa-eta` (0.1), `--cesa-lineage-thr` (0.5), and `--cesa-warmup` (5). It derives `baseline`, `rahp`, `cesa`, or `rahp_cesa` from the two flags. Both AG-ReID and LAGPeR RGB/RGB entrypoints omit `ChannelExchange`, `ChannelAdapGray`, and related VI channel simulation by default while retaining the established ordinary transforms. `agw` remains the only accepted architecture in these entrypoints and uses ResNet-50. DBSCAN, PGM unmatched completion, and evaluation protocols remain outside this method change.

The AG-ReID data root is shared by both modalities. Training reads
`aerial_modify/{trial}/bounding_box_train` and
`ground_modify/{trial}/bounding_box_train`; formal evaluation reads
`idx/test_aerial_{trial}.txt` and `idx/test_ground_{trial}.txt`. The caller
passes this shared root directly, without an `agreid_ir` or `agreid_rgb`
suffix. LAGPeR follows the prepared manifest contract above. The formal scripts freeze `batch-size=64` and `num-instances=16`, matching the supplied historical AG/LAG run scripts; the ground loader receives batch 32, which remains two complete 16-instance groups. Baseline, RAHP, CESA and Full share every non-method argument. Run them with:

| Setting | Frozen value for AG-ReID and LAGPeR |
| --- | --- |
| architecture / memory / pooling | `agw` / `CMhybrid` / `gem` |
| image size / batch / instances | `288x144` / `64` / `16` |
| epochs / iterations / trial | `50` / `400` / `1` |
| optimizer | lr `0.00035`, weight decay `0.0005`, momentum `0.2`, step `20` |
| clustering CLI | eps `0.6`, eps gap `0.02`, k1 `30`, k2 `6` |
| temperature / seed / workers | `0.05` / `1` / `8` |
| RAHP | beta `0.25`, KNN `20`, alpha `0.5` |
| CESA | rho `0.8`, eta `0.1`, lineage threshold `0.5`, warmup `5` |
| evaluation during training | `False` |

```bash
DATA_DIR=/path/to/AG-ReID-root LOGS_DIR=logs/agreid PRETRAINED_RESNET50=/path/to/resnet50-19c8e357.pth bash scripts/run_agreid_ablation.sh
DATA_DIR=/path/to/LAGPeR-root LOGS_DIR=logs/lagper PRETRAINED_RESNET50=/path/to/resnet50-19c8e357.pth bash scripts/run_lagper_ablation.sh
```

Both scripts default to `EPS=0.6`. A tuning run can set a shared value for all
four groups with `EPS=0.55 ... bash scripts/run_agreid_ablation.sh` (or the
LAGPeR script). Use a distinct `LOGS_DIR` for each value. Direct entrypoints
continue to accept `--eps VALUE`.

The standalone test scripts parameterize all paths and runtime loader settings. AG-ReID reports A to G and G to A. LAGPeR reports A to G, G to A, and G to A+G:

```bash
python test_agreid.py --data-dir /path/to/AG-ReID-root --checkpoint /path/to/stage2/model_final.pth.tar --pretrained-resnet50 /path/to/resnet50-19c8e357.pth --trial 1 --batch-size 64 --workers 8
python test_LAG.py --data-dir /path/to/LAGPeR-root --checkpoint /path/to/stage2/model_final.pth.tar --pretrained-resnet50 /path/to/resnet50-19c8e357.pth --trial 1 --batch-size 64 --workers 8
```

## Verification gates

Unit tests cover disabled CM equivalence, old-center hardness, candidate count/ties, filtered/duplicated reliability alignment, outliers, sparse lineage split/merge/threshold, persistence and partner switch, pre-exp calibration, checkpoint restoration and large sparse overlap construction. `--dry-run` validates the four CLI flag combinations for each entrypoint without loading data. Actual training smoke runs require dataset images and CUDA at the paths expected by the supplied adapters.
