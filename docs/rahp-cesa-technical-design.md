# RAHP / CESA technical design against this PCLHD tree

## Repository audit and scope

The starting tree contained only SYSU/RegDB. The user supplied `C:/Users/Jaxon/Desktop/fsdownload/PCLHD2` as the AG-ReID/LAGPeR reference. Its `train_agreid.py.bak_agva` and `train_lag_backup_before_agva.py` are the selected dual-domain training baselines; `clustercontrast/datasets/{agreid_ir,agreid_rgb,lag_ir,lag_rgb}.py`, `test_agreid.py`, `test_LAG.py`, and `prepare_lag.py` supply the existing dataset and evaluation rules. These files are copied into this tree without inventing train/query/gallery splits. The VI scripts remain the PCLHD ground-truth reference. The two new methods share one implementation each across both dataset entrypoints.

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

## CESA shared API and algorithm

`clustercontrast/methods/cesa.py` stores only previous aerial/ground label vectors, sparse matched-edge persistence and the next Stage 2 epoch index. The overlap contingency contains only observed pairs from aligned previous/current sample labels; `-1` is ignored in overlaps. Jaccard uses independently counted previous/current cluster sizes. A current cluster inherits a previous cluster only if both directions select one another as their unique highest-Jaccard match and Jaccard reaches the threshold. Ambiguous ties reset history, so a split or merge cannot copy one trajectory into multiple clusters.

At PGM's raw-cosine point, surviving previous edge keys are mapped through the two lineage maps into a sparse history boost. `calibrated_score = raw_cosine + min(stage2_epoch / warmup, 1) * eta * H_minus`. The caller then executes its existing `exp()`, reciprocal cost, Hungarian and unmatched logic unchanged. In these two entrypoints PGM is inline and `R` already remains in caller scope, so there is no return API to change. Source PGM rows are ground and columns are aerial; the integration transposes the score for CESA's aerial/ground state and converts each final `R` pair from `(ground, aerial)` to `(aerial, ground)`. `r2i/i2r` and the trainer's cross loss do not change. The first Stage 2 epoch has zero boost. After matching, each edge in final `R` receives `rho * H_prev + (1-rho)` only when its pair of lineage predecessors was a previous matched edge; otherwise it receives `1-rho`. Switching partner therefore restarts persistence. Checkpoint `cesa_state` contains `prev_labels_aerial`, `prev_labels_ground`, `edge_persistence`, and `stage2_epoch`; CESA checkpoints also store model, optimizer, scheduler, RNG and clustering eps for resume.

When `num_cluster_rgb < num_cluster_ir`, the frozen baseline does not execute PGM and retains `R=[]`, `r2i={}`, and `i2r={}`. CESA calls `advance_without_matching`: it records the current aligned aerial and ground pseudo labels, clears sparse edge persistence, and increments `stage2_epoch`. Its diagnostic reports `pgm_executed=False` and zero edges. This makes the next lineage transition compare adjacent epochs and prevents stale edges from crossing a no-match epoch.

## Checkpoint and evaluation protocol

Formal experiment checkpoint: **fixed final epoch**. Stage 1 and Stage 2 write `checkpoint.pth.tar` after every epoch and `model_final.pth.tar` only after the configured final epoch. Stage 2 initializes from the Stage 1 `model_final.pth.tar`. `--eval-during-train=False` is the default, so formal training does not read the test split each epoch. Enabling it is a debug-only logging option; Rank-1, mAP and mINP do not affect saving, resume, initialization, early stopping, or any training branch. Final evaluation is a separate command and loads the Stage 2 `model_final.pth.tar`.

RAHP Stage 1 requires `--memorybank CMhybrid`. An explicit `--use-rahp` with Stage 1 and any other memory bank raises `ValueError`. `--stage2-only --use-rahp` remains valid because it skips Stage 1.

## CLI, ablations, and boundaries

The shared CLI helper defines `--use-rahp`, `--rahp-beta` (0.25), `--rahp-knn` (20), `--rahp-alpha` (0.5), `--use-cesa`, `--cesa-rho` (0.8), `--cesa-eta` (0.1), `--cesa-lineage-thr` (0.5), and `--cesa-warmup` (5). It derives `baseline`, `rahp`, `cesa`, or `rahp_cesa` from the two flags. Both AG-ReID and LAGPeR RGB/RGB entrypoints omit `ChannelExchange`, `ChannelAdapGray`, and related VI channel simulation by default while retaining the established ordinary transforms. `agw` remains the only accepted architecture in these entrypoints and uses ResNet-50. DBSCAN, PGM unmatched completion, and evaluation protocols remain outside this method change.

The AG-ReID data root follows the supplied adapter's `agreid_ir/aerial_modify/bounding_box_train` and `agreid_rgb/ground_modify/bounding_box_train` layout plus its existing test/index paths. LAGPeR uses `aerial_modify/<trial>/bounding_box_train` and `ground_modify/<trial>/bounding_box_train` beneath one dataset root plus its existing test/index paths. For comparable Stage 1 hard-proxy ablations, use `--memorybank CMhybrid` in every group:

```bash
python train_agreid.py --data-dir /path/to/AG-ReID-adapter-root --memorybank CMhybrid --eval-during-train=False
python train_agreid.py --data-dir /path/to/AG-ReID-adapter-root --memorybank CMhybrid --eval-during-train=False --use-rahp
python train_agreid.py --data-dir /path/to/AG-ReID-adapter-root --memorybank CMhybrid --eval-during-train=False --use-cesa
python train_agreid.py --data-dir /path/to/AG-ReID-adapter-root --memorybank CMhybrid --eval-during-train=False --use-rahp --use-cesa

python train_lag.py --data-dir /path/to/LAGPeR-root --memorybank CMhybrid --eval-during-train=False
python train_lag.py --data-dir /path/to/LAGPeR-root --memorybank CMhybrid --eval-during-train=False --use-rahp
python train_lag.py --data-dir /path/to/LAGPeR-root --memorybank CMhybrid --eval-during-train=False --use-cesa
python train_lag.py --data-dir /path/to/LAGPeR-root --memorybank CMhybrid --eval-during-train=False --use-rahp --use-cesa
```

The standalone test scripts parameterize all paths and runtime loader settings. AG-ReID reports A to G and G to A. LAGPeR reports A to G, G to A, and G to A+G:

```bash
python test_agreid.py --data-dir /path/to/AG-ReID-root --checkpoint /path/to/stage2/model_final.pth.tar --trial 1 --batch-size 64 --workers 8
python test_LAG.py --data-dir /path/to/LAGPeR-root --checkpoint /path/to/stage2/model_final.pth.tar --trial 1 --batch-size 64 --workers 8
```

## Verification gates

Unit tests cover disabled CM equivalence, old-center hardness, candidate count/ties, filtered/duplicated reliability alignment, outliers, sparse lineage split/merge/threshold, persistence and partner switch, pre-exp calibration, checkpoint restoration and large sparse overlap construction. `--dry-run` validates the four CLI flag combinations for each entrypoint without loading data. Actual training smoke runs require dataset images and CUDA at the paths expected by the supplied adapters.
