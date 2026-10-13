# G2APS-ReID P3-A Dataset & Evaluation Adapter

## Scope and status

P3-A adds read-only G2APS training adapters, fixed P1 protocol parsing, explicit domain metadata, and CPU-only validation. It does not execute feature extraction, Jaccard distance, DBSCAN, Stage 1/Stage 2 training, or formal evaluation. It does not modify Cluster Memory, losses, D-AGVA, or the three-domain matching implementation.

Local status: **LOCAL_TESTS_PASSED**. The formal Linux validation has also been reported as **G2APS_P3A_VALIDATED** on `/data/home/lab338/Jaxon/dataset/G2APS_ReID_P1`.

## Modified source files

- `clustercontrast/datasets/g2aps_common.py`: strict filename parsing, identity-free training tuples, train/test isolation checks, deterministic ordering, per-row domain metadata, and pseudo-label replacement contract.
- `clustercontrast/datasets/g2aps_aerial.py`: c0/aerial training adapter.
- `clustercontrast/datasets/g2aps_ground.py`: c3/ground training adapter.
- `clustercontrast/datasets/g2aps_protocol.py`: fixed A→G+ and G→A+ protocol reader and validator.
- `clustercontrast/datasets/__init__.py`: registers `g2aps_aerial` and `g2aps_ground` without changing existing registry keys.
- `tools/validate_g2aps_adapter.py`: read-only real-data validation and bounded image-decode smoke test.
- `tests/test_g2aps_dataset_adapter.py`: synthetic regression tests.

## Adapter contracts

### Training

Both registered training adapters return the existing PCLHD tuple contract:

```text
(relative_image_path, training_label, camera_id)
```

`g2aps_aerial` accepts only c0 and expects 65,482 rows. `g2aps_ground` accepts only c3 and expects 35,389 rows. All returned training labels are the unsupervised placeholder `-1`; the ground-truth PID parsed from the filename is used transiently only for integrity checks and is not retained in the public training tuple or domain metadata. Images are not decoded by the adapter. `Preprocessor` opens an image only when its row is indexed.

The adapters recursively scan only `bounding_box_train`, sort paths lexicographically, validate the PID directory against the filename, reject PID 5555, reject unsupported cameras and malformed names, detect duplicate paths/records/physical files, and verify that valid training PIDs do not overlap PIDs found under the test/query trees.

Each row has aligned `G2APSDomainMetadata(relative_path, camid, domain)`. `domain_flags` uses the same row order, and initialization enforces:

```text
len(domain_flags) == len(train) == number of future feature rows
```

P3-A exposes this metadata; P3-B now consumes the aligned integer flags in ALL-Jaccard, prototype construction and label translation.

### Fixed evaluation

`G2APSProtocolDataset(root, protocol="A2G+")` and `protocol="G2A+"` read P1 lists directly. They never regenerate Query images. The parser validates all four protocol files against the counts and SHA256 digests in `protocol_metadata.json` before constructing either protocol.

Expected formal counts are:

| Protocol | Query | Gallery | Total list entries |
|---|---:|---:|---:|
| A→G+ | 4,876 c0 | 37,202 c3 | 42,078 |
| G→A+ | 4,876 c3 | 62,791 c0 | 67,667 |

The experiment list must begin with its fixed query list. Remaining entries are the gallery. The adapter checks canonical relative paths, file existence, filename/folder PID consistency, camera/domain direction, query/gallery path disjointness, 1,219 query identities, and at least one legal opposite-camera positive for every query. PID 5555 is accepted only from `bounding_box_gallery` and only as a gallery distractor. The adapter does not define G→A+G.

Because query and gallery cameras are opposite by construction, valid same-PID positives are not removed by the standard same-PID/same-camera filtering rule. No SYSU/RegDB-specific evaluation rule is applied by this adapter.

## Proof of training PID isolation

The existing PCLHD engine first runs feature extraction on `sorted(dataset.train)`. Feature extraction indexes features by path and does not use the tuple PID as a supervision target. After DBSCAN, the engine rebuilds each training row as `(fname, pseudo_label, camid)` and drops noise label `-1` before constructing `RandomMultipleGallerySampler`, Cluster Memory, and training loaders. Therefore:

1. the adapter exposes only `-1`, never the filename PID;
2. the raw tuples are suitable for ordered feature extraction but intentionally unsuitable for identity sampling;
3. the DBSCAN labels replace the placeholder before the sampler and losses are created;
4. the helper `replace_unsupervised_labels` and its regression test encode this boundary explicitly;
5. real PIDs remain available only in the evaluation adapter, where they are required for CMC/mAP/mINP computation.

No test PID or retrieval score is used for training, model selection, or hyperparameter selection in P3-A.

## Automated test coverage

The test suite covers registry preservation, c0/c3 splitting, malformed names, PID 5555 rejection, train/test PID isolation, deterministic ordering, identity-free training labels, pseudo-label replacement, domain-row alignment, protocol count/SHA validation, missing files, query/gallery overlap, legal positives, PID 5555 distractor semantics, lazy adapter construction, and compatibility with `Preprocessor`.

Run locally or on the server:

```bash
cd /data/home/lab338/Jaxon/PCLHD_repro_release
python -m unittest discover -s tests -p 'test_g2aps_dataset_adapter.py' -v
```

## Formal server validation

This command performs path/metadata validation over the complete P1 tree and decodes only two images from each requested split. It does not load a neural network or run clustering/training.

```bash
cd /data/home/lab338/Jaxon/PCLHD_repro_release
python tools/validate_g2aps_adapter.py \
  --data-dir /data/home/lab338/Jaxon/dataset/G2APS_ReID_P1 \
  --smoke-images 2
```

Success is reported only when the JSON status is `G2APS_P3A_VALIDATED`, the training counts are 65,482/35,389, protocol counts match P1 metadata, every query has a legal positive, and all row-alignment checks pass.

## Remaining P3-C blockers

1. P3-B has replaced G2APS path-string domain inference with explicit row-aligned metadata and shortened the caller-side distance lifecycle, but the current dense Jaccard/DBSCAN implementation remains infeasible for 65,482, 35,389, and especially 100,871 samples on 24 GB VRAM/64 GB RAM.
2. The training entry points do not yet expose G2APS dataset names, fixed A→G+/G→A+ evaluation construction, or a safe staged smoke-run configuration.
3. No formal model evaluation is authorized in P3-A/P3-B. Checkpoint policy and metric integration must be reviewed before P3-C.
