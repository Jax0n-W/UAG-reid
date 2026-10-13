#!/usr/bin/env python3
from __future__ import absolute_import, print_function

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from clustercontrast import datasets  # noqa: E402
from clustercontrast.datasets.g2aps_common import (  # noqa: E402
    UNSUPERVISED_TRAIN_PID,
    assert_domain_metadata_aligned,
)
from clustercontrast.datasets.g2aps_protocol import (  # noqa: E402
    G2APSProtocolDataset,
)
from clustercontrast.utils.data.preprocessor import Preprocessor  # noqa: E402


def smoke_read(dataset, count):
    if count <= 0:
        return []
    preprocessor = Preprocessor(dataset, root=None)
    results = []
    for index in range(min(count, len(dataset))):
        image, path, pid, camid, returned_index = preprocessor[index]
        results.append(
            {
                "path": path,
                "pid": pid,
                "camid": camid,
                "index": returned_index,
                "mode": image.mode,
                "size": list(image.size),
            }
        )
    return results


def absolute_samples(samples, root):
    root = Path(root)
    return [(str(root / path), pid, camid) for path, pid, camid in samples]


def main():
    parser = argparse.ArgumentParser(
        description="Read-only G2APS P3-A dataset/protocol validation"
    )
    parser.add_argument("--data-dir", required=True)
    parser.add_argument(
        "--smoke-images",
        type=int,
        default=2,
        help="Decode this many images per split after structural validation",
    )
    args = parser.parse_args()

    root = Path(args.data_dir).expanduser().resolve()
    aerial = datasets.create("g2aps_aerial", str(root), verbose=False)
    ground = datasets.create("g2aps_ground", str(root), verbose=False)
    a2g = G2APSProtocolDataset(str(root), "A2G+", verbose=False)
    g2a = G2APSProtocolDataset(str(root), "G2A+", verbose=False)

    for train_dataset in (aerial, ground):
        if any(pid != UNSUPERVISED_TRAIN_PID for _, pid, _ in train_dataset.train):
            raise AssertionError("Ground-truth PID entered a public training tuple")
        assert_domain_metadata_aligned(
            train_dataset.train, train_dataset.domain_metadata
        )

    result = {
        "status": "G2APS_P3A_VALIDATED",
        "data_root": str(root),
        "registry": {
            "g2aps_aerial": "g2aps_aerial" in datasets.names(),
            "g2aps_ground": "g2aps_ground" in datasets.names(),
        },
        "train": {
            "aerial": aerial.audit_summary,
            "ground": ground.audit_summary,
            "total_images": len(aerial.train) + len(ground.train),
            "ground_truth_pid_exposed_to_training": False,
        },
        "protocols": {
            "A2G+": asdict(a2g.summary),
            "G2A+": asdict(g2a.summary),
            "G2A+G_defined": False,
        },
        "domain_alignment": {
            "aerial_rows": len(aerial.domain_flags),
            "ground_rows": len(ground.domain_flags),
            "A2G+_query_rows": len(a2g.query_domain_flags),
            "A2G+_gallery_rows": len(a2g.gallery_domain_flags),
            "G2A+_query_rows": len(g2a.query_domain_flags),
            "G2A+_gallery_rows": len(g2a.gallery_domain_flags),
        },
        "image_smoke": {
            "train_aerial": smoke_read(
                absolute_samples(aerial.train, root), args.smoke_images
            ),
            "train_ground": smoke_read(
                absolute_samples(ground.train, root), args.smoke_images
            ),
            "A2G+_query": smoke_read(
                absolute_samples(a2g.query, root), args.smoke_images
            ),
            "A2G+_gallery": smoke_read(
                absolute_samples(a2g.gallery, root), args.smoke_images
            ),
            "G2A+_query": smoke_read(
                absolute_samples(g2a.query, root), args.smoke_images
            ),
            "G2A+_gallery": smoke_read(
                absolute_samples(g2a.gallery, root), args.smoke_images
            ),
        },
    }
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
