#!/usr/bin/env python3
from __future__ import absolute_import, print_function

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from clustercontrast import datasets  # noqa: E402
from clustercontrast.domain_metadata import (  # noqa: E402
    DOMAIN_AERIAL,
    DOMAIN_GROUND,
    concatenate_domain_rows,
    domain_flags_for_samples,
)


def main():
    parser = argparse.ArgumentParser(
        description="Read-only G2APS P3-B explicit-domain validation")
    parser.add_argument("--data-dir", required=True)
    args = parser.parse_args()

    root = Path(args.data_dir).expanduser().resolve()
    aerial = datasets.create("g2aps_aerial", str(root), verbose=False)
    ground = datasets.create("g2aps_ground", str(root), verbose=False)
    aerial_train = sorted(aerial.train)
    ground_train = sorted(ground.train)
    aerial_flags = domain_flags_for_samples(aerial, aerial_train,
                                            allow_legacy=False)
    ground_flags = domain_flags_for_samples(ground, ground_train,
                                            allow_legacy=False)
    rows = concatenate_domain_rows(
        (ground_train, ground_flags), (aerial_train, aerial_flags))

    if set(aerial_flags) != {DOMAIN_AERIAL}:
        raise AssertionError("Aerial adapter contains a non-aerial domain flag")
    if set(ground_flags) != {DOMAIN_GROUND}:
        raise AssertionError("Ground adapter contains a non-ground domain flag")
    if len(rows.samples) != 100871 or len(rows.flags) != 100871:
        raise AssertionError("Combined G2APS domain rows do not equal 100,871")
    if any(pid != -1 for _, pid, _ in rows.samples):
        raise AssertionError("A real PID entered G2APS training rows")

    print(json.dumps({
        "status": "G2APS_P3B_DOMAIN_VALIDATED",
        "data_root": str(root),
        "aerial_rows": len(aerial_flags),
        "ground_rows": len(ground_flags),
        "all_rows": len(rows.flags),
        "aerial_flag": DOMAIN_AERIAL,
        "ground_flag": DOMAIN_GROUND,
        "real_pid_exposed": False,
        "jaccard_executed": False,
        "training_executed": False,
    }, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
