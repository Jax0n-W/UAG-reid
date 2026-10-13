# -*- coding: utf-8 -*-
"""Utilities for using offline AGVA images in three-domain Stage 2.

The view split, clustering and pseudo labels keep using original images.  This
module only remaps paths after pseudo-label generation, so all aerial loss
loaders share one deterministic original/AGVA decision per image and epoch.
"""

from __future__ import absolute_import
import os

import os.path as osp
import random


def validate_probability(probability):
    probability = float(probability)
    if not 0.0 <= probability <= 1.0:
        raise ValueError("AGVA probability must be in [0, 1]")
    return probability


def build_candidate_map(samples, candidate_builder, require_complete=True):
    """Map each unique original image to its offline AGVA counterpart."""
    candidate_map = {}
    missing = []
    for fname in sorted({sample[0] for sample in samples}):
        candidate = candidate_builder(fname)
        if osp.isfile(candidate):
            candidate_map[fname] = candidate
        else:
            candidate_map[fname] = fname
            missing.append((fname, candidate))
    if require_complete and missing:
        examples = "; ".join(
            "{} -> {}".format(source, target)
            for source, target in missing[:3])
        raise FileNotFoundError(
            "AGVA images do not completely cover the aerial training set: "
            "missing={}/{}; examples: {}".format(
                len(missing), len(candidate_map), examples))
    return candidate_map, missing


def select_epoch_paths(candidate_map, probability, seed):
    """Create one reproducible path decision shared by every loss loader."""
    probability = validate_probability(probability)
    rng = random.Random(seed)
    selected = {}
    augmented = 0
    unavailable = 0
    for original in sorted(candidate_map):
        candidate = candidate_map[original]
        use_agva = rng.random() < probability
        if use_agva and candidate != original:
            selected[original] = candidate
            augmented += 1
        else:
            selected[original] = original
            unavailable += int(use_agva and candidate == original)
    stats = {
        "total": len(selected),
        "augmented": augmented,
        "original": len(selected) - augmented,
        "missing_fallback": unavailable,
    }
    return selected, stats


def select_stratified_epoch_paths(samples, candidate_map, far_paths,
                                  near_paths, probability, seed,
                                  group_probabilities=None):
    """Select an exact AGVA quota independently in each camera/view group.

    The returned mapping is created once per epoch and is intended to be
    shared by the Far, Near, ALL-Far and ALL-Near loss loaders. View grouping
    and clustering remain based on the original ``samples``.
    """
    probability = validate_probability(probability)
    if group_probabilities is None:
        group_probabilities = {"Far": probability, "Near": probability}
    else:
        if set(group_probabilities) != {"Far", "Near"}:
            raise ValueError(
                "group_probabilities must contain exactly Far and Near")
        group_probabilities = {
            group: validate_probability(group_probabilities[group])
            for group in ("Far", "Near")
        }
    far_paths = set(far_paths)
    near_paths = set(near_paths)
    if far_paths & near_paths:
        raise ValueError("Far and Near path sets must be disjoint")

    strata = {}
    seen = set()
    for fname, _, camera in samples:
        if fname in seen:
            raise ValueError(
                "duplicate aerial image path in samples: {}".format(fname))
        seen.add(fname)
        if fname in far_paths:
            group = "Far"
        elif fname in near_paths:
            group = "Near"
        else:
            raise ValueError(
                "aerial image is not assigned to Far or Near: {}".format(
                    fname))
        strata.setdefault((camera, group), []).append(fname)

    candidate_paths = set(candidate_map)
    if seen != candidate_paths:
        missing = sorted(seen - candidate_paths)
        extra = sorted(candidate_paths - seen)
        raise ValueError(
            "candidate map does not match aerial samples: missing={}, "
            "extra={}, first_missing={}, first_extra={}".format(
                len(missing), len(extra),
                missing[0] if missing else "none",
                extra[0] if extra else "none"))

    rng = random.Random(seed)
    selected = dict((fname, fname) for fname in sorted(seen))
    stratum_stats = {}
    augmented_total = 0
    unavailable_total = 0
    for camera, group in sorted(
            strata, key=lambda item: (str(item[0]), item[1])):
        originals = sorted(strata[(camera, group)])
        stratum_probability = group_probabilities[group]
        requested = int(len(originals) * stratum_probability + 0.5)
        available = [
            fname for fname in originals
            if candidate_map[fname] != fname
        ]
        chosen = set(rng.sample(available, min(requested, len(available))))
        for fname in chosen:
            selected[fname] = candidate_map[fname]
        augmented = len(chosen)
        unavailable = requested - augmented
        augmented_total += augmented
        unavailable_total += unavailable
        stratum_stats["camera={}|group={}".format(camera, group)] = {
            "total": len(originals),
            "probability": stratum_probability,
            "augmented": augmented,
            "original": len(originals) - augmented,
            "requested": requested,
            "missing_fallback": unavailable,
        }

    stats = {
        "total": len(selected),
        "augmented": augmented_total,
        "original": len(selected) - augmented_total,
        "missing_fallback": unavailable_total,
        "strata": stratum_stats,
    }
    return selected, stats


def remap_pseudo_dataset(dataset, selected_paths):
    """Replace only image paths while preserving pseudo labels and cameras."""
    return [
        (selected_paths.get(fname, fname), label, camera)
        for fname, label, camera in dataset
    ]



def remap_samples_to_enhanced_root(samples, enhanced_root):
    """Replace every sample path with the same-basename enhanced image."""
    enhanced_root = osp.abspath(osp.expanduser(str(enhanced_root)))
    if not osp.isdir(enhanced_root):
        raise FileNotFoundError(
            "Enhanced aerial directory does not exist: {}".format(enhanced_root))
    by_name = {}
    duplicates = []
    for root, _, filenames in os.walk(enhanced_root):
        for filename in filenames:
            path = osp.join(root, filename)
            if filename in by_name:
                duplicates.append(filename)
            else:
                by_name[filename] = path
    if duplicates:
        raise ValueError(
            "Enhanced aerial directory contains duplicate basenames: {}".format(
                ", ".join(sorted(set(duplicates))[:5])))
    output = []
    missing = []
    for fname, label, camera in samples:
        enhanced = by_name.get(osp.basename(fname))
        if enhanced is None:
            missing.append(fname)
        else:
            output.append((enhanced, label, camera))
    if missing or len(output) != len(samples):
        raise FileNotFoundError(
            "Enhanced aerial directory does not completely cover training data: "
            "matched={}, expected={}, missing={}, first_missing={}".format(
                len(output), len(samples), len(missing),
                missing[0] if missing else "none"))
    return output
