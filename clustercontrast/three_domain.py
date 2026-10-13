# -*- coding: utf-8 -*-
"""Utilities for the AG-ReID three-domain Stage 2 experiment.

The experiment keeps the original loss families and changes only the domain
partition and the BGM topology: A_far <-> A_near <-> Ground.
"""

from __future__ import absolute_import

import collections

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch.nn import functional as F

from .domain_metadata import (
    DOMAIN_AERIAL, DOMAIN_GROUND, SUBDOMAIN_AERIAL_FAR,
    SUBDOMAIN_AERIAL_NEAR, cross_domain_cluster_ids,
    generate_domain_balanced_cluster_features, validate_domain_flags,
    validate_subdomain_flags)


VALID_VIEW_GROUPS = ("A_far", "A_near")


def split_aerial_trainset(trainset, matched_score_map):
    """Split aerial samples with a complete, pre-matched view-score map."""
    far = []
    near = []
    missing = []
    invalid = []
    for sample in sorted(trainset):
        fname = sample[0]
        entry = matched_score_map.get(fname)
        if entry is None:
            missing.append(fname)
            continue
        group = entry.get("view_group")
        if group == "A_far":
            far.append(sample)
        elif group == "A_near":
            near.append(sample)
        else:
            invalid.append((fname, group))

    if missing:
        raise ValueError(
            "view score does not cover all aerial training images; missing={} "
            "example={}".format(len(missing), missing[0]))
    if invalid:
        raise ValueError(
            "invalid aerial view_group; expected A_far/A_near, got {} for {}".format(
                invalid[0][1], invalid[0][0]))
    if not far or not near:
        raise ValueError(
            "both A_far and A_near must be non-empty; far={}, near={}".format(
                len(far), len(near)))
    return far, near


def build_aerial_subdomain_flags(aerial_train, far_paths, near_paths):
    """Build Far/Near metadata without overloading physical camera/domain IDs."""
    far_paths = set(far_paths)
    near_paths = set(near_paths)
    overlap = far_paths & near_paths
    if overlap:
        raise ValueError("Far/Near path sets overlap: {}".format(sorted(overlap)[:20]))
    flags = []
    missing = []
    for fname, _, _ in aerial_train:
        if fname in far_paths:
            flags.append(SUBDOMAIN_AERIAL_FAR)
        elif fname in near_paths:
            flags.append(SUBDOMAIN_AERIAL_NEAR)
        else:
            missing.append(fname)
    if missing:
        raise ValueError("Far/Near metadata misses {} aerial rows; first={}".format(
            len(missing), missing[0]))
    validate_subdomain_flags(flags, [DOMAIN_AERIAL] * len(flags),
                             "aerial Far/Near flags")
    return tuple(flags)


@torch.no_grad()
def build_bidirectional_bgm(source_features, target_features):
    """Build total mappings in both directions for a rectangular BGM.

    Hungarian matching supplies the one-to-one core.  Any clusters on the
    larger side are mapped to their nearest prototype on the other side, so
    every pseudo-label used by the alternating cross loss has a valid target.
    """
    if source_features.ndim != 2 or target_features.ndim != 2:
        raise ValueError("BGM feature matrices must be two-dimensional")
    if source_features.size(0) == 0 or target_features.size(0) == 0:
        raise ValueError("BGM requires at least one cluster on each side")
    if source_features.size(1) != target_features.size(1):
        raise ValueError("BGM feature dimensions do not match")

    source = F.normalize(source_features, dim=1)
    target = F.normalize(target_features, dim=1)
    similarity = source.mm(target.t())
    cost = (1.0 / similarity.exp()).cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(cost)

    source_to_target = {
        int(source_index): int(target_index)
        for source_index, target_index in zip(row_ind, col_ind)
    }
    target_to_source = {
        int(target_index): int(source_index)
        for source_index, target_index in zip(row_ind, col_ind)
    }

    nearest_target = similarity.argmax(dim=1).cpu().tolist()
    nearest_source = similarity.argmax(dim=0).cpu().tolist()
    for source_index, target_index in enumerate(nearest_target):
        source_to_target.setdefault(source_index, int(target_index))
    for target_index, source_index in enumerate(nearest_source):
        target_to_source.setdefault(target_index, int(source_index))

    stats = {
        "source_clusters": int(source_features.size(0)),
        "target_clusters": int(target_features.size(0)),
        "hungarian_pairs": int(len(row_ind)),
        "source_fallbacks": int(source_features.size(0) - len(row_ind)),
        "target_fallbacks": int(target_features.size(0) - len(row_ind)),
    }
    return source_to_target, target_to_source, stats


def compose_three_domain_primary_loss(loss_far, loss_near, loss_ground,
                                      cross_far_near, cross_near_ground,
                                      far_weight, chain_cross_weight=0.25):
    """Compose optimization one without introducing a new loss function."""
    far_weight = float(far_weight)
    if not 0.0 <= far_weight <= 1.0:
        raise ValueError("far_weight must be in [0, 1]")
    near_weight = 1.0 - far_weight
    chain_cross = 0.5 * (cross_far_near + cross_near_ground)
    return (far_weight * loss_far + near_weight * loss_near + loss_ground
            + float(chain_cross_weight) * chain_cross)


def compose_three_domain_all_loss(loss_all_far, loss_all_near,
                                  loss_all_ground, far_weight):
    """Keep the original aerial-vs-ground balance in optimization two."""
    far_weight = float(far_weight)
    if not 0.0 <= far_weight <= 1.0:
        raise ValueError("far_weight must be in [0, 1]")
    return (far_weight * loss_all_far
            + (1.0 - far_weight) * loss_all_near
            + loss_all_ground)


@torch.no_grad()
def generate_cluster_features(labels, features):
    centers = collections.defaultdict(list)
    for index, label in enumerate(labels):
        label = int(label)
        if label != -1:
            centers[label].append(features[index])
    if not centers:
        return torch.empty(0, features.size(1), device=features.device)
    return torch.stack([
        torch.stack(centers[label], dim=0).mean(0)
        for label in sorted(centers)
    ], dim=0)


@torch.no_grad()
def generate_random_features(labels, features, num_clusters, num_instances):
    """Create the CMhard EMA sample bank with integer-safe indexing."""
    indexes = np.empty(num_clusters * num_instances, dtype=np.int64)
    labels = np.asarray(labels)
    for cluster_id in range(num_clusters):
        slots = [cluster_id + k * num_clusters for k in range(num_instances)]
        samples = np.random.choice(
            np.where(labels == cluster_id)[0], num_instances, replace=True)
        indexes[slots] = samples
    index_tensor = torch.as_tensor(indexes, dtype=torch.long,
                                   device=features.device)
    return features.index_select(0, index_tensor)


@torch.no_grad()
def generate_joint_cluster_features(labels, num_clusters, features,
                                    domain_flags):
    """Create one ALL prototype bank with equal aerial/ground contribution.

    Far and Near are first pooled into one aerial mean.  A joint cluster with
    both modalities is therefore 0.5 * aerial_mean + 0.5 * ground_mean, exactly
    preserving the original Stage 2 modality balance.
    """
    return generate_domain_balanced_cluster_features(
        labels, num_clusters, features, domain_flags)


def joint_cross_modal_flags(labels, domain_flags):
    """Return cluster ids containing ground and at least one aerial subgroup."""
    return cross_domain_cluster_ids(labels, domain_flags)
