from __future__ import absolute_import

from dataclasses import dataclass
import os.path as osp
import re

import numpy as np
import torch


DOMAIN_AERIAL = 0
DOMAIN_GROUND = 3
VALID_DOMAIN_FLAGS = (DOMAIN_AERIAL, DOMAIN_GROUND)

SUBDOMAIN_AERIAL_FAR = "A_far"
SUBDOMAIN_AERIAL_NEAR = "A_near"
SUBDOMAIN_GROUND = "Ground"
VALID_SUBDOMAIN_FLAGS = (
    SUBDOMAIN_AERIAL_FAR,
    SUBDOMAIN_AERIAL_NEAR,
    SUBDOMAIN_GROUND,
)

_G2APS_NAME = re.compile(r"^\d+_c[03]v\d+_\d+\.jpg$", re.IGNORECASE)
_CARGO_NAME = re.compile(r"^Cam(?P<camera>\d+)_", re.IGNORECASE)


class MissingDomainMetadataError(ValueError):
    pass


@dataclass(frozen=True)
class DomainRows:
    samples: tuple
    flags: tuple

    def __post_init__(self):
        validate_domain_flags(self.flags, len(self.samples), "DomainRows")


def _coerce_domain_flag(value, context):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("{} uses a boolean instead of physical domain 0/3".format(context))
    if hasattr(value, "item"):
        value = value.item()
    try:
        value = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("{} contains a non-integer domain flag: {!r}".format(
            context, value)) from exc
    if value not in VALID_DOMAIN_FLAGS:
        raise ValueError(
            "{} contains invalid domain flag {}; expected 0 (Aerial) or 3 "
            "(Ground)".format(context, value))
    return value


def validate_domain_flags(flags, expected_length=None, context="domain flags"):
    if flags is None:
        raise MissingDomainMetadataError("{} are missing".format(context))
    normalized = tuple(
        _coerce_domain_flag(value, "{}[{}]".format(context, index))
        for index, value in enumerate(flags)
    )
    if expected_length is not None and len(normalized) != int(expected_length):
        raise ValueError(
            "{} length {} does not match feature rows {}".format(
                context, len(normalized), expected_length))
    return normalized


def infer_legacy_domain_from_path(path):
    """Infer domains only for legacy datasets with established path semantics."""
    normalized = str(path).replace("\\", "/").lower()
    basename = osp.basename(normalized)
    if _G2APS_NAME.fullmatch(basename):
        raise MissingDomainMetadataError(
            "G2APS samples require explicit domain metadata; refusing path fallback: "
            "{}".format(path))
    if "aerial_modify" in normalized or "ir_modify" in normalized:
        return DOMAIN_AERIAL
    if "ground_modify" in normalized or "rgb_modify" in normalized:
        return DOMAIN_GROUND
    cargo = _CARGO_NAME.match(osp.basename(str(path)))
    if cargo is not None:
        camera = int(cargo.group("camera"))
        if 1 <= camera <= 5:
            return DOMAIN_AERIAL
        if 6 <= camera <= 13:
            return DOMAIN_GROUND
    raise MissingDomainMetadataError(
        "Cannot infer a legacy domain from path {!r}; explicit metadata is "
        "required".format(path))


def infer_legacy_domain_flags(samples):
    return validate_domain_flags(
        [infer_legacy_domain_from_path(sample[0]) for sample in samples],
        len(samples), "legacy inferred domain flags")


def _metadata_flag(item):
    if hasattr(item, "camid"):
        return item.camid
    if isinstance(item, dict) and "camid" in item:
        return item["camid"]
    raise ValueError("Domain metadata entry does not expose camid: {!r}".format(item))


def _metadata_path(item):
    if hasattr(item, "relative_path"):
        return item.relative_path
    if isinstance(item, dict) and "relative_path" in item:
        return item["relative_path"]
    raise ValueError(
        "Domain metadata entry does not expose relative_path: {!r}".format(item))


def domain_flags_for_samples(dataset, samples, allow_legacy=True):
    """Return physical-domain flags aligned to an arbitrary ordered sample view."""
    samples = tuple(samples)
    metadata = getattr(dataset, "domain_metadata", None)
    declared_flags = getattr(dataset, "domain_flags", None)
    if metadata is not None:
        if len(metadata) != len(getattr(dataset, "train", ())):
            raise ValueError("dataset.domain_metadata is not aligned with dataset.train")
        source = {}
        for item in metadata:
            path = _metadata_path(item)
            if path in source:
                raise ValueError("Duplicate path in dataset domain metadata: {}".format(path))
            source[path] = _coerce_domain_flag(
                _metadata_flag(item), "domain metadata for {}".format(path))
        flags = []
        for path, _, _ in samples:
            if path not in source:
                raise MissingDomainMetadataError(
                    "Explicit domain metadata does not cover sample {}".format(path))
            flags.append(source[path])
        flags = validate_domain_flags(flags, len(samples), "aligned domain flags")
        if declared_flags is not None:
            original_flags = validate_domain_flags(
                declared_flags, len(getattr(dataset, "train", ())),
                "dataset.domain_flags")
            original_samples = tuple(getattr(dataset, "train", ()))
            for index, ((path, _, _), flag) in enumerate(
                    zip(original_samples, original_flags)):
                if source.get(path) != flag:
                    raise ValueError(
                        "dataset.domain_flags conflicts with metadata at row {}".format(
                            index))
        return flags
    if declared_flags is not None:
        original_samples = tuple(getattr(dataset, "train", ()))
        original_flags = validate_domain_flags(
            declared_flags, len(original_samples), "dataset.domain_flags")
        mapping = {}
        for sample, flag in zip(original_samples, original_flags):
            if sample[0] in mapping:
                raise ValueError("Duplicate sample path in dataset.train: {}".format(
                    sample[0]))
            mapping[sample[0]] = flag
        try:
            return validate_domain_flags(
                [mapping[sample[0]] for sample in samples], len(samples),
                "aligned dataset.domain_flags")
        except KeyError as exc:
            raise MissingDomainMetadataError(
                "dataset.domain_flags do not cover {}".format(exc.args[0])) from exc
    if not allow_legacy:
        raise MissingDomainMetadataError("Explicit domain metadata is required")
    return infer_legacy_domain_flags(samples)


def concatenate_domain_rows(*parts):
    samples = []
    flags = []
    for index, part in enumerate(parts):
        if isinstance(part, DomainRows):
            part_samples, part_flags = part.samples, part.flags
        else:
            part_samples, part_flags = part
        part_samples = tuple(part_samples)
        part_flags = validate_domain_flags(
            part_flags, len(part_samples), "concat part {}".format(index))
        samples.extend(part_samples)
        flags.extend(part_flags)
    return DomainRows(tuple(samples), tuple(flags))


def reorder_domain_rows(samples, flags, indices):
    samples = tuple(samples)
    flags = validate_domain_flags(flags, len(samples), "reorder input flags")
    indices = tuple(int(index) for index in indices)
    for index in indices:
        if index < 0 or index >= len(samples):
            raise IndexError("Domain row index out of range: {}".format(index))
    return DomainRows(
        tuple(samples[index] for index in indices),
        tuple(flags[index] for index in indices))


def filter_domain_rows(samples, flags, keep_mask):
    samples = tuple(samples)
    flags = validate_domain_flags(flags, len(samples), "filter input flags")
    keep_mask = tuple(bool(value) for value in keep_mask)
    if len(keep_mask) != len(samples):
        raise ValueError("Filter mask length does not match domain rows")
    indices = [index for index, keep in enumerate(keep_mask) if keep]
    return reorder_domain_rows(samples, flags, indices)


def carry_domain_flags_by_basename(source_samples, source_flags, target_samples):
    """Carry explicit flags across a path-only remap preserving basenames."""
    source_samples = tuple(source_samples)
    target_samples = tuple(target_samples)
    source_flags = validate_domain_flags(
        source_flags, len(source_samples), "source remap domain flags")
    by_name = {}
    for sample, flag in zip(source_samples, source_flags):
        basename = osp.basename(sample[0])
        if basename in by_name:
            raise ValueError(
                "Source domain rows contain duplicate basename: {}".format(
                    basename))
        by_name[basename] = flag
    output = []
    seen = set()
    for sample in target_samples:
        basename = osp.basename(sample[0])
        if basename in seen:
            raise ValueError(
                "Target domain rows contain duplicate basename: {}".format(
                    basename))
        seen.add(basename)
        if basename not in by_name:
            raise MissingDomainMetadataError(
                "Path-remapped sample has no source domain row: {}".format(
                    sample[0]))
        output.append(by_name[basename])
    if len(output) != len(source_samples) or seen != set(by_name):
        missing = sorted(set(by_name) - seen)
        raise MissingDomainMetadataError(
            "Path remap changed the domain-row population: source={}, target={}, "
            "first_missing={}".format(
                len(source_samples), len(output),
                missing[0] if missing else "none"))
    return validate_domain_flags(
        output, len(target_samples), "path-remapped domain flags")


def validate_subdomain_flags(subdomain_flags, domain_flags, context="subdomain flags"):
    domain_flags = validate_domain_flags(
        domain_flags, len(subdomain_flags), "{} physical domains".format(context))
    normalized = tuple(subdomain_flags)
    for index, (subdomain, domain) in enumerate(zip(normalized, domain_flags)):
        if subdomain not in VALID_SUBDOMAIN_FLAGS:
            raise ValueError("{}[{}] is invalid: {!r}".format(context, index, subdomain))
        if domain == DOMAIN_GROUND and subdomain != SUBDOMAIN_GROUND:
            raise ValueError("Ground row {} cannot be assigned to {}".format(
                index, subdomain))
        if domain == DOMAIN_AERIAL and subdomain == SUBDOMAIN_GROUND:
            raise ValueError("Aerial row {} cannot be assigned to Ground".format(index))
    return normalized


@torch.no_grad()
def generate_domain_balanced_cluster_features(labels, num_clusters, features,
                                              domain_flags):
    domain_flags = validate_domain_flags(
        domain_flags, len(labels), "prototype domain flags")
    if len(labels) != features.size(0):
        raise ValueError("Cluster labels do not match feature rows")
    domain_features = {
        DOMAIN_AERIAL: {},
        DOMAIN_GROUND: {},
    }
    for index, raw_label in enumerate(labels):
        label = int(raw_label)
        if label == -1:
            continue
        bucket = domain_features[domain_flags[index]].setdefault(label, [])
        bucket.append(features[index])

    centers = []
    for cluster_id in range(int(num_clusters)):
        aerial_rows = domain_features[DOMAIN_AERIAL].get(cluster_id, [])
        ground_rows = domain_features[DOMAIN_GROUND].get(cluster_id, [])
        aerial_mean = torch.stack(aerial_rows).mean(0) if aerial_rows else None
        ground_mean = torch.stack(ground_rows).mean(0) if ground_rows else None
        if aerial_mean is None and ground_mean is None:
            raise ValueError("Cluster {} has no feature rows".format(cluster_id))
        if aerial_mean is None:
            centers.append(ground_mean)
        elif ground_mean is None:
            centers.append(aerial_mean)
        else:
            centers.append(0.5 * (aerial_mean + ground_mean))
    return torch.stack(centers, dim=0)


def cross_domain_cluster_ids(labels, domain_flags):
    domain_flags = validate_domain_flags(
        domain_flags, len(labels), "cross-domain cluster flags")
    aerial = set()
    ground = set()
    for raw_label, domain in zip(labels, domain_flags):
        label = int(raw_label)
        if label == -1:
            continue
        (aerial if domain == DOMAIN_AERIAL else ground).add(label)
    return aerial & ground


def partition_cross_domain_pseudo_samples(samples, labels, domain_flags):
    """Translate ALL labels into aerial/ground pseudo sets using only flags."""
    samples = tuple(samples)
    if len(samples) != len(labels):
        raise ValueError("ALL labels do not match sample rows")
    domain_flags = validate_domain_flags(
        domain_flags, len(samples), "ALL label-translation domain flags")
    valid_clusters = cross_domain_cluster_ids(labels, domain_flags)
    aerial = []
    ground = []
    for (fname, _, camid), raw_label, domain in zip(
            samples, labels, domain_flags):
        label = int(raw_label)
        if label == -1 or label not in valid_clusters:
            continue
        target = aerial if domain == DOMAIN_AERIAL else ground
        target.append((fname, label, camid))
    return aerial, ground, valid_clusters
