"""Sparse cross-epoch persistence of the actual PGM matched edge set."""

from dataclasses import dataclass, field
import math
import warnings

import numpy as np
import torch


def _labels(values):
    result = np.asarray(values, dtype=np.int64).reshape(-1)
    return result


def build_overlap_contingency(previous, current):
    """Count only observed overlaps in O(N) storage, ignoring -1 pairs."""
    previous = _labels(previous)
    current = _labels(current)
    if previous.shape != current.shape:
        raise ValueError("previous/current labels must have identical sample order and length")
    previous_values, previous_counts = np.unique(previous[previous >= 0], return_counts=True)
    current_values, current_counts = np.unique(current[current >= 0], return_counts=True)
    size_previous = dict(zip(previous_values.tolist(), previous_counts.tolist()))
    size_current = dict(zip(current_values.tolist(), current_counts.tolist()))
    valid = (previous >= 0) & (current >= 0)
    if not valid.any():
        return {}, size_previous, size_current
    pairs, counts = np.unique(np.stack((previous[valid], current[valid]), axis=1),
                              axis=0, return_counts=True)
    overlap = {(int(pair[0]), int(pair[1])): int(count)
               for pair, count in zip(pairs, counts)}
    return overlap, size_previous, size_current


def compute_mutual_lineage(previous, current, threshold=0.5):
    """Map current cluster -> previous cluster via unique mutual-best Jaccard."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("lineage threshold must be in [0, 1]")
    overlap, size_previous, size_current = build_overlap_contingency(previous, current)
    best_previous = {}
    best_current = {}
    ambiguous_previous = set()
    ambiguous_current = set()
    for (old, new), count in overlap.items():
        jaccard = count / (size_previous[old] + size_current[new] - count)
        if jaccard < threshold:
            continue
        prev_record = best_previous.get(new)
        if prev_record is None or jaccard > prev_record[0] and not math.isclose(jaccard, prev_record[0]):
            best_previous[new] = (jaccard, old)
            ambiguous_previous.discard(new)
        elif math.isclose(jaccard, prev_record[0]):
            ambiguous_previous.add(new)
        curr_record = best_current.get(old)
        if curr_record is None or jaccard > curr_record[0] and not math.isclose(jaccard, curr_record[0]):
            best_current[old] = (jaccard, new)
            ambiguous_current.discard(old)
        elif math.isclose(jaccard, curr_record[0]):
            ambiguous_current.add(old)
    lineage = {}
    for new, (_, old) in best_previous.items():
        if (new not in ambiguous_previous and old not in ambiguous_current
                and best_current.get(old, (None, None))[1] == new):
            lineage[new] = old
    return lineage


def build_history_boost(lineage_aerial, lineage_ground, previous_edges):
    """Map only surviving sparse previous edge keys to current score cells."""
    successor_a = {old: new for new, old in lineage_aerial.items()}
    successor_g = {old: new for new, old in lineage_ground.items()}
    boost = {}
    for (old_a, old_g), persistence in previous_edges.items():
        new_a = successor_a.get(old_a)
        new_g = successor_g.get(old_g)
        if new_a is not None and new_g is not None:
            boost[(new_a, new_g)] = float(persistence)
    return boost


def update_edge_persistence(matched_edges, lineage_aerial, lineage_ground,
                            previous_edges, rho=0.8):
    """Update persistence from final PGM R, never from label translations."""
    if not 0.0 <= rho < 1.0:
        raise ValueError("rho must be in [0, 1)")
    result = {}
    continued = 0
    switched = 0
    previous_a_partners = {}
    previous_g_partners = {}
    for old_a, old_g in previous_edges:
        previous_a_partners.setdefault(old_a, set()).add(old_g)
        previous_g_partners.setdefault(old_g, set()).add(old_a)
    for new_a, new_g in matched_edges:
        pair = (int(new_a), int(new_g))
        old_a = lineage_aerial.get(pair[0])
        old_g = lineage_ground.get(pair[1])
        predecessor = (old_a, old_g)
        if old_a is not None and old_g is not None and predecessor in previous_edges:
            result[pair] = rho * previous_edges[predecessor] + (1.0 - rho)
            continued += 1
        else:
            result[pair] = 1.0 - rho
            if ((old_a is not None and old_a in previous_a_partners) or
                    (old_g is not None and old_g in previous_g_partners)):
                switched += 1
    return result, continued, switched


@dataclass
class CESAPreparedEpoch:
    lineage_aerial: dict
    lineage_ground: dict
    applied_boosts: dict
    num_aerial_clusters: int
    num_ground_clusters: int


@dataclass
class CESAState:
    rho: float = 0.8
    eta: float = 0.1
    lineage_threshold: float = 0.5
    warmup: int = 5
    prev_labels_aerial: object = None
    prev_labels_ground: object = None
    edge_persistence: dict = field(default_factory=dict)
    stage2_epoch: int = 0

    def __post_init__(self):
        if self.warmup < 1 or self.eta < 0:
            raise ValueError("warmup >= 1 and eta >= 0 are required")
        if not 0.0 <= self.rho < 1.0 or not 0.0 <= self.lineage_threshold <= 1.0:
            raise ValueError("invalid rho or lineage threshold")

    def prepare(self, raw_cosine, current_aerial, current_ground):
        """Calibrate raw cosine before the caller's original PGM exp()."""
        aerial = _labels(current_aerial)
        ground = _labels(current_ground)
        if self.prev_labels_aerial is None or self.prev_labels_ground is None:
            lineage_a = {}
            lineage_g = {}
        else:
            lineage_a = compute_mutual_lineage(
                self.prev_labels_aerial, aerial, self.lineage_threshold)
            lineage_g = compute_mutual_lineage(
                self.prev_labels_ground, ground, self.lineage_threshold)
        boost = build_history_boost(lineage_a, lineage_g, self.edge_persistence)
        factor = min(self.stage2_epoch / self.warmup, 1.0) * self.eta
        calibrated = raw_cosine.clone() if boost and factor else raw_cosine
        applied = {}
        if boost and factor:
            for (a, g), value in boost.items():
                if a >= raw_cosine.size(0) or g >= raw_cosine.size(1):
                    raise ValueError("lineage cluster ID exceeds PGM score shape")
                applied[(a, g)] = factor * value
                calibrated[a, g] += applied[(a, g)]
        prepared = CESAPreparedEpoch(
            lineage_a, lineage_g, applied,
            len(np.unique(aerial[aerial >= 0])),
            len(np.unique(ground[ground >= 0])))
        return calibrated, prepared

    def complete(self, prepared, matched_edges, current_aerial, current_ground):
        """Commit final PGM edges, then advance sparse state and diagnostics."""
        edges, continued, switched = update_edge_persistence(
            matched_edges, prepared.lineage_aerial,
            prepared.lineage_ground, self.edge_persistence, self.rho)
        self.prev_labels_aerial = _labels(current_aerial).copy()
        self.prev_labels_ground = _labels(current_ground).copy()
        self.edge_persistence = edges
        self.stage2_epoch += 1
        values = list(edges.values())
        applied = list(prepared.applied_boosts.values())
        count = len(edges)
        return {
            'enabled': True,
            'valid_aerial_lineages': len(prepared.lineage_aerial),
            'valid_ground_lineages': len(prepared.lineage_ground),
            'aerial_lineage_survival_rate': len(prepared.lineage_aerial) / max(1, prepared.num_aerial_clusters),
            'ground_lineage_survival_rate': len(prepared.lineage_ground) / max(1, prepared.num_ground_clusters),
            'current_pgm_edge_count': count,
            'continued_edge_count': continued,
            'new_edge_count': count - continued,
            'partner_switch_count': switched,
            'partner_switch_rate': switched / max(1, count),
            'mean_historical_persistence': float(np.mean(values)) if values else 0.0,
            'max_persistence': max(values) if values else 0.0,
            'mean_applied_history_boost': float(np.mean(applied)) if applied else 0.0,
            'number_of_boosted_pgm_cells': len(applied),
        }

    def state_dict(self):
        return {'prev_labels_aerial': None if self.prev_labels_aerial is None else torch.from_numpy(self.prev_labels_aerial.copy()),
                'prev_labels_ground': None if self.prev_labels_ground is None else torch.from_numpy(self.prev_labels_ground.copy()),
                'edge_persistence': dict(self.edge_persistence),
                'stage2_epoch': int(self.stage2_epoch)}

    def load_state_dict(self, state):
        if not state:
            warnings.warn("Checkpoint has no cesa_state; starting with empty CESA history")
            self.prev_labels_aerial = None
            self.prev_labels_ground = None
            self.edge_persistence = {}
            self.stage2_epoch = 0
            return
        self.prev_labels_aerial = (None if state['prev_labels_aerial'] is None
                                   else _labels(state['prev_labels_aerial']).copy())
        self.prev_labels_ground = (None if state['prev_labels_ground'] is None
                                   else _labels(state['prev_labels_ground']).copy())
        self.edge_persistence = {
            (int(a), int(g)): float(value)
            for (a, g), value in state['edge_persistence'].items()}
        self.stage2_epoch = int(state['stage2_epoch'])


def format_cesa_epoch(diagnostics):
    d = diagnostics
    return ('[CESA] enabled={} lineages_a={} lineages_g={} survival_a={:.4f} '
            'survival_g={:.4f} edges={} continued={} new={} switches={} '
            'switch_rate={:.4f} mean_h={:.4f} max_h={:.4f} mean_boost={:.6f} '
            'boosted_cells={}').format(
                d['enabled'], d['valid_aerial_lineages'], d['valid_ground_lineages'],
                d['aerial_lineage_survival_rate'], d['ground_lineage_survival_rate'],
                d['current_pgm_edge_count'], d['continued_edge_count'],
                d['new_edge_count'], d['partner_switch_count'],
                d['partner_switch_rate'], d['mean_historical_persistence'],
                d['max_persistence'], d['mean_applied_history_boost'],
                d['number_of_boosted_pgm_cells'])


def disabled_cesa_diagnostics(edge_count):
    return {'enabled': False, 'valid_aerial_lineages': 0,
            'valid_ground_lineages': 0,
            'aerial_lineage_survival_rate': 0.0,
            'ground_lineage_survival_rate': 0.0,
            'current_pgm_edge_count': int(edge_count),
            'continued_edge_count': 0, 'new_edge_count': int(edge_count),
            'partner_switch_count': 0, 'partner_switch_rate': 0.0,
            'mean_historical_persistence': 0.0, 'max_persistence': 0.0,
            'mean_applied_history_boost': 0.0,
            'number_of_boosted_pgm_cells': 0}
