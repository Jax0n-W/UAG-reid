"""Reliability-aware hard prototype selection with raw-cosine reliability."""

import math
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


def _faiss_knn(vectors, k, query_chunk, use_gpu):
    import faiss

    cpu_index = faiss.IndexFlatIP(vectors.shape[1])
    if use_gpu:
        resources = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(resources, torch.cuda.current_device(), cpu_index)
    else:
        index = cpu_index
    index.add(vectors)
    result = torch.empty((vectors.shape[0], k), dtype=torch.long)
    for start in range(0, vectors.shape[0], query_chunk):
        stop = min(start + query_chunk, vectors.shape[0])
        _, neighbors = index.search(vectors[start:stop], k + 1)
        for offset, row in enumerate(neighbors):
            own = start + offset
            others = row[row != own][:k]
            if len(others) != k:
                raise RuntimeError("KNN search returned too few non-self neighbors")
            result[own] = torch.from_numpy(others.copy())
    return result


def _cosine_knn_indices(features, k, query_chunk=256, reference_chunk=4096,
                         backend='auto', return_backend=False):
    """Exact inner-product KNN without constructing an N x N matrix."""
    count = features.size(0)
    if k == 0:
        result = torch.empty((count, 0), dtype=torch.long)
        return (result, 'none') if return_backend else result
    if backend not in ('auto', 'gpu_faiss', 'cpu_faiss', 'torch_chunk'):
        raise ValueError("backend must be auto, gpu_faiss, cpu_faiss, or torch_chunk")
    try:
        import faiss
    except ImportError:
        faiss = None
    vectors_np = np.ascontiguousarray(
        features.detach().cpu().numpy(), dtype=np.float32)
    gpu_available = (faiss is not None and torch.cuda.is_available()
                     and hasattr(faiss, 'StandardGpuResources')
                     and hasattr(faiss, 'index_cpu_to_gpu'))
    if backend in ('auto', 'gpu_faiss') and gpu_available:
        try:
            result = _faiss_knn(vectors_np, k, query_chunk, use_gpu=True)
            return (result, 'gpu_faiss') if return_backend else result
        except Exception:
            if backend == 'gpu_faiss':
                raise
    elif backend == 'gpu_faiss':
        raise RuntimeError('GPU FAISS exact KNN is unavailable')
    if backend in ('auto', 'cpu_faiss') and faiss is not None:
        result = _faiss_knn(vectors_np, k, query_chunk, use_gpu=False)
        return (result, 'cpu_faiss') if return_backend else result
    if backend == 'cpu_faiss':
        raise RuntimeError('CPU FAISS exact KNN is unavailable')

    result = torch.empty((count, k), dtype=torch.long)
    vectors = features.cpu()
    for start in range(0, count, query_chunk):
        stop = min(start + query_chunk, count)
        query = vectors[start:stop]
        best_scores = torch.full((stop - start, k), -float('inf'))
        best_indices = torch.full((stop - start, k), -1, dtype=torch.long)
        for ref_start in range(0, count, reference_chunk):
            ref_stop = min(ref_start + reference_chunk, count)
            scores = query.mm(vectors[ref_start:ref_stop].t())
            overlaps = torch.arange(start, stop)
            in_block = (overlaps >= ref_start) & (overlaps < ref_stop)
            rows = torch.nonzero(in_block, as_tuple=False).flatten()
            if rows.numel():
                scores[rows, overlaps[rows] - ref_start] = -float('inf')
            indices = torch.arange(ref_start, ref_stop).expand(stop - start, -1)
            merged_scores = torch.cat((best_scores, scores), dim=1)
            merged_indices = torch.cat((best_indices, indices), dim=1)
            best_scores, positions = merged_scores.topk(k, dim=1)
            best_indices = merged_indices.gather(1, positions)
        result[start:stop] = best_indices
    return (result, 'torch_chunk') if return_backend else result


def compute_rahp_reliability(features, pseudo_labels, knn=20, alpha=0.5,
                             eps=1e-6, query_chunk=256, reference_chunk=4096,
                             return_diagnostics=False, knn_backend='auto'):
    """Return q in full extracted-feature order; outlier entries are zero.

    Each invocation is one independent domain/label space. Only pseudo labels
    enter this computation; dataset identity and ground-truth IDs never do.
    """
    if knn < 1 or not 0.0 <= alpha <= 1.0 or eps <= 0:
        raise ValueError("knn >= 1, alpha in [0, 1], and eps > 0 are required")
    vectors = torch.as_tensor(features, dtype=torch.float32).detach().cpu()
    labels = torch.as_tensor(pseudo_labels, dtype=torch.long).flatten().cpu()
    if vectors.ndim != 2 or vectors.size(0) != labels.numel():
        raise ValueError("features and pseudo_labels must have aligned rows")
    q_full = torch.zeros(labels.numel(), dtype=torch.float32)
    valid = labels != -1
    valid_count = int(valid.sum())
    diagnostics = {'enabled': True, 'num_valid_samples': valid_count,
                   'num_clusters': 0, 'mean_q_nbr': 0.0,
                   'mean_q_margin': 0.0, 'mean_q': 0.0,
                   'knn_backend': 'none', 'knn_k': 0, 'knn_time': 0.0}
    if not valid_count:
        return (q_full, diagnostics) if return_diagnostics else q_full

    vectors = F.normalize(vectors[valid], dim=1)
    valid_labels = labels[valid]
    cluster_labels, inverse = torch.unique(valid_labels, sorted=True,
                                           return_inverse=True)
    cluster_count = cluster_labels.numel()
    diagnostics['num_clusters'] = int(cluster_count)
    neighbor_count = min(knn, valid_count - 1)
    if neighbor_count:
        started = time.perf_counter()
        neighbor_indices, selected_backend = _cosine_knn_indices(
            vectors, neighbor_count, query_chunk, reference_chunk,
            backend=knn_backend, return_backend=True)
        diagnostics.update(knn_backend=selected_backend,
                           knn_k=neighbor_count,
                           knn_time=time.perf_counter() - started)
        q_nbr = (valid_labels[neighbor_indices] == valid_labels[:, None]).float().mean(1)
    else:
        q_nbr = torch.ones(valid_count)

    if cluster_count == 1:
        q_margin = torch.ones(valid_count)
    else:
        centers = torch.zeros((cluster_count, vectors.size(1)), dtype=vectors.dtype)
        centers.index_add_(0, inverse, vectors)
        centers = F.normalize(centers, dim=1)
        margins = torch.empty(valid_count)
        for start in range(0, valid_count, query_chunk):
            stop = min(start + query_chunk, valid_count)
            scores = vectors[start:stop].mm(centers.t())
            rows = torch.arange(stop - start)
            own = scores[rows, inverse[start:stop]].clone()
            scores[rows, inverse[start:stop]] = -float('inf')
            margins[start:stop] = own - scores.max(dim=1).values
        median = margins.median()
        mad = (margins - median).abs().median()
        standardized = ((margins - median) / (mad + eps)).clamp(-5, 5)
        q_margin = torch.sigmoid(standardized)

    q_valid = (q_nbr + eps).pow(alpha) * (q_margin + eps).pow(1 - alpha)
    q_full[valid] = q_valid
    diagnostics.update(mean_q_nbr=float(q_nbr.mean()),
                       mean_q_margin=float(q_margin.mean()),
                       mean_q=float(q_valid.mean()))
    return (q_full, diagnostics) if return_diagnostics else q_full


def gather_batch_reliability(q_filtered, indexes, duplicate_views=False, device=None):
    """Gather by the filtered Preprocessor index, then mirror dual views."""
    table = torch.as_tensor(q_filtered, dtype=torch.float32)
    index = torch.as_tensor(indexes, dtype=torch.long, device='cpu').flatten()
    batch = table[index]
    if duplicate_views:
        batch = torch.cat((batch, batch), dim=0)
    return batch.to(device) if device is not None else batch


def _cpu_vector(values):
    if isinstance(values, (list, tuple)) and values and isinstance(values[0], torch.Tensor):
        return torch.stack([value.detach().cpu() for value in values]).flatten()
    return torch.as_tensor(values).detach().cpu().flatten()


def select_rahp_hard(hardness, reliability, beta, batch_positions=None):
    """Choose maximum reliability among the hardest ceil(beta * B) items."""
    if not 0.0 < beta <= 1.0:
        raise ValueError("rahp_beta must be in (0, 1]")
    h = _cpu_vector(hardness).tolist()
    q = _cpu_vector(reliability).tolist()
    if not h or len(h) != len(q):
        raise ValueError("hardness and reliability must have equal nonzero length")
    positions = list(range(len(h))) if batch_positions is None else list(batch_positions)
    if len(positions) != len(h):
        raise ValueError("batch_positions must align with hardness")
    candidate_count = max(1, math.ceil(beta * len(h)))
    candidates = sorted(range(len(h)), key=lambda i: (-h[i], positions[i]))[:candidate_count]
    selected = min(candidates, key=lambda i: (-q[i], -h[i], positions[i]))
    percentile = sum(value <= h[selected] for value in h) / len(h)
    return selected, candidate_count, percentile


@dataclass
class RAHPSelectionStats:
    updates: int = 0
    replacements: int = 0
    candidate_count_sum: int = 0
    baseline_reliability_sum: float = 0.0
    selected_reliability_sum: float = 0.0
    selected_percentile_sum: float = 0.0

    def observe(self, baseline_index, selected_index, reliability,
                candidate_count, selected_percentile):
        q = _cpu_vector(reliability)
        self.updates += 1
        self.replacements += int(baseline_index != selected_index)
        self.candidate_count_sum += candidate_count
        self.baseline_reliability_sum += float(q[baseline_index])
        self.selected_reliability_sum += float(q[selected_index])
        self.selected_percentile_sum += selected_percentile

    def summary(self):
        count = max(1, self.updates)
        return {'mean_candidate_count': self.candidate_count_sum / count,
                'replacement_rate': self.replacements / count,
                'baseline_hard_mean_reliability': self.baseline_reliability_sum / count,
                'rahp_hard_mean_reliability': self.selected_reliability_sum / count,
                'selected_hard_percentile': self.selected_percentile_sum / count}


def format_rahp_epoch(reliability_diagnostics, selection_stats, enabled=True):
    """One compact epoch-level diagnostic line across independent spaces."""
    records = reliability_diagnostics or []
    total = sum(item['num_valid_samples'] for item in records)
    clusters = sum(item['num_clusters'] for item in records)
    def weighted(key):
        return (sum(item[key] * item['num_valid_samples'] for item in records) / total
                if total else 0.0)
    selected = selection_stats.summary() if selection_stats else RAHPSelectionStats().summary()
    summary = ('[RAHP] enabled={} valid={} clusters={} q_nbr={:.4f} q_margin={:.4f} '
            'q={:.4f} candidates={:.2f} replacement={:.4f} baseline_q={:.4f} '
            'selected_q={:.4f} hard_percentile={:.4f}').format(
                bool(enabled), total, clusters, weighted('mean_q_nbr'),
                weighted('mean_q_margin'), weighted('mean_q'),
                selected['mean_candidate_count'], selected['replacement_rate'],
                selected['baseline_hard_mean_reliability'],
                selected['rahp_hard_mean_reliability'],
                selected['selected_hard_percentile'])
    knn_lines = [
        '[RAHP-KNN] backend={} N={} K={} time={:.6f}'.format(
            item.get('knn_backend', 'none'), item['num_valid_samples'],
            item.get('knn_k', 0), item.get('knn_time', 0.0))
        for item in records
    ]
    return summary + (('\n' + '\n'.join(knn_lines)) if knn_lines else '')


def disabled_rahp_diagnostics(pseudo_labels):
    labels = torch.as_tensor(pseudo_labels, dtype=torch.long).flatten()
    valid = labels[labels != -1]
    return {'enabled': False, 'num_valid_samples': int(valid.numel()),
            'num_clusters': int(torch.unique(valid).numel()),
            'mean_q_nbr': 0.0, 'mean_q_margin': 0.0, 'mean_q': 0.0,
            'knn_backend': 'disabled', 'knn_k': 0, 'knn_time': 0.0}
