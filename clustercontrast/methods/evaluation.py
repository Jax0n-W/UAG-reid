"""Shared AG-ReID evaluation and strict AGW checkpoint loading."""

import os.path as osp

import numpy as np
import torch
import torch.nn.functional as F


AGREID_MODALITIES = {'ground': 1, 'aerial': 2}
AGREID_DIRECTIONS = {
    'a2g': ('aerial', 'ground'),
    'g2a': ('ground', 'aerial'),
}


def agreid_direction_modalities(direction):
    try:
        query_name, gallery_name = AGREID_DIRECTIONS[direction]
    except KeyError as error:
        raise ValueError('Unknown AG-ReID direction: {}'.format(direction)) from error
    return (query_name, AGREID_MODALITIES[query_name],
            gallery_name, AGREID_MODALITIES[gallery_name])


def load_agreid_split(data_dir, trial, modality):
    if modality not in AGREID_MODALITIES:
        raise ValueError('Unknown AG-ReID modality: {}'.format(modality))
    index_path = osp.join(
        data_dir, 'idx', 'test_{}_{}.txt'.format(modality, trial))
    if not osp.isfile(index_path):
        raise FileNotFoundError(
            'AG-ReID evaluation index not found: {}'.format(index_path))
    with open(index_path, 'r') as index_file:
        rows = [row for row in index_file.read().splitlines() if row.strip()]
    if not rows:
        raise RuntimeError(
            'AG-ReID evaluation split is empty: {}'.format(index_path))
    paths = [osp.join(data_dir, row.split()[0]) for row in rows]
    labels = np.asarray([int(row.split()[1]) for row in rows])
    return paths, labels


def extract_agreid_features(model, loader, num_samples, modal):
    """Extract flip-averaged, L2-normalized AG-ReID features."""
    if modal not in AGREID_MODALITIES.values():
        raise ValueError('AG-ReID modal must be 1 (ground) or 2 (aerial)')
    model.eval()
    features = []
    with torch.no_grad():
        for inputs, _labels in loader:
            flipped = torch.flip(inputs, dims=[3])
            inputs = inputs.cuda()
            flipped = flipped.cuda()
            original_features = model(inputs, inputs, modal)
            flipped_features = model(flipped, flipped, modal)
            averaged = (original_features.detach()
                        + flipped_features.detach()) / 2
            features.append(F.normalize(averaged, p=2, dim=1).cpu().numpy())
    if not features:
        raise RuntimeError('AG-ReID evaluation loader produced no batches')
    result = np.concatenate(features, axis=0)
    if result.shape[0] != int(num_samples):
        raise RuntimeError(
            'AG-ReID feature count mismatch: expected {}, got {}'.format(
                num_samples, result.shape[0]))
    return result


def evaluate_agreid_distances(distmat, q_pids, g_pids, max_rank=20,
                              q_camids=None, g_camids=None):
    """Canonical CMC, mAP, and mINP implementation for AG-ReID."""
    distmat = np.asarray(distmat)
    q_pids = np.asarray(q_pids)
    g_pids = np.asarray(g_pids)
    if distmat.ndim != 2 or distmat.shape != (len(q_pids), len(g_pids)):
        raise ValueError('AG-ReID distance matrix shape does not match labels')
    num_q, num_g = distmat.shape
    if num_q == 0 or num_g == 0:
        raise RuntimeError('AG-ReID query and gallery must both be non-empty')
    if not np.all(np.isfinite(distmat)):
        raise RuntimeError('AG-ReID distance matrix contains non-finite values')
    max_rank = min(int(max_rank), num_g)
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    if q_camids is None:
        q_camids = np.ones(num_q, dtype=np.int32)
    else:
        q_camids = np.asarray(q_camids)
    if g_camids is None:
        g_camids = np.full(num_g, 2, dtype=np.int32)
    else:
        g_camids = np.asarray(g_camids)

    all_cmc = []
    all_ap = []
    all_inp = []
    for query_index in range(num_q):
        order = indices[query_index]
        remove = ((g_pids[order] == q_pids[query_index])
                  & (g_camids[order] == q_camids[query_index]))
        raw_cmc = matches[query_index][~remove]
        if not np.any(raw_cmc):
            continue
        cumulative = raw_cmc.cumsum()
        last_relevant = np.flatnonzero(raw_cmc)[-1]
        all_inp.append(cumulative[last_relevant] / (last_relevant + 1.0))
        cumulative[cumulative > 1] = 1
        cmc = cumulative[:max_rank]
        if len(cmc) < max_rank:
            cmc = np.pad(cmc, (0, max_rank - len(cmc)), mode='edge')
        all_cmc.append(cmc)
        precisions = raw_cmc.cumsum() / (np.arange(len(raw_cmc)) + 1.0)
        all_ap.append(float((precisions * raw_cmc).sum() / raw_cmc.sum()))

    if not all_cmc:
        raise RuntimeError(
            'AG-ReID evaluation has no valid query identity in the gallery')
    cmc = np.asarray(all_cmc, dtype=np.float32).mean(axis=0)
    m_ap = float(np.mean(all_ap))
    m_inp = float(np.mean(all_inp))
    values = np.concatenate([cmc, np.asarray([m_ap, m_inp])])
    if not np.all(np.isfinite(values)):
        raise RuntimeError('AG-ReID evaluation produced non-finite metrics')
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise RuntimeError('AG-ReID evaluation metrics must be in [0, 1]')
    return cmc, m_ap, m_inp


def _rank_at(cmc, rank):
    return float(cmc[min(rank, len(cmc)) - 1])


def evaluate_agreid_feature_pair(query_features, query_labels,
                                 gallery_features, gallery_labels):
    similarity = np.matmul(query_features, gallery_features.T)
    cmc, m_ap, m_inp = evaluate_agreid_distances(
        -similarity, query_labels, gallery_labels)
    metrics = {
        'rank1': _rank_at(cmc, 1),
        'rank5': _rank_at(cmc, 5),
        'rank10': _rank_at(cmc, 10),
        'rank20': _rank_at(cmc, 20),
        'mAP': m_ap,
        'mINP': m_inp,
    }
    if not all(np.isfinite(value) and 0.0 <= value <= 1.0
               for value in metrics.values()):
        raise RuntimeError('AG-ReID evaluation produced invalid metrics')
    return metrics


def evaluate_agreid_direction(model, direction, query_loader, query_labels,
                              gallery_loader, gallery_labels):
    query_name, query_modal, gallery_name, gallery_modal = \
        agreid_direction_modalities(direction)
    del query_name, gallery_name
    query_features = extract_agreid_features(
        model, query_loader, len(query_labels), query_modal)
    gallery_features = extract_agreid_features(
        model, gallery_loader, len(gallery_labels), gallery_modal)
    return evaluate_agreid_feature_pair(
        query_features, query_labels, gallery_features, gallery_labels)


def evaluate_agreid_bidirectional(model, aerial_loader, aerial_labels,
                                  ground_loader, ground_labels):
    """Evaluate both directions after extracting each modality once."""
    aerial_features = extract_agreid_features(
        model, aerial_loader, len(aerial_labels), modal=2)
    ground_features = extract_agreid_features(
        model, ground_loader, len(ground_labels), modal=1)
    a2g = evaluate_agreid_feature_pair(
        aerial_features, aerial_labels, ground_features, ground_labels)
    g2a = evaluate_agreid_feature_pair(
        ground_features, ground_labels, aerial_features, aerial_labels)
    return a2g, g2a


def load_agw_checkpoint_strict(model, checkpoint):
    state_dict = checkpoint.get('state_dict') if isinstance(checkpoint, dict) else None
    if not isinstance(state_dict, dict):
        raise RuntimeError('Checkpoint architecture does not match AGW evaluation model: missing state_dict.')
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as error:
        raise RuntimeError(
            'Checkpoint architecture does not match AGW evaluation model.') from error
    return model
