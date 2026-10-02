from __future__ import absolute_import

import torch
from scipy.optimize import linear_sum_assignment


def _validate_total_mapping(edges, r2i, i2r, num_ground, num_aerial):
    expected_ground = set(range(num_ground))
    expected_aerial = set(range(num_aerial))
    assert set(r2i.keys()) == expected_ground, (
        'PGM r2i coverage failure: expected {}, got {}'.format(
            expected_ground, set(r2i.keys())))
    assert set(i2r.keys()) == expected_aerial, (
        'PGM i2r coverage failure: expected {}, got {}'.format(
            expected_aerial, set(i2r.keys())))
    assert all(0 <= aerial < num_aerial for aerial in r2i.values()), (
        'PGM r2i contains an invalid aerial index')
    assert all(0 <= ground < num_ground for ground in i2r.values()), (
        'PGM i2r contains an invalid ground index')
    assert all(0 <= ground < num_ground and 0 <= aerial < num_aerial
               for ground, aerial in edges), (
        'PGM final edge set contains an invalid cluster index')


def build_total_pgm_mapping(cost):
    """Build trainer-safe total mappings around a rectangular Hungarian core.

    Args:
        cost: finite 2-D tensor shaped ``[num_ground, num_aerial]``.

    Returns:
        ``(R, r2i, i2r, diagnostics)`` where every ground key exists in r2i,
        every aerial key exists in i2r, and every R tuple is
        ``(ground_cluster, aerial_cluster)``.
    """
    if not torch.is_tensor(cost) or cost.ndim != 2:
        raise TypeError('PGM cost must be a 2-D torch tensor')
    num_ground, num_aerial = map(int, cost.shape)
    if num_ground <= 0 or num_aerial <= 0:
        raise ValueError('PGM cost must have at least one cluster per modality')

    working_cost = cost.detach().cpu()
    if not bool(torch.isfinite(working_cost).all()):
        raise ValueError('PGM cost contains non-finite values')

    row_ind, col_ind = linear_sum_assignment(working_cost.numpy())
    edges = []
    r2i = {}
    i2r = {}
    for row, column in zip(row_ind, col_ind):
        ground = int(row)
        aerial = int(column)
        edges.append((ground, aerial))
        r2i[ground] = aerial
        i2r[aerial] = ground
    core_matches = len(edges)

    # Preserve the original PCLHD unmatched-ground Hungarian completion.
    unmatched_ground = sorted(set(range(num_ground)) - set(r2i.keys()))
    ground_completion = 0
    if unmatched_ground:
        extra_rows, extra_columns = linear_sum_assignment(
            working_cost[unmatched_ground].numpy())
        for local_row, column in zip(extra_rows, extra_columns):
            ground = int(unmatched_ground[int(local_row)])
            aerial = int(column)
            edges.append((ground, aerial))
            r2i[ground] = aerial
            ground_completion += 1

    # Extreme G >> A can leave rows after the original second assignment.
    residual_ground = sorted(set(range(num_ground)) - set(r2i.keys()))
    for ground in residual_ground:
        aerial = int(torch.argmin(working_cost[ground, :]).item())
        edges.append((ground, aerial))
        r2i[ground] = aerial
        ground_completion += 1

    # Symmetric total completion for the G < A case.  Do not overwrite r2i.
    unmatched_aerial = sorted(set(range(num_aerial)) - set(i2r.keys()))
    aerial_completion = 0
    for aerial in unmatched_aerial:
        ground = int(torch.argmin(working_cost[:, aerial]).item())
        edges.append((ground, aerial))
        i2r[aerial] = ground
        aerial_completion += 1

    _validate_total_mapping(
        edges, r2i, i2r, num_ground=num_ground, num_aerial=num_aerial)
    diagnostics = {
        'ground_clusters': num_ground,
        'aerial_clusters': num_aerial,
        'core_matches': core_matches,
        'ground_completion': ground_completion,
        'aerial_completion': aerial_completion,
        'total_edges': len(edges),
        'r2i_coverage': len(r2i),
        'i2r_coverage': len(i2r),
    }
    return edges, r2i, i2r, diagnostics
