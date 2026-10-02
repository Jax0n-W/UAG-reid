"""Run the actual Stage 2 PGM blocks on a synthetic matching fixture."""

from pathlib import Path
import textwrap
import unittest

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from clustercontrast.methods.cesa import (
    CESAState, disabled_cesa_diagnostics, format_cesa_epoch,
)


ROOT = Path(__file__).resolve().parents[1]


def original_pgm_oracle(rgb, ir):
    """Frozen original matching steps including its unmatched completion."""
    rgb = F.normalize(rgb, dim=1)
    ir = F.normalize(ir, dim=1)
    similarity = (torch.mm(rgb, ir.T) / 1).exp().cpu()
    reciprocal = 1 / similarity
    cost = torch.cat((reciprocal,
                      torch.zeros(reciprocal.shape[0],
                                  reciprocal.shape[0] - reciprocal.shape[1])), 1)
    i2r, r2i, edges, unmatched = {}, {}, [], []
    rows, columns = linear_sum_assignment(cost)
    for idx, row in enumerate(rows):
        if columns[idx] < similarity.shape[1]:
            edges.append((row, columns[idx]))
            r2i[row] = columns[idx]
            i2r[columns[idx]] = row
        else:
            unmatched.append(row)
    unmatched_cost = cost[unmatched][:, :reciprocal.shape[1]]
    extra_rows, extra_columns = linear_sum_assignment(unmatched_cost)
    for idx, row in enumerate(extra_rows):
        edges.append((unmatched[row], extra_columns[idx]))
        r2i[unmatched[row]] = extra_columns[idx]
    return edges, r2i, i2r


def rectangular_pgm_oracle(rgb, ir):
    """Rectangular assignment with the frozen unmatched-RGB completion."""
    rgb = F.normalize(rgb, dim=1)
    ir = F.normalize(ir, dim=1)
    cost = 1 / (torch.mm(rgb, ir.T) / 1).exp().cpu()
    rows, columns = linear_sum_assignment(cost)
    edges = [(int(row), int(column))
             for row, column in zip(rows, columns)]
    r2i = {row: column for row, column in edges}
    i2r = {column: row for row, column in edges}
    matched_rows = set(r2i)
    unmatched = sorted(set(range(rgb.shape[0])) - matched_rows)
    if unmatched:
        extra_rows, extra_columns = linear_sum_assignment(cost[unmatched])
        for local_row, column in zip(extra_rows, extra_columns):
            row = unmatched[int(local_row)]
            column = int(column)
            edges.append((row, column))
            r2i[row] = column
    return edges, r2i, i2r


def run_actual_stage2_pgm(script, cesa_state=None, num_cluster_rgb=3,
                          num_cluster_ir=2, pseudo_labels_rgb=None,
                          pseudo_labels_ir=None, include_diagnostics=False):
    source = (ROOT / script).read_text(encoding='utf-8')
    start = source.index('        ######################## PGM')
    end = source.index('        ####################################', start)
    code = textwrap.dedent(source[start:end])
    basis = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0],
                          [0.0, -1.0]])
    rgb = basis[:num_cluster_rgb].clone()
    ir = basis[:num_cluster_ir].clone()
    if pseudo_labels_rgb is None:
        pseudo_labels_rgb = [label for label in range(num_cluster_rgb)
                             for _ in range(2)]
    if pseudo_labels_ir is None:
        pseudo_labels_ir = [label for label in range(num_cluster_ir)
                            for _ in range(2)]
    namespace = {'torch': torch, 'F': F,
                 'linear_sum_assignment': linear_sum_assignment,
                 'format_cesa_epoch': format_cesa_epoch,
                 'disabled_cesa_diagnostics': disabled_cesa_diagnostics,
                 'num_cluster_rgb': num_cluster_rgb,
                 'num_cluster_ir': num_cluster_ir,
                 'cluster_features_rgb': rgb, 'cluster_features_ir': ir,
                 'pseudo_labels_rgb': pseudo_labels_rgb,
                 'pseudo_labels_ir': pseudo_labels_ir,
                 'cesa_state': cesa_state}
    exec(code, namespace)
    result = (namespace['R'], namespace['r2i'], namespace['i2r'])
    if include_diagnostics:
        return result + (namespace.get('cesa_diag'),)
    return result


class PGMIntegrationTests(unittest.TestCase):
    def test_cesa_off_matches_frozen_original_pgm(self):
        rgb = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
        ir = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        expected_edges, expected_r2i, expected_i2r = original_pgm_oracle(rgb, ir)
        for script in ('train_agreid.py', 'train_lag.py'):
            with self.subTest(script=script):
                edges, r2i, i2r = run_actual_stage2_pgm(script)
                self.assertEqual(edges, expected_edges)
                self.assertEqual(r2i, expected_r2i)
                self.assertEqual(i2r, expected_i2r)

    def test_cesa_epoch_zero_matches_baseline_pgm(self):
        for script in ('train_agreid.py', 'train_lag.py'):
            with self.subTest(script=script):
                baseline = run_actual_stage2_pgm(script)
                with_cesa = run_actual_stage2_pgm(script, CESAState())
                self.assertEqual(with_cesa, baseline)

    def test_agreid_executes_rectangular_pgm_when_rgb_is_smaller(self):
        aerial = [0, 0, 1, 1, 2, 2]
        ground = [0, 0, 1, 1]
        rgb = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        ir = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
        expected = rectangular_pgm_oracle(rgb, ir)
        baseline = run_actual_stage2_pgm(
            'train_agreid.py', num_cluster_rgb=2, num_cluster_ir=3,
            pseudo_labels_rgb=ground, pseudo_labels_ir=aerial)
        self.assertEqual(baseline, expected)
        self.assertEqual(len(baseline[0]), 2)
        self.assertEqual(set(baseline[1]), {0, 1})

        state = CESAState()
        result = run_actual_stage2_pgm(
            'train_agreid.py', state, num_cluster_rgb=2, num_cluster_ir=3,
            pseudo_labels_rgb=ground, pseudo_labels_ir=aerial,
            include_diagnostics=True)
        self.assertEqual(result[:3], expected)
        self.assertTrue(result[3]['enabled'])
        self.assertTrue(result[3]['pgm_executed'])
        self.assertEqual(result[3]['current_pgm_edge_count'], 2)
        self.assertEqual(state.stage2_epoch, 1)

    def test_lag_cluster_count_inversion_is_unchanged(self):
        aerial = [0, 0, 1, 1, 2, 2]
        ground = [0, 0, 1, 1]
        result = run_actual_stage2_pgm(
            'train_lag.py', CESAState(), num_cluster_rgb=2,
            num_cluster_ir=3, pseudo_labels_rgb=ground,
            pseudo_labels_ir=aerial, include_diagnostics=True)
        self.assertEqual(result[:3], ([], {}, {}))
        self.assertFalse(result[3]['pgm_executed'])


if __name__ == '__main__':
    unittest.main()
