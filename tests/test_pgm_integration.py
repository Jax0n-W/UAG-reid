"""Run the actual AG-ReID Stage 2 PGM block and total-mapping helper."""

from pathlib import Path
import textwrap
import unittest

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from clustercontrast.methods.cesa import (
    CESAState, disabled_cesa_diagnostics, format_cesa_epoch,
)
from clustercontrast.methods.pgm import build_total_pgm_mapping


ROOT = Path(__file__).resolve().parents[1]


def original_pgm_oracle(rgb, ir):
    """Frozen original matching steps including unmatched-RGB completion."""
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


def deterministic_cost(num_ground, num_aerial):
    ground = torch.arange(num_ground, dtype=torch.float32).unsqueeze(1)
    aerial = torch.arange(num_aerial, dtype=torch.float32).unsqueeze(0)
    return (ground - torch.remainder(aerial, num_ground)).abs() + aerial * 1e-6


class RecordingCESAState(CESAState):
    def __post_init__(self):
        super().__post_init__()
        self.received_edges = None

    def complete(self, prepared, matched_edges, current_aerial, current_ground):
        self.received_edges = list(matched_edges)
        return super().complete(
            prepared, matched_edges, current_aerial, current_ground)


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
    namespace = {
        'torch': torch, 'F': F,
        'linear_sum_assignment': linear_sum_assignment,
        'build_total_pgm_mapping': build_total_pgm_mapping,
        'format_cesa_epoch': format_cesa_epoch,
        'disabled_cesa_diagnostics': disabled_cesa_diagnostics,
        'num_cluster_rgb': num_cluster_rgb,
        'num_cluster_ir': num_cluster_ir,
        'cluster_features_rgb': rgb,
        'cluster_features_ir': ir,
        'pseudo_labels_rgb': pseudo_labels_rgb,
        'pseudo_labels_ir': pseudo_labels_ir,
        'cesa_state': cesa_state,
    }
    exec(code, namespace)
    result = (namespace['R'], namespace['r2i'], namespace['i2r'])
    if include_diagnostics:
        return result + (namespace.get('cesa_diag'),)
    return result


class PGMIntegrationTests(unittest.TestCase):
    def test_ground_smaller_mapping_is_total_and_trainer_safe(self):
        edges, r2i, i2r, diagnostics = build_total_pgm_mapping(
            deterministic_cost(2, 3))
        self.assertEqual(len(r2i), 2)
        self.assertEqual(len(i2r), 3)
        self.assertEqual(set(r2i), {0, 1})
        self.assertEqual(set(i2r), {0, 1, 2})
        self.assertEqual(diagnostics['core_matches'], 2)
        self.assertEqual(diagnostics['ground_completion'], 0)
        self.assertEqual(diagnostics['aerial_completion'], 1)
        self.assertEqual(len(edges), 3)
        labels_rgb = [0, 1]
        labels_ir = [0, 1, 2]
        self.assertEqual(len([r2i[label] for label in labels_rgb]), 2)
        self.assertEqual(len([i2r[label] for label in labels_ir]), 3)

    def test_realistic_ground_smaller_ratio_is_total(self):
        edges, r2i, i2r, diagnostics = build_total_pgm_mapping(
            deterministic_cost(35, 236))
        self.assertEqual(diagnostics['core_matches'], 35)
        self.assertEqual(diagnostics['ground_completion'], 0)
        self.assertEqual(diagnostics['aerial_completion'], 201)
        self.assertEqual(len(edges), 236)
        self.assertEqual(set(r2i), set(range(35)))
        self.assertEqual(set(i2r), set(range(236)))
        self.assertTrue(all(0 <= value < 236 for value in r2i.values()))
        self.assertTrue(all(0 <= value < 35 for value in i2r.values()))

    def test_original_ground_larger_fixture_is_unchanged(self):
        rgb = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
        ir = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        expected = original_pgm_oracle(rgb, ir)
        for script in ('train_agreid.py', 'train_lag.py'):
            with self.subTest(script=script):
                self.assertEqual(run_actual_stage2_pgm(script), expected)

    def test_equal_cluster_counts_need_no_completion(self):
        edges, r2i, i2r, diagnostics = build_total_pgm_mapping(
            deterministic_cost(4, 4))
        self.assertEqual(len(edges), 4)
        self.assertEqual(len(r2i), 4)
        self.assertEqual(len(i2r), 4)
        self.assertEqual(diagnostics['core_matches'], 4)
        self.assertEqual(diagnostics['ground_completion'], 0)
        self.assertEqual(diagnostics['aerial_completion'], 0)

    def test_extreme_imbalances_are_total(self):
        for num_ground, num_aerial in ((1, 10), (10, 1)):
            with self.subTest(shape=(num_ground, num_aerial)):
                edges, r2i, i2r, diagnostics = build_total_pgm_mapping(
                    deterministic_cost(num_ground, num_aerial))
                self.assertEqual(set(r2i), set(range(num_ground)))
                self.assertEqual(set(i2r), set(range(num_aerial)))
                self.assertEqual(len(edges), max(num_ground, num_aerial))
                self.assertEqual(
                    diagnostics['total_edges'], max(num_ground, num_aerial))

    def test_cesa_epoch_zero_matches_baseline_in_both_orientations(self):
        fixtures = [
            (2, 3, [0, 0, 1, 1], [0, 0, 1, 1, 2, 2]),
            (3, 2, [0, 0, 1, 1, 2, 2], [0, 0, 1, 1]),
        ]
        for num_rgb, num_ir, ground, aerial in fixtures:
            with self.subTest(shape=(num_rgb, num_ir)):
                baseline = run_actual_stage2_pgm(
                    'train_agreid.py', num_cluster_rgb=num_rgb,
                    num_cluster_ir=num_ir, pseudo_labels_rgb=ground,
                    pseudo_labels_ir=aerial)
                with_cesa = run_actual_stage2_pgm(
                    'train_agreid.py', CESAState(),
                    num_cluster_rgb=num_rgb, num_cluster_ir=num_ir,
                    pseudo_labels_rgb=ground, pseudo_labels_ir=aerial)
                self.assertEqual(with_cesa, baseline)

    def test_cesa_complete_receives_final_core_plus_completion_edges(self):
        aerial = [0, 0, 1, 1, 2, 2]
        ground = [0, 0, 1, 1]
        state = RecordingCESAState()
        edges, r2i, i2r, diagnostics = run_actual_stage2_pgm(
            'train_agreid.py', state, num_cluster_rgb=2,
            num_cluster_ir=3, pseudo_labels_rgb=ground,
            pseudo_labels_ir=aerial, include_diagnostics=True)
        self.assertEqual(len(edges), 3)
        self.assertEqual(len(r2i), 2)
        self.assertEqual(len(i2r), 3)
        self.assertEqual(state.received_edges,
                         [(aerial_idx, ground_idx)
                          for ground_idx, aerial_idx in edges])
        self.assertTrue(diagnostics['pgm_executed'])
        self.assertEqual(diagnostics['current_pgm_edge_count'], 3)

    def test_raw_cosine_cesa_exp_hungarian_completion_order_is_unchanged(self):
        source = (ROOT / 'train_agreid.py').read_text(encoding='utf-8')
        positions = [
            source.index('raw_cosine ='),
            source.index('cesa_state.prepare', source.index('raw_cosine =')),
            source.index('calibrated_score.exp()', source.index('raw_cosine =')),
            source.index('build_total_pgm_mapping(cost)',
                         source.index('raw_cosine =')),
            source.index('cesa_state.complete', source.index('raw_cosine =')),
        ]
        self.assertEqual(positions, sorted(positions))

    def test_lag_cluster_count_inversion_is_unchanged(self):
        result = run_actual_stage2_pgm(
            'train_lag.py', CESAState(), num_cluster_rgb=2,
            num_cluster_ir=3, pseudo_labels_rgb=[0, 0, 1, 1],
            pseudo_labels_ir=[0, 0, 1, 1, 2, 2],
            include_diagnostics=True)
        self.assertEqual(result[:3], ([], {}, {}))
        self.assertFalse(result[3]['pgm_executed'])


if __name__ == '__main__':
    unittest.main()
