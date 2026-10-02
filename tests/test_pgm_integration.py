"""Run the actual Stage 2 PGM blocks on a synthetic matching fixture."""

from pathlib import Path
import textwrap
import unittest

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from clustercontrast.methods.cesa import CESAState


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


def run_actual_stage2_pgm(script, cesa_state=None):
    source = (ROOT / script).read_text(encoding='utf-8')
    start = source.index('        ######################## PGM')
    end = source.index('        print("Finish Bipartite Graph Matching")', start)
    code = textwrap.dedent(source[start:end])
    rgb = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    ir = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    namespace = {'torch': torch, 'F': F,
                 'linear_sum_assignment': linear_sum_assignment,
                 'num_cluster_rgb': 3, 'num_cluster_ir': 2,
                 'cluster_features_rgb': rgb, 'cluster_features_ir': ir,
                 'pseudo_labels_rgb': [0, 0, 1, 1, 2, 2],
                 'pseudo_labels_ir': [0, 0, 1, 1],
                 'cesa_state': cesa_state}
    exec(code, namespace)
    return namespace['R'], namespace['r2i'], namespace['i2r']


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


if __name__ == '__main__':
    unittest.main()
