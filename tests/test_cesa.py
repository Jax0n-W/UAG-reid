import unittest
import tempfile
from pathlib import Path
import random

import numpy as np
import torch

from clustercontrast.methods.cesa import (
    CESAState, build_overlap_contingency, compute_mutual_lineage,
    update_edge_persistence,
)
from clustercontrast.methods.checkpoint import capture_rng_state, restore_rng_state


class CESATests(unittest.TestCase):
    def test_exact_continuation_and_threshold_reset(self):
        previous = [0, 0, 0, 1, 1, 1]
        current = [5, 5, 5, 6, 6, 6]
        self.assertEqual(compute_mutual_lineage(previous, current, 0.5),
                         {5: 0, 6: 1})
        self.assertEqual(compute_mutual_lineage([0, 0, 0, 0],
                                                [1, 2, 2, 2], 0.8), {})

    def test_split_inherits_only_strongest_child(self):
        previous = [0] * 6 + [1] * 4
        current = [2] * 4 + [3] * 2 + [4] * 4
        self.assertEqual(compute_mutual_lineage(previous, current, 0.3),
                         {2: 0, 4: 1})

    def test_merge_inherits_only_strongest_predecessor(self):
        previous = [0] * 4 + [1] * 2 + [2] * 4
        current = [3] * 6 + [4] * 4
        self.assertEqual(compute_mutual_lineage(previous, current, 0.3),
                         {3: 0, 4: 2})

    def test_outliers_do_not_enter_overlap(self):
        overlap, old_size, new_size = build_overlap_contingency(
            [0, 0, -1, 1], [2, -1, 2, 3])
        self.assertEqual(overlap, {(0, 2): 1, (1, 3): 1})
        self.assertEqual(old_size, {0: 2, 1: 1})
        self.assertEqual(new_size, {2: 2, 3: 1})

    def test_epoch_zero_raw_pgm_score_is_exactly_unchanged(self):
        raw = torch.tensor([[0.1, 0.5], [0.9, -0.2]])
        state = CESAState()
        calibrated, prepared = state.prepare(raw, [0, 0, 1, 1], [0, 0, 1, 1])
        self.assertIs(calibrated, raw)
        self.assertEqual(prepared.applied_boosts, {})
        self.assertTrue(torch.equal(calibrated.exp(), raw.exp()))

    def test_persistence_sequence_and_partner_switch(self):
        state = CESAState(rho=0.8)
        raw = torch.zeros((2, 2))
        aerial = [0, 0, 1, 1]
        ground = [0, 0, 1, 1]
        observed = []
        for _ in range(4):
            _, prepared = state.prepare(raw, aerial, ground)
            state.complete(prepared, [(0, 0)], aerial, ground)
            observed.append(state.edge_persistence[(0, 0)])
        np.testing.assert_allclose(observed, [0.2, 0.36, 0.488, 0.5904])
        _, prepared = state.prepare(raw, aerial, ground)
        diag = state.complete(prepared, [(0, 1)], aerial, ground)
        self.assertAlmostEqual(state.edge_persistence[(0, 1)], 0.2)
        self.assertEqual(diag['partner_switch_count'], 1)

    def test_pre_exp_boost_and_resume_continuity(self):
        state = CESAState(rho=0.8, eta=0.1, warmup=5)
        aerial = [0, 0, 1, 1]
        ground = [0, 0, 1, 1]
        raw = torch.tensor([[0.5, 0.1], [0.1, 0.5]])
        for _ in range(2):
            _, prepared = state.prepare(raw, aerial, ground)
            state.complete(prepared, [(0, 0)], aerial, ground)
        saved = state.state_dict()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'checkpoint.pt'
            torch.save({'cesa_state': saved}, path)
            saved = torch.load(path, map_location='cpu')['cesa_state']
        restored = CESAState(rho=0.8, eta=0.1, warmup=5)
        restored.load_state_dict(saved)
        calibrated, prepared = restored.prepare(raw, aerial, ground)
        expected_delta = (2 / 5) * 0.1 * 0.36
        self.assertAlmostEqual(float(calibrated[0, 0] - raw[0, 0]), expected_delta, places=6)
        self.assertAlmostEqual(float(calibrated.exp()[0, 0]),
                               float(torch.exp(raw[0, 0] + expected_delta)), places=6)
        self.assertEqual(prepared.applied_boosts.keys(), {(0, 0)})
        restored.complete(prepared, [(0, 0)], aerial, ground)
        self.assertAlmostEqual(restored.edge_persistence[(0, 0)], 0.488)

    def test_missing_checkpoint_state_resets_safely(self):
        state = CESAState()
        with self.assertWarnsRegex(UserWarning, 'no cesa_state'):
            state.load_state_dict(None)
        self.assertEqual(state.edge_persistence, {})
        self.assertEqual(state.stage2_epoch, 0)

    def test_no_matching_epoch_resets_edges_and_advances_labels(self):
        state = CESAState()
        state.prev_labels_aerial = np.array([9, 9, 8, 8])
        state.prev_labels_ground = np.array([7, 7, 6, 6])
        state.edge_persistence = {(9, 7): 0.75}
        state.stage2_epoch = 3
        current_aerial = [0, 0, 1, 1]
        current_ground = [2, 2, 3, 3]
        diagnostics = state.advance_without_matching(
            current_aerial, current_ground)
        self.assertFalse(diagnostics['pgm_executed'])
        self.assertEqual(diagnostics['current_pgm_edge_count'], 0)
        self.assertEqual(state.edge_persistence, {})
        self.assertEqual(state.prev_labels_aerial.tolist(), current_aerial)
        self.assertEqual(state.prev_labels_ground.tolist(), current_ground)
        self.assertEqual(state.stage2_epoch, 4)

    def test_rng_checkpoint_roundtrip(self):
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        snapshot = capture_rng_state()
        expected = (random.random(), float(np.random.rand()), float(torch.rand(())))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'rng.pt'
            torch.save(snapshot, path)
            snapshot = torch.load(path, map_location='cpu')
        restore_rng_state(snapshot)
        actual = (random.random(), float(np.random.rand()), float(torch.rand(())))
        self.assertEqual(actual, expected)

    def test_sparse_large_lineage_has_only_observed_pairs(self):
        size = 20000
        previous = np.repeat(np.arange(size, dtype=np.int64), 2)
        current = previous + 100000
        overlap, _, _ = build_overlap_contingency(previous, current)
        self.assertEqual(len(overlap), size)
        lineage = compute_mutual_lineage(previous, current, 0.5)
        self.assertEqual(len(lineage), size)
        self.assertEqual(lineage[100123], 123)

    def test_update_uses_actual_edges_not_label_translation(self):
        updated, continued, switched = update_edge_persistence(
            [(4, 8), (4, 9)], {4: 1}, {8: 2, 9: 3},
            {(1, 2): 0.36}, rho=0.8)
        self.assertEqual(set(updated), {(4, 8), (4, 9)})
        self.assertAlmostEqual(updated[(4, 8)], 0.488)
        self.assertAlmostEqual(updated[(4, 9)], 0.2)
        self.assertEqual((continued, switched), (1, 1))


if __name__ == '__main__':
    unittest.main()
