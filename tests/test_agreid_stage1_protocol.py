import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

from clustercontrast.methods.checkpoint import (
    BEST_SELECTION_METRIC, best_checkpoint_path, final_checkpoint_path,
    resolve_stage1_initialization, save_best_checkpoint,
    save_fixed_epoch_checkpoint, select_agreid_best,
    should_evaluate_during_train,
)
from clustercontrast.utils.serialization import load_torch_file


ROOT = Path(__file__).resolve().parents[1]


def _metrics(rank1):
    return {
        'rank1': rank1,
        'rank5': rank1,
        'rank10': rank1,
        'rank20': rank1,
        'mAP': rank1,
        'mINP': rank1,
    }


class AGReIDStage1ProtocolTests(unittest.TestCase):
    def test_stage1_uses_shared_bidirectional_evaluation_and_logs(self):
        source = (ROOT / 'train_agreid.py').read_text(encoding='utf-8')
        stage1 = source.split('def main_worker_stage1', 1)[1].split(
            'def main_worker_stage2', 1)[0]
        self.assertIn('evaluate_agreid_for_training(', stage1)
        self.assertIn("_print_agreid_metrics('Stage1', epoch + 1, 'a2g'", stage1)
        self.assertIn("_print_agreid_metrics('Stage1', epoch + 1, 'g2a'", stage1)
        self.assertNotIn('Test Trial:', stage1)
        self.assertNotIn('Debug evaluation epoch', stage1)

    def test_stage1_best_sequence_and_a2g_is_diagnostic_only(self):
        best_rank1, best_epoch = float('-inf'), None
        cases = [
            (0.20, 0.40, 1, True, 1),
            (0.30, 0.50, 2, True, 2),
            (0.90, 0.45, 3, False, 2),
            (0.99, 0.50, 4, False, 2),
        ]
        for a2g, g2a, epoch, expected_update, expected_epoch in cases:
            best_rank1, best_epoch, updated = select_agreid_best(
                _metrics(a2g), _metrics(g2a),
                best_rank1, best_epoch, epoch)
            self.assertEqual(updated, expected_update)
            self.assertEqual(best_epoch, expected_epoch)
        self.assertEqual(best_rank1, 0.50)

    def test_stage1_latest_final_and_best_are_independent(self):
        with tempfile.TemporaryDirectory() as directory:
            best_state = {
                'state_dict': {'weight': torch.tensor([2])},
                'epoch': 2,
                'best_R1': 0.50,
                'best_epoch': 2,
                'selection_metric': BEST_SELECTION_METRIC,
                'eval_a2g': _metrics(0.30),
                'eval_g2a': _metrics(0.50),
            }
            save_best_checkpoint(best_state, directory)
            final_state = dict(best_state, epoch=3)
            final_state['state_dict'] = {'weight': torch.tensor([3])}
            final_state['eval_g2a'] = _metrics(0.45)
            save_fixed_epoch_checkpoint(
                final_state, directory, is_final_epoch=True)

            best = load_torch_file(best_checkpoint_path(directory))
            latest = load_torch_file(Path(directory) / 'checkpoint.pth.tar')
            final = load_torch_file(final_checkpoint_path(directory))
            self.assertEqual(best['epoch'], 2)
            self.assertEqual(latest['epoch'], 3)
            self.assertEqual(final['epoch'], 3)
            self.assertEqual(best['best_R1'], 0.50)
            self.assertEqual(best['best_epoch'], 2)
            self.assertEqual(best['selection_metric'], 'g2a_rank1')
            self.assertEqual(best['eval_g2a']['rank1'], 0.50)
            self.assertEqual(best['eval_a2g']['rank1'], 0.30)

    def test_stage1_initialization_selects_exact_requested_file(self):
        with tempfile.TemporaryDirectory() as directory:
            final_path = Path(final_checkpoint_path(directory))
            best_path = Path(best_checkpoint_path(directory))
            final_path.write_bytes(b'final')
            best_path.write_bytes(b'best')
            self.assertEqual(
                resolve_stage1_initialization(directory, 'final'),
                str(final_path))
            self.assertEqual(
                resolve_stage1_initialization(directory, 'best'),
                str(best_path))

    def test_missing_stage1_best_fails_without_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(final_checkpoint_path(directory)).write_bytes(b'final')
            with self.assertRaisesRegex(FileNotFoundError, 'Stage1 best'):
                resolve_stage1_initialization(directory, 'best')

    def test_stage2_resume_ignores_stage1_initialization(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertIsNone(resolve_stage1_initialization(
                directory, 'best', stage2_resume=True))

    def test_stage1_init_cli_defaults_to_final_and_accepts_best(self):
        default = subprocess.run(
            [sys.executable, str(ROOT / 'train_agreid.py'),
             '--dry-run', '--stage2-only'],
            cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(default.returncode, 0, default.stdout + default.stderr)
        self.assertIn('stage1_init=final', default.stdout)

        best = subprocess.run(
            [sys.executable, str(ROOT / 'train_agreid.py'),
             '--dry-run', '--stage2-only', '--stage1-init', 'best'],
            cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(best.returncode, 0, best.stdout + best.stderr)
        self.assertIn('stage1_init=best', best.stdout)

    def test_evaluation_disabled_does_not_create_stage1_best(self):
        args = SimpleNamespace(eval_during_train=False, eval_step=1)
        with tempfile.TemporaryDirectory() as directory:
            self.assertFalse(should_evaluate_during_train(args, 0))
            save_fixed_epoch_checkpoint(
                {'state_dict': {}, 'epoch': 1}, directory,
                is_final_epoch=True)
            self.assertFalse(Path(best_checkpoint_path(directory)).exists())


if __name__ == '__main__':
    unittest.main()
