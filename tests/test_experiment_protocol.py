import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

from clustercontrast.methods.checkpoint import (
    final_checkpoint_path, save_fixed_epoch_checkpoint,
    should_evaluate_during_train,
)


ROOT = Path(__file__).resolve().parents[1]


class ExperimentProtocolTests(unittest.TestCase):
    def test_fixed_final_checkpoint_is_independent_of_test_rank(self):
        with tempfile.TemporaryDirectory() as directory:
            ranks = [0.99, 0.10, 0.20]
            for epoch, _rank in enumerate(ranks, start=1):
                save_fixed_epoch_checkpoint(
                    {'epoch': epoch, 'weight': torch.tensor([epoch])},
                    directory, is_final_epoch=(epoch == len(ranks)))
                if epoch < len(ranks):
                    self.assertFalse(Path(final_checkpoint_path(directory)).exists())
            final = torch.load(final_checkpoint_path(directory), map_location='cpu')
            latest = torch.load(Path(directory) / 'checkpoint.pth.tar', map_location='cpu')
            self.assertEqual(final['epoch'], 3)
            self.assertEqual(latest['epoch'], 3)

    def test_stage2_loads_stage1_final_checkpoint(self):
        for script in ('train_agreid.py', 'train_lag.py'):
            source = (ROOT / script).read_text(encoding='utf-8')
            self.assertIn('final_checkpoint_path(', source)
            self.assertNotIn('model_best.pth.tar', source)

    def test_evaluation_gate_defaults_off_and_never_controls_saving(self):
        args = SimpleNamespace(eval_during_train=False, eval_step=1)
        self.assertFalse(should_evaluate_during_train(args, 0))
        args.eval_during_train = True
        args.eval_step = 2
        self.assertFalse(should_evaluate_during_train(args, 0))
        self.assertTrue(should_evaluate_during_train(args, 1))
        for script in ('train_agreid.py', 'train_lag.py'):
            source = (ROOT / script).read_text(encoding='utf-8')
            self.assertNotIn('is_best', source)
            self.assertNotIn('best_R1', source)

    def test_rahp_stage1_requires_cmhybrid_but_stage2_only_is_allowed(self):
        for script in ('train_agreid.py', 'train_lag.py'):
            blocked = subprocess.run(
                [sys.executable, str(ROOT / script), '--dry-run', '--use-rahp'],
                cwd=ROOT, capture_output=True, text=True)
            self.assertNotEqual(blocked.returncode, 0)
            self.assertIn('RAHP Stage1 requires --memorybank CMhybrid.',
                          blocked.stdout + blocked.stderr)
            allowed = subprocess.run(
                [sys.executable, str(ROOT / script), '--dry-run', '--stage2-only',
                 '--use-rahp'], cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(allowed.returncode, 0, allowed.stdout + allowed.stderr)

    def test_evaluation_scripts_expose_parameterized_paths(self):
        for script in ('test_agreid.py', 'test_LAG.py'):
            completed = subprocess.run(
                [sys.executable, str(ROOT / script), '--help'], cwd=ROOT,
                capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            for flag in ('--data-dir', '--checkpoint', '--trial', '--batch-size', '--workers'):
                self.assertIn(flag, completed.stdout)
            source = (ROOT / script).read_text(encoding='utf-8')
            self.assertNotIn('/home/', source)
            self.assertNotIn('model_best.pth.tar', source)


if __name__ == '__main__':
    unittest.main()
