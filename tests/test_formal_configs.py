from pathlib import Path
import re
import shlex
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = {
    '--arch': 'agw',
    '--memorybank': 'CMhybrid',
    '--height': '288',
    '--width': '144',
    '--batch-size': '64',
    '--num-instances': '16',
    '--epochs': '50',
    '--iters': '400',
    '--lr': '0.00035',
    '--weight-decay': '0.0005',
    '--momentum': '0.2',
    '--step-size': '20',
    '--eps': '$EPS',
    '--k1': '30',
    '--k2': '6',
    '--temp': '0.05',
    '--seed': '1',
    '--trial': '1',
    '--workers': '8',
    '--pooling-type': 'gem',
    '--rahp-beta': '0.25',
    '--rahp-knn': '20',
    '--rahp-alpha': '0.5',
    '--cesa-rho': '0.8',
    '--cesa-eta': '0.1',
    '--cesa-lineage-thr': '0.5',
    '--cesa-warmup': '5',
}


def common_tokens(source):
    block = re.search(r'COMMON_ARGS=\((.*?)\n\)', source, re.DOTALL)
    if block is None:
        raise AssertionError('COMMON_ARGS not found')
    return shlex.split(block.group(1), comments=True, posix=True)


class FormalConfigTests(unittest.TestCase):
    def test_four_groups_share_one_frozen_configuration(self):
        expected_runs = [
            'run_experiment baseline',
            'run_experiment rahp --use-rahp',
            'run_experiment cesa --use-cesa',
            'run_experiment full --use-rahp --use-cesa',
        ]
        for script in ('run_agreid_ablation.sh', 'run_lagper_ablation.sh'):
            source = (ROOT / 'scripts' / script).read_text(encoding='utf-8')
            tokens = common_tokens(source)
            for option, value in REQUIRED.items():
                self.assertIn(option, tokens, (script, option))
                if value is not None:
                    self.assertEqual(tokens[tokens.index(option) + 1], value)
            run_lines = [line.strip() for line in source.splitlines()
                         if line.startswith('run_experiment ')]
            self.assertEqual(run_lines, expected_runs)
            self.assertNotIn('--use-rahp', tokens)
            self.assertNotIn('--use-cesa', tokens)
            self.assertIn('EPS="${EPS:-0.6}"', source)

    def test_agreid_uses_official_historical_best_protocol(self):
        source = (ROOT / 'scripts' / 'run_agreid_ablation.sh').read_text(
            encoding='utf-8')
        tokens = common_tokens(source)
        self.assertIn('--eval-during-train=True', tokens)
        self.assertEqual(tokens[tokens.index('--eval-step') + 1], '1')
        self.assertEqual(tokens[tokens.index('--stage1-init') + 1], 'best')
        self.assertNotIn('--stage1-init final', source)
        self.assertIn(
            'OFFICIAL AG-ReID HISTORICAL ABLATION PROTOCOL:', source)
        self.assertIn('Stage1-best -> Stage2 -> Stage2-best', source)
        self.assertIn('--logs-dir "$LOGS_DIR/$name"', source)

    def test_lagper_retains_fixed_final_protocol(self):
        source = (ROOT / 'scripts' / 'run_lagper_ablation.sh').read_text(
            encoding='utf-8')
        tokens = common_tokens(source)
        self.assertIn('--eval-during-train=False', tokens)
        self.assertNotIn('--stage1-init', tokens)

    def test_frozen_batch_contains_complete_sampler_groups(self):
        self.assertGreaterEqual(64, 2 * 16)
        self.assertEqual((64 // 2) % 16, 0)

    def test_eps_cli_override_is_visible_in_dry_run(self):
        for script in ('train_agreid.py', 'train_lag.py'):
            completed = subprocess.run(
                [sys.executable, str(ROOT / script), '--eps', '0.55',
                 '--dry-run'], cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn('eps=0.55', completed.stdout)

    def test_clustering_uses_cli_eps_and_knn_values(self):
        for script in ('train_agreid.py', 'train_lag.py'):
            source = (ROOT / script).read_text(encoding='utf-8')
            self.assertIn('ir_eps = args.eps', source)
            self.assertIn('rgb_eps = args.eps', source)
            self.assertIn('all_eps = args.eps', source)
            self.assertNotIn('ir_eps = 0.3', source)
            self.assertNotIn('rgb_eps = 0.3', source)
            self.assertNotIn('all_eps = 0.3', source)
            self.assertNotIn('k1=40, k2=32', source)


if __name__ == '__main__':
    unittest.main()
