from pathlib import Path
import re
import shlex
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
    '--eps': '0.6',
    '--k1': '30',
    '--k2': '6',
    '--temp': '0.05',
    '--seed': '1',
    '--trial': '1',
    '--workers': '8',
    '--pooling-type': 'gem',
    '--eval-during-train=False': None,
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

    def test_frozen_batch_contains_complete_sampler_groups(self):
        self.assertGreaterEqual(64, 2 * 16)
        self.assertEqual((64 // 2) % 16, 0)


if __name__ == '__main__':
    unittest.main()
