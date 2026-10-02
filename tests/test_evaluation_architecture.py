import gc
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import torch
from torch import nn

from clustercontrast import models
from clustercontrast.methods.evaluation import load_agw_checkpoint_strict
from clustercontrast.models.resnet_agw import (
    resolve_resnet50_pretrained, resnet50,
)


ROOT = Path(__file__).resolve().parents[1]


class EvaluationArchitectureTests(unittest.TestCase):
    def test_test_entrypoints_default_to_agw(self):
        for script in ('test_agreid.py', 'test_LAG.py'):
            completed = subprocess.run(
                [sys.executable, str(ROOT / script), '--help'],
                cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn('--arch {agw}', completed.stdout)
            self.assertIn('default: agw', completed.stdout)

    def test_agw_checkpoint_roundtrip_is_strict(self):
        with tempfile.TemporaryDirectory() as directory:
            pretrained = Path(directory) / 'resnet50-19c8e357.pth'
            backbone = resnet50(pretrained=False)
            torch.save(backbone.state_dict(), pretrained)
            del backbone

            training_model = nn.DataParallel(models.create(
                'agw', num_features=0, norm=True, dropout=0,
                num_classes=0, pooling_type='gem',
                pretrained_path=str(pretrained)))
            checkpoint_path = Path(directory) / 'model_final.pth.tar'
            torch.save({'state_dict': training_model.state_dict()}, checkpoint_path)
            del training_model
            gc.collect()

            evaluation_model = nn.DataParallel(models.create(
                'agw', num_features=0, norm=True, dropout=0,
                num_classes=0, pooling_type='gem',
                pretrained_path=str(pretrained)))
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            load_agw_checkpoint_strict(evaluation_model, checkpoint)

            incompatible = dict(checkpoint['state_dict'])
            incompatible.pop(next(iter(incompatible)))
            with self.assertRaisesRegex(
                    RuntimeError, 'Checkpoint architecture does not match AGW'):
                load_agw_checkpoint_strict(
                    evaluation_model, {'state_dict': incompatible})

    def test_pretrained_path_priority_and_missing_error(self):
        with tempfile.TemporaryDirectory() as directory:
            cli_path = Path(directory) / 'cli.pth'
            env_path = Path(directory) / 'env.pth'
            cli_path.write_bytes(b'cli')
            env_path.write_bytes(b'env')
            with mock.patch.dict(os.environ, {
                    'PCLHD_RESNET50_PRETRAINED': str(env_path)}):
                self.assertEqual(
                    resolve_resnet50_pretrained(str(cli_path)),
                    str(cli_path.resolve()))
                self.assertEqual(
                    resolve_resnet50_pretrained(), str(env_path.resolve()))
            with self.assertRaisesRegex(
                    FileNotFoundError, 'Provide --pretrained-resnet50'):
                resolve_resnet50_pretrained(str(Path(directory) / 'missing.pth'))


if __name__ == '__main__':
    unittest.main()
