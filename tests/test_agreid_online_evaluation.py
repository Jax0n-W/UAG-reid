import random
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
from PIL import Image
import torch
from torch import nn

import train_agreid
from clustercontrast.methods.checkpoint import (
    BEST_SELECTION_METRIC, best_checkpoint_path, capture_rng_state,
    final_checkpoint_path, restore_best_state, save_best_checkpoint,
    save_fixed_epoch_checkpoint, select_agreid_best,
    should_evaluate_during_train,
)
from clustercontrast.methods.evaluation import (
    evaluate_agreid_bidirectional, evaluate_agreid_direction,
    evaluate_agreid_distances, extract_agreid_features,
)
from clustercontrast.utils.serialization import load_torch_file
from test_agreid import eval_agreid


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


def _checkpoint_state(epoch, best_rank1, best_epoch, a2g, g2a):
    return {
        'state_dict': {'ema': torch.tensor([epoch])},
        'model_state_dict': {'online': torch.tensor([epoch])},
        'optimizer_state_dict': {'epoch': epoch},
        'scheduler_state_dict': {'last_epoch': epoch - 1},
        'rng_state': capture_rng_state(),
        'dbscan_eps': (0.6, 0.6, 0.6),
        'cesa_state': None,
        'epoch': epoch,
        'best_R1': best_rank1,
        'best_epoch': best_epoch,
        'selection_metric': BEST_SELECTION_METRIC,
        'eval_g2a': g2a,
        'eval_a2g': a2g,
    }


class AGReIDOnlineEvaluationTests(unittest.TestCase):
    def test_evaluation_disabled_does_not_create_best_checkpoint(self):
        args = SimpleNamespace(eval_during_train=False, eval_step=1)
        with tempfile.TemporaryDirectory() as directory:
            self.assertFalse(should_evaluate_during_train(args, 0))
            save_fixed_epoch_checkpoint(
                {'epoch': 1}, directory, is_final_epoch=True)
            self.assertFalse(Path(best_checkpoint_path(directory)).exists())
            self.assertTrue(Path(final_checkpoint_path(directory)).exists())

    def test_best_selection_uses_strict_g2a_rank1(self):
        best_rank1, best_epoch = float('-inf'), None
        best_rank1, best_epoch, updated = select_agreid_best(
            _metrics(0.40), _metrics(0.50), best_rank1, best_epoch, 1)
        self.assertEqual((best_rank1, best_epoch, updated), (0.50, 1, True))

        best_rank1, best_epoch, updated = select_agreid_best(
            _metrics(0.45), _metrics(0.55), best_rank1, best_epoch, 2)
        self.assertEqual((best_rank1, best_epoch, updated), (0.55, 2, True))

        best_rank1, best_epoch, updated = select_agreid_best(
            _metrics(0.90), _metrics(0.52), best_rank1, best_epoch, 3)
        self.assertEqual((best_rank1, best_epoch, updated), (0.55, 2, False))

        best_rank1, best_epoch, updated = select_agreid_best(
            _metrics(0.99), _metrics(0.55), best_rank1, best_epoch, 4)
        self.assertEqual((best_rank1, best_epoch, updated), (0.55, 2, False))

    def test_latest_final_and_best_checkpoints_are_independent(self):
        with tempfile.TemporaryDirectory() as directory:
            a2g = _metrics(0.90)
            g2a = _metrics(0.55)
            best_state = _checkpoint_state(2, 0.55, 2, a2g, g2a)
            save_best_checkpoint(best_state, directory)
            save_fixed_epoch_checkpoint(
                _checkpoint_state(3, 0.55, 2, _metrics(0.95), _metrics(0.52)),
                directory, is_final_epoch=True)

            best = load_torch_file(best_checkpoint_path(directory))
            latest = load_torch_file(Path(directory) / 'checkpoint.pth.tar')
            final = load_torch_file(final_checkpoint_path(directory))
            self.assertEqual(best['epoch'], 2)
            self.assertEqual(latest['epoch'], 3)
            self.assertEqual(final['epoch'], 3)
            required = {
                'state_dict', 'model_state_dict', 'optimizer_state_dict',
                'scheduler_state_dict', 'rng_state', 'dbscan_eps',
                'cesa_state', 'epoch', 'best_R1', 'best_epoch',
                'selection_metric', 'eval_g2a', 'eval_a2g',
            }
            self.assertTrue(required.issubset(best))
            self.assertEqual(best['selection_metric'], 'g2a_rank1')
            self.assertEqual(best['best_R1'], 0.55)
            self.assertEqual(best['best_epoch'], 2)

    def test_legacy_checkpoint_restores_default_best_state(self):
        best_rank1, best_epoch = restore_best_state({'epoch': 7})
        self.assertEqual(best_rank1, float('-inf'))
        self.assertIsNone(best_epoch)

    def test_a2g_and_g2a_modal_routing(self):
        features = [np.asarray([[1.0, 0.0]]), np.asarray([[1.0, 0.0]])]
        with mock.patch(
                'clustercontrast.methods.evaluation.extract_agreid_features',
                side_effect=features) as extract:
            result = evaluate_agreid_direction(
                object(), 'a2g', object(), np.asarray([0]),
                object(), np.asarray([0]))
            self.assertEqual([call.args[3] for call in extract.call_args_list], [2, 1])
            self.assertEqual(result['rank1'], 1.0)

        with mock.patch(
                'clustercontrast.methods.evaluation.extract_agreid_features',
                side_effect=features) as extract:
            evaluate_agreid_direction(
                object(), 'g2a', object(), np.asarray([0]),
                object(), np.asarray([0]))
            self.assertEqual([call.args[3] for call in extract.call_args_list], [1, 2])

    def test_bidirectional_evaluation_extracts_each_modality_once(self):
        features = [np.asarray([[1.0, 0.0]]), np.asarray([[1.0, 0.0]])]
        with mock.patch(
                'clustercontrast.methods.evaluation.extract_agreid_features',
                side_effect=features) as extract:
            a2g, g2a = evaluate_agreid_bidirectional(
                object(), object(), np.asarray([0]),
                object(), np.asarray([0]))

        self.assertEqual(
            [call.kwargs['modal'] for call in extract.call_args_list], [2, 1])
        self.assertEqual(extract.call_count, 2)
        self.assertEqual(a2g['rank1'], 1.0)
        self.assertEqual(g2a['rank1'], 1.0)

    def test_train_and_formal_evaluators_are_identical(self):
        query_labels = np.asarray([0, 1])
        gallery_labels = np.asarray([0, 1] * 10)
        distances = np.arange(40, dtype=np.float64).reshape(2, 20) / 40.0

        train_result = train_agreid.eval_regdb(
            distances, query_labels, gallery_labels)
        formal_result = eval_agreid(
            distances, query_labels, gallery_labels)

        np.testing.assert_array_equal(train_result[0], formal_result[0])
        self.assertEqual(train_result[1:], formal_result[1:])

    def test_canonical_feature_extraction_flips_averages_and_normalizes(self):
        class FeatureModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.modals = []

            def forward(self, first, _second, modal):
                self.modals.append(modal)
                return first[:, 0, 0, :]

        model = FeatureModel()
        inputs = torch.tensor([[[[1.0, 3.0]]]])
        loader = [(inputs, torch.tensor([0]))]
        with mock.patch.object(torch.Tensor, 'cuda', lambda tensor: tensor):
            features = extract_agreid_features(
                model, loader, num_samples=1, modal=2)

        expected = np.asarray([[2 ** -0.5, 2 ** -0.5]])
        np.testing.assert_allclose(features, expected, rtol=1e-6)
        self.assertEqual(model.modals, [2, 2])

    def test_canonical_evaluator_rejects_invalid_results(self):
        with self.assertRaisesRegex(RuntimeError, 'non-finite'):
            evaluate_agreid_distances(
                np.asarray([[np.nan]]), np.asarray([0]), np.asarray([0]))
        with self.assertRaisesRegex(RuntimeError, 'no valid query'):
            evaluate_agreid_distances(
                np.asarray([[0.0]]), np.asarray([0]), np.asarray([1]))

    def test_training_evaluation_restores_rng_and_model_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            index_dir = Path(directory) / 'idx'
            index_dir.mkdir()
            image_path = Path(directory) / 'image.jpg'
            Image.new('RGB', (4, 4), color=(10, 20, 30)).save(image_path)
            for modality in ('aerial', 'ground'):
                (index_dir / 'test_{}_1.txt'.format(modality)).write_text(
                    'image.jpg 0\n', encoding='utf-8')

            args = SimpleNamespace(
                height=288, width=144, test_batch=64, workers=0)
            model = nn.Linear(1, 1)
            model.train()

            random.seed(11)
            np.random.seed(11)
            torch.manual_seed(11)
            expected = (random.random(), np.random.random(), torch.rand(1))
            random.seed(11)
            np.random.seed(11)
            torch.manual_seed(11)

            def consume_rng(*_args, **_kwargs):
                random.random()
                np.random.random()
                torch.rand(1)
                return _metrics(0.5)

            with mock.patch(
                    'train_agreid.evaluate_agreid_bidirectional',
                    side_effect=consume_rng):
                train_agreid.evaluate_agreid_for_training(
                    model, args, directory, 1)

            actual = (random.random(), np.random.random(), torch.rand(1))
            self.assertEqual(actual[0], expected[0])
            self.assertEqual(actual[1], expected[1])
            torch.testing.assert_close(actual[2], expected[2])
            self.assertTrue(model.training)

    def test_eval_cli_flags_parse(self):
        completed = subprocess.run(
            [sys.executable, str(ROOT / 'train_agreid.py'), '--dry-run',
             '--stage2-only', '--eval-during-train=True', '--eval-step', '1'],
            cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertIn('eval_during_train=True', completed.stdout)


if __name__ == '__main__':
    unittest.main()
