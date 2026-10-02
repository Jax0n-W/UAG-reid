import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from clustercontrast.datasets.lag_ir import lag_ir
from clustercontrast.datasets.lag_rgb import lag_rgb
from clustercontrast.datasets.lagper_protocol import (
    evaluation_arrays, parse_lagper_filename,
)
from test_LAG import eval_lagper, valid_gallery_mask


ROOT = Path(__file__).resolve().parents[1]


def make_prepared_tree(root):
    root = Path(root)
    meta = root / 'meta'
    images = root / 'images'
    meta.mkdir()
    images.mkdir()
    protocol = {
        'dataset': 'LAGPeR',
        'protocol': 'official-scene-split',
        'train_scenes': 4,
        'test_scenes': 3,
        'train_aerial_cameras': 4,
        'train_ground_cameras': 8,
        'test_aerial_cameras': 3,
        'test_ground_cameras': 6,
    }
    (meta / 'protocol.json').write_text(json.dumps(protocol), encoding='utf-8')
    specs = {
        'train_aerial.txt': ('0001_c03_000001.jpg', 1, 3, 'aerial'),
        'train_ground.txt': ('0001_c13_000001.jpg', 1, 13, 'ground'),
        'query_aerial.txt': ('0002_c21_000001.jpg', 2, 21, 'aerial'),
        'gallery_aerial.txt': ('0002_c22_000001.jpg', 2, 22, 'aerial'),
        'query_ground.txt': ('0002_c03_000002.jpg', 2, 3, 'ground'),
        'gallery_ground.txt': ('0002_c13_000002.jpg', 2, 13, 'ground'),
    }
    for manifest, (filename, pid, camid, view) in specs.items():
        (images / filename).write_bytes(b'fixture')
        (meta / manifest).write_text(
            'images/{} {} {} {}\n'.format(filename, pid, camid, view),
            encoding='utf-8')


class LAGPeRProtocolTests(unittest.TestCase):
    def test_camera_parser_preserves_real_camera_number(self):
        self.assertEqual(parse_lagper_filename('0001_c03_000002.jpg'), (1, 3))
        self.assertEqual(parse_lagper_filename('0001_c13_000008.jpg'), (1, 13))

    def test_dataset_classes_read_only_official_prepared_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            make_prepared_tree(directory)
            aerial = lag_ir(directory, trial=1, verbose=False)
            ground = lag_rgb(directory, trial=1, verbose=False)
            self.assertEqual(aerial.train[0][2], 3)
            self.assertEqual(ground.train[0][2], 13)
            paths, pids, camids = evaluation_arrays(
                directory, 'gallery', 'ground')
            self.assertTrue(Path(paths[0]).is_file())
            self.assertEqual((pids.tolist(), camids.tolist()), ([2], [13]))

    def test_missing_official_metadata_fails_fast(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, 'Official LAGPeR prepared metadata'):
                lag_ir(directory, verbose=False)

    def test_legacy_prepare_script_is_blocked_and_has_no_random_split(self):
        source = (ROOT / 'prepare_lag.py').read_text(encoding='utf-8')
        self.assertIn('NOT FOR FORMAL LAGPER EXPERIMENTS', source)
        self.assertNotIn('random.shuffle', source)
        self.assertNotIn('train_ratio', source)
        completed = subprocess.run(
            [sys.executable, str(ROOT / 'prepare_lag.py')],
            cwd=ROOT, capture_output=True, text=True)
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn('BLOCKED: NEED REAL LAGPER DATA TREE',
                      completed.stdout + completed.stderr)
        training_source = (ROOT / 'train_lag.py').read_text(encoding='utf-8')
        self.assertNotIn("'aerial_modify' in", training_source)
        self.assertNotIn("'ground_modify' in", training_source)

    def test_same_identity_same_camera_filter_keeps_other_camera_positives(self):
        gallery_pids = np.array([1, 1, 1, 2])
        gallery_camids = np.array([3, 13, 21, 8])
        keep = valid_gallery_mask(1, 3, gallery_pids, gallery_camids)
        self.assertEqual(keep.tolist(), [False, True, True, True])
        distances = np.array([[0.0, 0.1, 0.2, 0.3]])
        cmc, mean_ap, _ = eval_lagper(
            distances, np.array([1]), gallery_pids,
            q_camids=np.array([3]), g_camids=gallery_camids)
        self.assertEqual(float(cmc[0]), 1.0)
        self.assertGreater(mean_ap, 0.0)


if __name__ == '__main__':
    unittest.main()
