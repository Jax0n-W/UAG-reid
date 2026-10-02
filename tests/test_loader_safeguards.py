import unittest

import numpy as np

from clustercontrast.utils.data import IterLoader
from train_agreid import associated_analysis_for_all


class _EmptyLoader:
    def __iter__(self):
        return iter(())

    def __len__(self):
        return 0


class LoaderSafeguardTests(unittest.TestCase):
    def test_empty_iterloader_fails_with_actionable_error(self):
        loader = IterLoader(_EmptyLoader(), length=1)
        loader.new_epoch()
        with self.assertRaisesRegex(RuntimeError, 'produced no batches'):
            loader.next()

    def test_all_association_uses_concat_boundary(self):
        labels = np.asarray([0, 1, -1, 0, 2, 1], dtype=np.int32)
        flags_ir, flags_rgb = associated_analysis_for_all(
            labels, num_ground_samples=3, log_dir=None)
        self.assertEqual(flags_rgb[0], 1)
        self.assertEqual(flags_ir[0], 1)
        self.assertEqual(flags_rgb[1], 1)
        self.assertEqual(flags_ir[1], 1)
        self.assertEqual(flags_rgb[2], 0)
        self.assertEqual(flags_ir[2], 1)


if __name__ == '__main__':
    unittest.main()
