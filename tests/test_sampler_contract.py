import random
import unittest
from collections import Counter

import numpy as np
import torch
from torch.utils.data import DataLoader

from clustercontrast.utils.data.sampler import (
    RandomMultipleGallerySampler,
    RandomMultipleGallerySamplerNoCam,
)


NUM_INSTANCES = 16


def _sample_counts(data_source, indices):
    return Counter(data_source[index][1] for index in indices)


class SamplerContractTests(unittest.TestCase):
    def setUp(self):
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)

    def test_gallery_sampler_singleton_pid_yields_num_instances(self):
        data_source = [('singleton.jpg', 0, 0)]
        sampler = RandomMultipleGallerySampler(data_source, NUM_INSTANCES)

        indices = list(sampler)

        self.assertEqual(len(sampler), NUM_INSTANCES)
        self.assertEqual(len(indices), len(sampler))
        self.assertEqual(indices, [0] * NUM_INSTANCES)

    def test_gallery_sampler_multi_image_pid_yields_num_instances(self):
        data_source = [
            ('a.jpg', 0, 0),
            ('b.jpg', 0, 0),
            ('c.jpg', 0, 0),
        ]
        sampler = RandomMultipleGallerySampler(data_source, NUM_INSTANCES)

        indices = list(sampler)

        self.assertEqual(len(indices), len(sampler))
        self.assertEqual(len(indices), NUM_INSTANCES)
        self.assertGreater(len(set(indices)), 1)

    def test_gallery_sampler_mixed_four_pids_yields_full_batch(self):
        data_source = [
            ('p0.jpg', 0, 0),
            ('p1-a.jpg', 1, 0), ('p1-b.jpg', 1, 0),
            ('p2-a.jpg', 2, 0), ('p2-b.jpg', 2, 1),
            ('p3-a.jpg', 3, 0), ('p3-b.jpg', 3, 1), ('p3-c.jpg', 3, 2),
        ]
        sampler = RandomMultipleGallerySampler(data_source, NUM_INSTANCES)

        indices = list(sampler)

        self.assertEqual(len(indices), len(sampler))
        self.assertEqual(len(indices), 64)
        self.assertEqual(_sample_counts(data_source, indices), Counter({pid: 16 for pid in range(4)}))

        loader = DataLoader(
            list(range(len(data_source))),
            batch_size=64,
            sampler=sampler,
            drop_last=True,
            num_workers=0,
        )
        batches = list(loader)
        self.assertEqual(len(batches), 1)
        self.assertEqual(len(batches[0]), 64)

    def test_no_cam_sampler_singleton_pid_yields_num_instances(self):
        data_source = [('singleton.jpg', 0, 0)]
        sampler = RandomMultipleGallerySamplerNoCam(data_source, NUM_INSTANCES)

        indices = list(sampler)

        self.assertEqual(len(indices), len(sampler))
        self.assertEqual(indices, [0] * NUM_INSTANCES)

    def test_no_cam_sampler_mixed_pids_matches_reported_length(self):
        data_source = [
            ('p0.jpg', 0, 0),
            ('p1-a.jpg', 1, 0), ('p1-b.jpg', 1, 1),
            ('p2-a.jpg', 2, 0), ('p2-b.jpg', 2, 1), ('p2-c.jpg', 2, 2),
            ('p3.jpg', 3, 0),
        ]
        sampler = RandomMultipleGallerySamplerNoCam(data_source, NUM_INSTANCES)

        indices = list(sampler)

        self.assertEqual(len(indices), len(sampler))
        self.assertEqual(len(indices), 64)
        self.assertEqual(_sample_counts(data_source, indices), Counter({pid: 16 for pid in range(4)}))

    def test_gallery_sampler_retains_different_camera_preference(self):
        data_source = [
            ('cam0.jpg', 0, 0),
            ('cam1.jpg', 0, 1),
        ]
        sampler = RandomMultipleGallerySampler(data_source, NUM_INSTANCES)

        indices = list(sampler)

        selected_camera = data_source[indices[0]][2]
        self.assertEqual(len(indices), len(sampler))
        self.assertTrue(all(data_source[index][2] != selected_camera for index in indices[1:]))

    def test_agreid_like_all_loader_produces_a_batch(self):
        data_source = [('aerial-singleton.jpg', 0, 0)]
        data_source.extend((f'aerial-{index}.jpg', 1, index % 3) for index in range(300))
        data_source.extend((f'ground-{index}.jpg', 2, index % 2) for index in range(200))
        data_source.extend((f'ground-extra-{index}.jpg', 3, index % 2) for index in range(180))
        sampler = RandomMultipleGallerySampler(data_source, NUM_INSTANCES)
        loader = DataLoader(
            list(range(len(data_source))),
            batch_size=64,
            sampler=sampler,
            drop_last=True,
            num_workers=0,
        )

        batch = next(iter(loader))

        self.assertEqual(len(sampler), 64)
        self.assertEqual(len(batch), 64)


if __name__ == '__main__':
    unittest.main()
