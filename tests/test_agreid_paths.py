import os
import tempfile
import unittest

import train_agreid
from test_agreid import process_test_agreid


class AGReIDPathContractTests(unittest.TestCase):
    def test_training_loaders_use_shared_root_and_trial_directory(self):
        with tempfile.TemporaryDirectory() as root:
            aerial_dir = os.path.join(
                root, 'aerial_modify', '1', 'bounding_box_train')
            ground_dir = os.path.join(
                root, 'ground_modify', '1', 'bounding_box_train')
            os.makedirs(aerial_dir)
            os.makedirs(ground_dir)

            aerial_image = os.path.join(
                aerial_dir, 'P0001_T00001_C03_F00001.jpg')
            ground_image = os.path.join(
                ground_dir, 'P0002_T00002_C13_F00002.jpg')
            open(aerial_image, 'wb').close()
            open(ground_image, 'wb').close()

            aerial = train_agreid.get_data('agreid_ir', root, trial=1)
            ground = train_agreid.get_data('agreid_rgb', root, trial=1)

            self.assertEqual(aerial.train, [(aerial_image, 0, 3)])
            self.assertEqual(ground.train, [(ground_image, 0, 13)])

    def test_evaluation_reads_agreid_aerial_and_ground_indexes(self):
        with tempfile.TemporaryDirectory() as root:
            index_dir = os.path.join(root, 'idx')
            os.makedirs(index_dir)
            with open(os.path.join(index_dir, 'test_aerial_1.txt'), 'w') as f:
                f.write('aerial/image.jpg 7\n')
            with open(os.path.join(index_dir, 'test_ground_1.txt'), 'w') as f:
                f.write('ground/image.jpg 8\n')

            aerial_paths, aerial_pids = process_test_agreid(
                root, trial=1, modal='aerial')
            ground_paths, ground_pids = process_test_agreid(
                root, trial=1, modal='ground')

            self.assertEqual(aerial_paths, [os.path.join(root, 'aerial/image.jpg')])
            self.assertEqual(aerial_pids.tolist(), [7])
            self.assertEqual(ground_paths, [os.path.join(root, 'ground/image.jpg')])
            self.assertEqual(ground_pids.tolist(), [8])


if __name__ == '__main__':
    unittest.main()
