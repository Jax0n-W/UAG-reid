from __future__ import print_function, absolute_import
import os.path as osp
import os
import re
from ..utils.data import BaseImageDataset


class lag_ir(BaseImageDataset):
    """
    LAG aerial images -> IR modality (thermal)

    Actual data structure (from prepare_lag.py):
    dataset_root/
      aerial_modify/{trial}/bounding_box_train/   (training, per-trial)
      query_aerial/                                (test query, with PID subdirs)
      bounding_box_test_aerial/                    (test gallery, with PID subdirs)
      idx/                                         (protocol files)
    """
    dataset_dir = ''

    def __init__(self, root, trial=1, verbose=True, **kwargs):
        super(lag_ir, self).__init__()
        self.dataset_dir = root  # root is the dataset root e.g. C:/.../LAG/
        # Training data: organized per-trial under aerial_modify/
        self.train_dir = osp.join(self.dataset_dir, 'aerial_modify', str(trial), 'bounding_box_train')

        # Test data paths
        self.query_dir = osp.join(self.dataset_dir, 'query_aerial')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test_aerial')

        self._check_before_run()

        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        if verbose:
            print("=> lag_ir (aerial/IR) loaded, trial={}".format(trial))
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    @property
    def images_dir(self):
        return self.dataset_dir

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            print("Warning: '{}' is not available, query will be empty".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            print("Warning: '{}' is not available, gallery will be empty".format(self.gallery_dir))

    def _process_dir(self, dir_path, relabel=False):
        if not osp.exists(dir_path):
            return []
        # Handle both flat files (train dirs) and nested PID subdirs (test dirs)
        img_paths = []
        for root_dir, _, files in os.walk(dir_path):
            for fname in files:
                if fname.endswith('.jpg'):
                    img_paths.append(osp.join(root_dir, fname))
        # Pattern: {pid}_c{camid}_{frame}.jpg
        # Example: 0001_c3_000002.jpg -> pid=0001, camera_id=3
        pattern = re.compile(r'(\d+)_c(\d+)_\d+\.jpg')

        pid_container = set()
        for img_path in img_paths:
            match = pattern.search(osp.basename(img_path))
            if match is None:
                continue
            pid = int(match.group(1))
            if pid == -1:
                continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            match = pattern.search(osp.basename(img_path))
            if match is None:
                continue
            pid = int(match.group(1))
            camid = int(match.group(2))
            if pid == -1:
                continue  # junk images are just ignored
            if relabel:
                pid = pid2label[pid]
            # Store relative path for Preprocessor compatibility
            rel_path = osp.relpath(img_path, self.images_dir)
            dataset.append((rel_path, pid, camid))

        return dataset