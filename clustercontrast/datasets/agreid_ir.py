from __future__ import print_function, absolute_import
import os.path as osp
import glob
import re
from ..utils.data import BaseImageDataset


class agreid_ir(BaseImageDataset):
    """
    AG-ReID aerial images -> IR modality (thermal)
    
    Server structure:
    dataset_root/aerial_modify/{trial}/bounding_box_train/   (training)
    dataset_root/bounding_box_test_aerial/            (gallery)
    dataset_root/query_all_aerial/                    (query)
    dataset_root/idx/                                 (protocol files)
    """
    dataset_dir = 'aerial_modify'

    def __init__(self, root, trial=1, verbose=True, **kwargs):
        super(agreid_ir, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, str(trial), 'bounding_box_train')

        # Test data is at root level (not under aerial_modify/)
        self.query_dir = osp.join(root, 'query_all_aerial')
        self.gallery_dir = osp.join(root, 'bounding_box_test_aerial')

        self._check_before_run()

        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        if verbose:
            print("=> agreid_ir (aerial) loaded, trial={}".format(trial))
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        # query/gallery may not exist in train-only setup
        if not osp.exists(self.query_dir):
            print("Warning: '{}' is not available, query will be empty".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            print("Warning: '{}' is not available, gallery will be empty".format(self.gallery_dir))

    def _process_dir(self, dir_path, relabel=False):
        if not osp.exists(dir_path):
            return []
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        # pattern: P{pid}_T{traj_id}_C{camera_id}_F{frame}.jpg
        # Example: P0034_T02210_C10_F19471.jpg -> pid=0034, camera_id=10
        pattern = re.compile(r'P(\d+)_T\w+_C(\d+)_F\d+\.jpg')

        pid_container = set()
        for img_path in img_paths:
            match = pattern.search(img_path)
            if match is None:
                continue
            pid = int(match.group(1))
            if pid == -1:
                continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            match = pattern.search(img_path)
            if match is None:
                continue
            pid = int(match.group(1))
            camid = int(match.group(2))
            if pid == -1:
                continue  # junk images are just ignored
            if relabel:
                pid = pid2label[pid]
            dataset.append((img_path, pid, camid))

        return dataset
