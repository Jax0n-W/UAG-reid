from __future__ import print_function, absolute_import

from ..utils.data import BaseImageDataset
from .lagper_protocol import load_lagper_records


class lag_rgb(BaseImageDataset):
    """Official prepared LAGPeR ground domain; no split is created here."""

    dataset_dir = ''

    def __init__(self, root, trial=1, verbose=True, **kwargs):
        super(lag_rgb, self).__init__()
        if trial != 1:
            raise ValueError('Official LAGPeR scene split has one fixed trial (trial=1).')
        self.dataset_dir = root
        train = self._training_records('ground')
        query = load_lagper_records(root, 'query', 'ground')
        gallery = load_lagper_records(root, 'gallery', 'ground')
        if verbose:
            print('=> lag_rgb (official ground split) loaded')
            self.print_dataset_statistics(train, query, gallery)
        self.train, self.query, self.gallery = train, query, gallery
        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(gallery)

    @property
    def images_dir(self):
        return self.dataset_dir

    def _training_records(self, view):
        records = load_lagper_records(self.dataset_dir, 'train', view)
        pid2label = {pid: label for label, pid in enumerate(sorted({r[1] for r in records}))}
        return [(path, pid2label[pid], camid) for path, pid, camid in records]
