from __future__ import absolute_import

from .base_dataset import BaseDataset, BaseImageDataset
from .preprocessor import Preprocessor


class IterLoader:
    def __init__(self, loader, length=None):
        self.loader = loader
        self.length = length
        self.iter = None

    def __len__(self):
        if self.length is not None:
            return self.length

        return len(self.loader)

    def new_epoch(self):
        self.iter = iter(self.loader)

    def next(self):
        try:
            return next(self.iter)
        except StopIteration:
            self.iter = iter(self.loader)
            try:
                return next(self.iter)
            except StopIteration as error:
                raise RuntimeError(
                    'IterLoader source produced no batches. Check the '
                    'pseudo-labeled sample count, sampler size, batch size, '
                    'and drop_last setting.') from error
