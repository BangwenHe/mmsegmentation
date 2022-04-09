import os
import os.path as osp

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class SelfMadeGANDataset(CustomDataset):
    """
    自己拍摄的用于训练GAN的数据集
    """
    CLASSES = ('background', 'person')

    PALETTE = [[0, 0, 0], [128, 0, 0]]

    def __init__(self, **kwargs):
        super(SelfMadeGANDataset, self).__init__(
            img_suffix='.jpg', seg_map_suffix='.png', **kwargs)
        assert osp.exists(self.img_dir)

