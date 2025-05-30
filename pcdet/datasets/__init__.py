import torch
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler as _DistributedSampler

from .kitti.lidar_kitti_dataset import LiDARKittiDataset
from .kitti.stereo_kitti_dataset import StereoKittiDataset
from .kitti.mono_kitti_dataset import MonoKittiDataset
from .kitti_streaming.lidar_kitti_streaming import LiDARKittiStreaming
from .kitti_streaming.stereo_kitti_streaming import StereoKittiStreaming, StereoKittiStreamingInfer
from pcdet.utils import common_utils

__all__ = {
    'LiDARKittiDataset': LiDARKittiDataset,
    'StereoKittiDataset': StereoKittiDataset,
    'MonoKittiDataset': MonoKittiDataset,
    'LiDARKittiStreaming': LiDARKittiStreaming,
    'StereoKittiStreaming': StereoKittiStreaming,
    'StereoKittiStreamingInfer': StereoKittiStreamingInfer,
}


class DistributedSampler(_DistributedSampler):

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank)
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)


def build_dataloader(dataset_cfg, class_names, batch_size, dist, root_path=None, workers=4,
                     logger=None, training=True, merge_all_iters_to_one_epoch=False, total_epochs=0):

    dataset = __all__[dataset_cfg.DATASET](
        dataset_cfg=dataset_cfg,
        class_names=class_names,
        root_path=root_path,
        training=training,
        logger=logger,
    )

    if merge_all_iters_to_one_epoch:
        assert hasattr(dataset, 'merge_all_iters_to_one_epoch')
        dataset.merge_all_iters_to_one_epoch(merge=True, epochs=total_epochs)

    if dist:
        if training:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            rank, world_size = common_utils.get_dist_info()
            sampler = DistributedSampler(dataset, world_size, rank, shuffle=False)
    else:
        sampler = None
    dataloader = DataLoader(
        dataset, batch_size=batch_size, pin_memory=True, num_workers=workers,
        shuffle=(sampler is None) and training, collate_fn=dataset.collate_batch,
        drop_last=training, sampler=sampler, timeout=0
    )

    return dataset, dataloader, sampler
