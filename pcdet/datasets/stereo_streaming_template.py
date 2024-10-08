from collections import defaultdict
from pathlib import Path
import numpy as np
import json
import torch.utils.data as torch_data
import concurrent.futures as futures

from pcdet.utils import common_utils, box_utils, depth_map_utils
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
from .augmentor.stereo_streaming_augmentor import StereoStreamingAugmentor
from .processor.data_processor import DataProcessor
from .processor.point_feature_encoder import PointFeatureEncoder
from pcdet.utils.calibration_kitti import Calibration


class Empirical():
    def __init__(self, inf_time_path, logger=None):
        with open(inf_time_path, 'r') as f:
            # a list of model inference time
            model_time_samples = json.load(f)
        model_time_samples = [float(x) for x in model_time_samples]
        self.samples = np.array(model_time_samples)

        if logger is not None:
            logger.info('mean model time: {}'.format(self.mean()))
            logger.info('std model time: {}'.format(self.std()))
            logger.info('draw model time: {}'.format(self.draw()))
            logger.info('max model time: {}'.format(self.max()))
            logger.info('min model time: {}'.format(self.min()))

    def draw(self):
        this_time = np.random.choice(self.samples)
        return this_time

    def mean(self):
        return self.samples.mean()

    def std(self):
        return self.samples.std(ddof=1)

    def min(self):
        return self.samples.min()

    def max(self):
        return self.samples.max()


class StereoStreamingTemplate(torch_data.Dataset):
    def __init__(self, dataset_cfg=None, class_names=None, training=True, root_path=None, logger=None):
        super().__init__()
        self.dataset_cfg = dataset_cfg
        self.training = training
        self.class_names = class_names
        self.logger = logger
        self.root_path = root_path if root_path is not None else Path(
            self.dataset_cfg.DATA_PATH)
        self.prepare_data_path = self.root_path / dataset_cfg.PREPARE_PATH
        self.logger = logger
        if self.dataset_cfg is None or class_names is None:
            return

        self.annos_frequency = self.dataset_cfg.get('ANNOS_FREQUENCY', 10)
        self.infer_time_path = self.dataset_cfg.get('INFER_TIME_PATH', None)
        self.box3d_supervision = self.dataset_cfg.get(
            'BOX3D_SUPERVISION', 'token')
        self.depth_supervision = self.dataset_cfg.get(
            'DEPTH_SUPERVISION', 'token')
        if self.mode == 'test' and self.infer_time_path is not None:
            if Path(self.infer_time_path).exists():
                self.empirical = Empirical(self.infer_time_path, self.logger)
            else:
                self.empirical = None
                self.logger.info(
                    'INFER_TIME_PATH is not exists, this will result in the inability to stream metric!')
        else:
            self.logger.info(
                'INFER_TIME_PATH is None, this will result in the inability to stream metric!')
            self.empirical = None

        self.complete_and_predict_depth = getattr(
            self.dataset_cfg, 'complete_and_predict_depth', False)
        self.voxel_occupancy_range = getattr(
            self.dataset_cfg, 'voxel_occupancy_range', False)
        self.voxel_occupancy_range_limit = getattr(
            self.dataset_cfg, 'voxel_occupancy_range_limit', False)

        self.point_cloud_range = np.array(
            self.dataset_cfg.POINT_CLOUD_RANGE, dtype=np.float32)
        self.stereo_point_cloud_range = np.array(
            getattr(self.dataset_cfg, 'STEREO_POINT_CLOUD_RANGE',
                    self.dataset_cfg.POINT_CLOUD_RANGE),
            dtype=np.float32)
        self.voxel_size = self.dataset_cfg.VOXEL_SIZE
        grid_size = (
            self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(self.voxel_size)
        self.grid_size = np.round(grid_size).astype(np.int64)

        if self.dataset_cfg.get("STEREO_VOXEL_SIZE", None):
            self.stereo_voxel_size = self.dataset_cfg.STEREO_VOXEL_SIZE
            stereo_grid_size = (
                self.stereo_point_cloud_range[3:6] - self.stereo_point_cloud_range[0:3]) / np.array(self.stereo_voxel_size)
            self.stereo_grid_size = np.round(stereo_grid_size).astype(np.int64)

        if self.training:
            self.data_augmentor = StereoStreamingAugmentor(
                self.root_path, self.dataset_cfg.TRAIN_DATA_AUGMENTOR, self.class_names, training=self.training, logger=self.logger
            )
        else:
            if getattr(self.dataset_cfg, 'TEST_DATA_AUGMENTOR', None) is not None:
                self.data_augmentor = StereoStreamingAugmentor(
                    self.root_path, self.dataset_cfg.TEST_DATA_AUGMENTOR, self.class_names, training=self.training, logger=self.logger
                )
                # logger.warn('using data augmentor in test mode')
            else:
                self.data_augmentor = None
        self.max_crop_shape = getattr(
            self.data_augmentor, 'max_crop_shape', None)

        if self.dataset_cfg.get('POINT_FEATURE_ENCODING'):
            self.point_feature_encoder = PointFeatureEncoder(
                self.dataset_cfg.POINT_FEATURE_ENCODING,
                point_cloud_range=self.point_cloud_range
            )
        else:
            self.point_feature_encoder = None
        if self.dataset_cfg.get('DATA_PROCESSOR'):
            self.data_processor = DataProcessor(
                self.dataset_cfg.DATA_PROCESSOR, point_cloud_range=self.point_cloud_range, training=self.training
            )
        else:
            self.data_processor = None
        self.total_epochs = 0
        self._merge_all_iters_to_one_epoch = False

    @property
    def mode(self):
        return 'train' if self.training else 'test'

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['logger']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)

    def set_epoch(self, cur_epoch):
        self.epoch = cur_epoch
        if hasattr(self, 'data_augmentor'):
            self.data_augmentor.set_epoch(cur_epoch)

    @staticmethod
    def generate_prediction_dicts(batch_dict, pred_dicts, class_names, output_path=None):
        raise NotImplementedError

    def merge_all_iters_to_one_epoch(self, merge=True, epochs=None):
        if merge:
            self._merge_all_iters_to_one_epoch = True
            self.total_epochs = epochs
        else:
            self._merge_all_iters_to_one_epoch = False

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        """
        To support a custom dataset, implement this function to load the raw data (and labels), then transform them to
        the unified normative coordinate and call the function self.prepare_data() to process the data and send them
        to the model.

        Args:
            index:

        Returns:

        """
        raise NotImplementedError

    def prepare_data(self, data_dict):
        """
        Args:
            data_dict:
                prev2:
                prev:
                token:
                next:
                    frame_valid:
                    input_data:
                        points: (N, 3 + C_in)
                    gt_info:
                        gt_valid: True, bool
                        gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                        gt_names: optional, (N), string
                ...

        Returns:
            data_dict:
                frame_id: string
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                use_lead_xyz: bool
                voxels: optional (num_voxels, max_points_per_voxel, 3 + C)
                voxel_coords: optional (num_voxels, 3)
                voxel_num_points: optional (num_voxels)
                ...
        """
        if self.training:
            # TODO: in case using data augmentor, please pay attention to the coordinate
            data_dict = self.data_augmentor.forward(data_dict)
            if len(data_dict[self.box3d_supervision]['gt_info']['gt_boxes']) == 0:
                new_index = np.random.randint(self.__len__())
                return self.__getitem__(new_index)
            if len(data_dict[self.box3d_supervision]['input_data']['points']) < 200:
                new_index = np.random.randint(self.__len__())
                return self.__getitem__(new_index)

        elif (not self.training) and self.data_augmentor:
            # only do some basic image scaling and cropping
            data_dict = self.data_augmentor.forward(data_dict)

        def process_single_frame(data_tuple):
            tag, frame_data = data_tuple[0], data_tuple[1]
            if not frame_data['frame_valid']:
                return data_tuple

            frame_valid, input_data, gt_info = frame_data[
                'frame_valid'], frame_data['input_data'], frame_data['gt_info']
            if gt_info.get('gt_boxes', None) is not None:
                if 'gt_boxes_no3daug' not in gt_info:
                    gt_info['gt_boxes_no3daug'] = gt_info['gt_boxes'].copy()

                selected = common_utils.keep_arrays_by_name(
                    gt_info['gt_names'], self.class_names)
                if len(selected) != len(gt_info['gt_names']):
                    for key in ['gt_names', 'gt_boxes', 'gt_truncated', 'gt_occluded', 'gt_difficulty', 'gt_index', 'gt_boxes_no3daug', 'object_id']:
                        gt_info[key] = gt_info[key][selected]
                gt_classes = np.array([self.class_names.index(
                    n) + 1 for n in gt_info['gt_names']], dtype=np.int32)
                gt_info['gt_boxes'] = np.concatenate(
                    (gt_info['gt_boxes'], gt_classes.reshape(-1, 1).astype(np.float32)), axis=1)
                gt_info['gt_boxes_no3daug'] = np.concatenate(
                    (gt_info['gt_boxes_no3daug'], gt_classes.reshape(-1, 1).astype(np.float32)), axis=1)

            # convert to 2d gt boxes
            image_shape = input_data['left_img'].shape[:2]
            if 'gt_boxes' in gt_info:
                gt_boxes_no3daug = gt_info['gt_boxes_no3daug']
                gt_boxes_no3daug_cam = box_utils.boxes3d_lidar_to_kitti_camera(
                    gt_boxes_no3daug, None, pseudo_lidar=True)
                gt_info['gt_boxes_2d'] = box_utils.boxes3d_kitti_camera_to_imageboxes(
                    gt_boxes_no3daug_cam, input_data['calib'], image_shape, fix_neg_z_bug=True)
                gt_info['gt_centers_2d'] = box_utils.boxes3d_kitti_camera_to_imagecenters(
                    gt_boxes_no3daug_cam, input_data['calib'], image_shape)
                gt_info['gt_boxes_2d'] = np.concatenate(
                    [gt_info['gt_boxes_2d'], gt_classes.reshape(-1, 1).astype(np.float32)], axis=1)

            if self.point_feature_encoder:
                input_data = self.point_feature_encoder.forward(input_data)
            # TODO
            if self.data_processor:
                processed_data = self.data_processor.forward(
                    data_dict={'input_data': input_data, 'gt_info': gt_info})
                input_data, gt_info = processed_data['input_data'], processed_data['gt_info']

            # generate depth gt image
            # uses no 3d augs
            points_no3daug = input_data.get(
                'points_no3daug', input_data['points'])
            completion_points = input_data.get(
                'completion_points', points_no3daug)
            rect_points = Calibration.lidar_pseudo_to_rect(
                completion_points[:, :3])
            input_data['depth_gt_img'] = depth_map_utils.points_to_depth_map(
                rect_points, image_shape, input_data['calib'])
            if self.complete_and_predict_depth:
                rect_input_points = Calibration.lidar_pseudo_to_rect(
                    points_no3daug[:, :3])
                input_data['input_depth_gt_img'] = depth_map_utils.points_to_depth_map(
                    rect_input_points, image_shape, input_data['calib'])
            if 'gt_boxes_no3daug' in gt_info:
                input_data['depth_fgmask_img'] = roiaware_pool3d_utils.depth_map_in_boxes_cpu(
                    input_data['depth_gt_img'], gt_info['gt_boxes_no3daug'][:, :7], input_data['calib'], expand_distance=0., expand_ratio=1.0)

            if 'random_T' in input_data:
                input_data['inv_random_T'] = np.linalg.inv(
                    input_data['random_T'])

            input_data.pop('points_no3daug', None)
            input_data.pop('did_3d_transformation', None)
            input_data.pop('road_plane', None)
            input_data.pop('completion_points', None)
            frame_data = {
                'frame_valid': frame_valid,
                'input_data': input_data,
                'gt_info': gt_info
            }
            return (tag, frame_data)

        data_list = [(k, v) for k, v in data_dict.items()]
        with futures.ThreadPoolExecutor(4) as executor:
            infos = executor.map(process_single_frame, data_list)
        re_infos = list(infos)
        data_dict = {k: v for (k, v) in re_infos}
        return data_dict

    @staticmethod
    def collate_batch(batch_list, _unused=False):
        # len of batch_list: batchsize
        # batch_list: [{data_dict},{data_dict},...] data_dict-> keys:{token, prev2, prev, next}
        assert len(batch_list) == 1, 'Only supports one batchsize!'
        if batch_list[0] is None:
            return None

        data_dict = {}
        for tag, frame_data in batch_list[0].items():
            # frame tag
            data_dict[tag] = {}
            for key, val in frame_data.items():
                # frame_valid, input_data, gt_info
                if isinstance(val, dict):
                    data_dict[tag][key] = {}
                    for sub_key, sub_val in val.items():
                        # point, voxel, ...
                        data_dict[tag][key][sub_key] = []
                else:
                    data_dict[tag][key] = []

        # data_dict: -> data_dict['token]['input_data'][0] (list)
        for cur_sample in batch_list:
            # batch size
            for tag, frame_data in cur_sample.items():
                for key, val in frame_data.items():
                    if isinstance(val, dict):
                        for sub_key, sub_val in val.items():
                            data_dict[tag][key][sub_key].append(sub_val)
                    else:
                        data_dict[tag][key].append(val)
        batch_size = len(batch_list)

        def process_single_frame(data_tuple):
            ret = {}
            tag, frame_data = data_tuple[0], data_tuple[1]
            frame_valid = frame_data['frame_valid']

            # The first frame has no history information
            if not frame_valid[0]:
                return (tag, None)
            input_data, gt_info = frame_data['input_data'], frame_data['gt_info']

            for key, val in input_data.items():
                if key in ['voxels', 'voxel_num_points']:
                    ret[key] = np.concatenate(val, axis=0)
                elif key in ['points', 'voxel_coords']:
                    coors = []
                    for i, coor in enumerate(val):
                        coor_pad = np.pad(
                            coor, ((0, 0), (1, 0)), mode='constant', constant_values=i)
                        coors.append(coor_pad)
                    ret[key] = np.concatenate(coors, axis=0)
                elif key in ['left_img', 'right_img', 'depth_gt_img', 'depth_fgmask_img', 'input_depth_gt_img']:
                    if key in ['depth_gt_img', 'depth_fgmask_img', 'input_depth_gt_img']:
                        val = [np.expand_dims(x, -1) for x in val]
                    max_h = np.max([x.shape[0] for x in val])
                    max_w = np.max([x.shape[1] for x in val])
                    pad_h = (max_h - 1) // 32 * 32 + 32 - max_h
                    pad_w = (max_w - 1) // 32 * 32 + 32 - max_w
                    assert pad_h < 32 and pad_w < 32
                    padded_imgs = []
                    for i, img in enumerate(val):
                        if key in ['left_img', 'right_img']:
                            mean = np.array(
                                [0.485, 0.456, 0.406], dtype=np.float32)
                            std = np.array(
                                [0.229, 0.224, 0.225], dtype=np.float32)
                            img = (img.astype(np.float32) / 255 - mean) / std
                        img = np.pad(img, ((0, pad_h + max_h - img.shape[0]), (0, pad_w + max_w - img.shape[1]),
                                           (0, 0)), mode='constant')
                        padded_imgs.append(img)
                    ret[key] = np.stack(
                        padded_imgs, axis=0).transpose(0, 3, 1, 2)
                elif key in ['scene', 'this_sample_idx', 'prev2_sample_idx', 'prev_sample_idx', 'next_sample_idx', 'image_shape', 'random_T', 'inv_random_T']:
                    ret[key] = np.stack(val, axis=0)
                elif key in ['voxel_size']:
                    ret[key] = val[0]
                elif key in ['calib', 'calib_ori', 'use_lead_xyz']:
                    ret[key] = val

            for key, val in gt_info.items():
                if key in ['gt_boxes', 'gt_boxes_no3daug', 'gt_boxes_2d', 'gt_centers_2d', 'gt_annots_boxes_2d', 'gt_boxes_2d_ignored', 'gt_boxes_camera']:
                    max_gt = max([len(x) for x in val])
                    batch_gt_boxes3d = np.zeros(
                        (batch_size, max_gt, val[0].shape[-1]), dtype=np.float32)
                    for k in range(batch_size):
                        batch_gt_boxes3d[k, :val[k].__len__(), :] = val[k]
                    ret[key] = batch_gt_boxes3d
                elif key in ['object_id', 'gt_boxes_mask']:  # gt_boxes_mask
                    ret[key] = val
                elif key in ['gt_names', 'gt_truncated', 'gt_occluded', 'gt_difficulty', 'gt_index']:
                    ret[key] = [np.array(x) for x in val]
                elif key in ['voxel_size']:
                    ret[key] = val[0]
            ret['batch_size'] = batch_size
            return (tag, ret)

        data_list = [(k, v) for k, v in data_dict.items()]

        with futures.ThreadPoolExecutor(4) as executor:
            infos = executor.map(process_single_frame, data_list)
        re_infos = list(infos)

        ret_dict = {x[0]: x[1] for x in re_infos}
        return ret_dict
