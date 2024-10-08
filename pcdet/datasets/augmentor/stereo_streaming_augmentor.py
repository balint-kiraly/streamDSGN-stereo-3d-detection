# data augmentor for stereo data_dict.

from functools import partial
import numpy as np
import cv2
import concurrent.futures as futures

from pcdet.utils import common_utils, box_utils
from . import augmentor_utils, stereo_streaming_sampler


class StereoStreamingAugmentor(object):
    def __init__(self, root_path, augmentor_configs, class_names, training, logger=None):
        self.root_path = root_path
        self.class_names = class_names
        self.logger = logger
        self.training = training
        self.data_augmentor_queue = []
        self.max_crop_shape = None
        self.scale_factor = None

        if augmentor_configs is not None:
            aug_config_list = augmentor_configs if isinstance(
                augmentor_configs, list) else augmentor_configs.AUG_CONFIG_LIST
            for cur_cfg in aug_config_list:
                if not isinstance(augmentor_configs, list):
                    if cur_cfg.NAME in augmentor_configs.DISABLE_AUG_LIST:
                        continue
                if cur_cfg.NAME == 'random_crop':
                    self.max_crop_shape = (
                        cur_cfg.MAX_CROP_H, cur_cfg.MAX_CROP_W)
                if cur_cfg.NAME == 'random_scale':
                    self.scale_factor = cur_cfg.SCALE_FACTOR
                if cur_cfg.NAME in ["gt_sampling"]:
                    cur_augmentor = getattr(self, cur_cfg.NAME)(config=cur_cfg)
                else:
                    cur_augmentor = partial(
                        getattr(self, cur_cfg.NAME), config=cur_cfg)
                self.data_augmentor_queue.append(cur_augmentor)

        if self.max_crop_shape is not None and self.scale_factor is not None:
            self.max_crop_shape = (int(
                self.max_crop_shape[0]*self.scale_factor), int(self.max_crop_shape[1]*self.scale_factor))

    def set_epoch(self, cur_epoch):
        self.epoch = cur_epoch
        for aug in self.data_augmentor_queue:
            if hasattr(aug, 'set_epoch'):
                aug.set_epoch(cur_epoch)

    def gt_sampling(self, config=None):
        db_sampler = stereo_streaming_sampler.StreamingSampler(
            root_path=self.root_path,
            sampler_cfg=config,
            class_names=self.class_names,
            logger=self.logger
        )
        return db_sampler

    def pre_2d_transformation(self, data_dict):
        assert 'did_3d_transformation' not in data_dict['input_data']
        assert 'gt_boxes_no3daug' not in data_dict['gt_info']
        assert 'points_no3daug' not in data_dict['input_data']

    def pre_world_transformation(self, data_dict):
        data_dict['input_data']['did_3d_transformation'] = True
        if 'gt_boxes_no3daug' not in data_dict['gt_info']:
            data_dict['gt_info']['gt_boxes_no3daug'] = data_dict['gt_info']['gt_boxes'].copy()
        if 'points_no3daug' not in data_dict['input_data']:
            data_dict['input_data']['points_no3daug'] = data_dict['input_data']['points'].copy()

    def random_crop(self, data_list, config=None):
        def process_single_frame(data_tuple):
            tag, frame_data = data_tuple[0], data_tuple[1]
            frame_valid = frame_data['frame_valid']
            if not frame_valid:
                return (tag, frame_data)

            self.pre_2d_transformation(frame_data)
            input_data, gt_info = frame_data['input_data'], frame_data['gt_info']

            old_h, old_w = input_data['image_shape'][0], input_data['image_shape'][1]
            crop_h, crop_w = min(config.MAX_CROP_H, old_h), min(
                config.MAX_CROP_W, old_w)
            assert crop_h <= old_h and crop_w <= old_w and 0 <= crop_rel_x <= 1 and 0 <= crop_rel_y <= 1

            x1 = int((old_w - crop_w) * crop_rel_x)
            y1 = int((old_h - crop_h) * crop_rel_y)
            y2 = min(config.get('MAX_CROP_H_LIMIT', 100000), y1 + crop_h)
            input_data['left_img'] = input_data['left_img'][y1: y2,
                                                            x1:x1 + crop_w]
            input_data['right_img'] = input_data['right_img'][y1: y2,
                                                              x1:x1 + crop_w]
            input_data['calib'].offset(x1, y1)
            if 'image_shape' in input_data:
                input_data['image_shape'] = input_data['left_img'].shape[:2]
            if 'gt_boxes_2d_ignored' in gt_info:
                gt_info['gt_boxes_2d_ignored'] = gt_info['gt_boxes_2d_ignored'].copy()
                gt_info['gt_boxes_2d_ignored'][:, [0, 2]] -= x1
                gt_info['gt_boxes_2d_ignored'][:, [1, 3]] -= y1
            if 'gt_annots_boxes_2d' in gt_info:
                gt_info['gt_annots_boxes_2d'] = gt_info['gt_annots_boxes_2d'].copy()
                gt_info['gt_annots_boxes_2d'][:, [0, 2]] -= x1
                gt_info['gt_annots_boxes_2d'][:, [1, 3]] -= y1

            frame_data = {
                'frame_valid': frame_valid,
                'input_data': input_data,
                'gt_info': gt_info
            }
            return (tag, frame_data)

        crop_rel_x = np.random.uniform(
            low=config.MIN_REL_X, high=config.MAX_REL_X) / 2 + 0.5
        crop_rel_y = np.random.uniform(
            low=config.MIN_REL_Y, high=config.MAX_REL_Y) / 2 + 0.5

        with futures.ThreadPoolExecutor(4) as executor:
            infos = executor.map(process_single_frame, data_list)
        re_infos = list(infos)
        return re_infos

    def random_scale(self, data_list, config=None):
        def process_single_frame(data_tuple):
            tag, frame_data = data_tuple[0], data_tuple[1]
            frame_valid = frame_data['frame_valid']
            if not frame_valid:
                return (tag, frame_data)

            self.pre_2d_transformation(frame_data)
            input_data, gt_info = frame_data['input_data'], frame_data['gt_info']
            old_h, old_w = input_data['left_img'].shape[0], input_data['left_img'].shape[1]
            new_h, new_w = int(old_h * scale_factor), int(old_w * scale_factor)
            input_data['calib'].image_scale(scale_factor)
            input_data['left_img'] = cv2.resize(
                input_data['left_img'], (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            input_data['right_img'] = cv2.resize(
                input_data['right_img'], (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            if 'image_shape' in input_data:
                input_data['image_shape'] = input_data['left_img'].shape[:2]
            if 'gt_boxes_2d_ignored' in gt_info:
                gt_info['gt_boxes_2d_ignored'] = gt_info['gt_boxes_2d_ignored'].copy()
                gt_info['gt_boxes_2d_ignored'] *= scale_factor
            if 'gt_annots_boxes_2d' in gt_info:
                gt_info['gt_annots_boxes_2d'] = gt_info['gt_annots_boxes_2d'].copy()
                gt_info['gt_annots_boxes_2d'] *= scale_factor

            frame_data = {
                'frame_valid': frame_valid,
                'input_data': input_data,
                'gt_info': gt_info
            }
            return (tag, frame_data)

        scale_factor = config['SCALE_FACTOR']
        with futures.ThreadPoolExecutor(4) as executor:
            infos = executor.map(process_single_frame, data_list)
        re_infos = list(infos)
        return re_infos

    def random_flip(self, data_list, config=None):
        if np.random.randint(2) > 0.5:
            return data_list

        def process_single_frame(data_tuple):
            tag, frame_data = data_tuple[0], data_tuple[1]
            frame_valid = frame_data['frame_valid']
            if not frame_valid:
                return (tag, frame_data)
            input_data, gt_info = frame_data['input_data'], frame_data['gt_info']

            pts_rect = input_data['calib'].lidar_pseudo_to_rect(
                input_data['points'])
            pts_rect[:, 0] *= -1
            input_points = input_data['calib'].rect_to_lidar_pseudo(pts_rect)
            input_data['points'] = input_points
            if 'completion_points' in input_data:
                completion_pts_rect = input_data['calib'].lidar_pseudo_to_rect(
                    input_data['completion_points'])
                completion_pts_rect[:, 0] *= -1
                completion_input_points = input_data['calib'].rect_to_lidar_pseudo(
                    completion_pts_rect)
                input_data['completion_points'] = completion_input_points

            input_data['left_img'], input_data['right_img'] = input_data['right_img'][:,
                                                                                      ::-1], input_data['left_img'][:, ::-1]

            # gt boxes
            if 'gt_boxes_2d_ignored' in gt_info:
                gt_info['gt_boxes_2d_ignored'] = box_utils.boxes2d_fliplr(
                    gt_info['gt_boxes_2d_ignored'], input_data['image_shape'])
            if 'gt_annots_boxes_2d' in gt_info:
                gt_info['gt_annots_boxes_2d'] = box_utils.boxes2d_fliplr(
                    gt_info['gt_annots_boxes_2d'], input_data['image_shape'])

            if 'gt_boxes' in gt_info:
                gt_boxes_camera = box_utils.boxes3d_lidar_to_kitti_camera(
                    gt_info['gt_boxes'], input_data['calib'], pseudo_lidar=True, pseduo_cam2_view=False)
                gt_boxes_camera = box_utils.boxes3d_fliplr(
                    gt_boxes_camera, cam_view=True)
                gt_boxes_lidar = box_utils.boxes3d_kitti_camera_to_lidar(
                    gt_boxes_camera, input_data['calib'], pseudo_lidar=True, pseudo_cam2_view=False)
                gt_info['gt_boxes'] = gt_boxes_lidar

            input_data['calib'].fliplr(
                image_width=input_data['image_shape'][1])

            frame_data = {
                'frame_valid': frame_valid,
                'input_data': input_data,
                'gt_info': gt_info
            }
            return (tag, frame_data)

        with futures.ThreadPoolExecutor(4) as executor:
            infos = executor.map(process_single_frame, data_list)
        re_infos = list(infos)
        return re_infos

    def filter_truncated(self, data_list, config=None):
        # reproject bboxes into image space and do filtering by truncated ratio
        area_ratio_threshold = config.AREA_RATIO_THRESH
        area_2d_ratio_threshold = config.AREA_2D_RATIO_THRESH
        gt_truncated_threshold = config.GT_TRUNCATED_THRESH

        def process_single_frame(data_tuple):
            tag, frame_data = data_tuple[0], data_tuple[1]
            frame_valid = frame_data['frame_valid']
            if not frame_valid:
                return (tag, frame_data)

            self.pre_2d_transformation(frame_data)
            input_data, gt_info = frame_data['input_data'], frame_data['gt_info']

            assert 'gt_boxes' in gt_info, 'should not call filter_truncated in test mode'

            valid_mask = gt_info['gt_boxes_mask'][gt_info['gt_boxes_mask']]
            if area_ratio_threshold is not None:
                assert area_ratio_threshold >= 0.9, 'AREA_RATIO_THRESH should be >= 0.9'
                image_shape = input_data['left_img'].shape[:2]
                calib = input_data['calib']
                gt_boxes_cam = box_utils.boxes3d_lidar_to_kitti_camera(
                    gt_info['gt_boxes'][gt_info['gt_boxes_mask']], None, pseudo_lidar=True)

                boxes2d_image, _ = box_utils.boxes3d_kitti_camera_to_imageboxes(
                    gt_boxes_cam, calib, image_shape, return_neg_z_mask=True, fix_neg_z_bug=True)
                truncated_ratio = 1 - box_utils.boxes3d_kitti_camera_inside_image_mask(
                    gt_boxes_cam, calib, image_shape, reduce=False).mean(-1)
                valid_mask &= truncated_ratio < area_ratio_threshold

            if area_2d_ratio_threshold is not None:
                assert area_2d_ratio_threshold >= 0.9, 'AREA_2D_RATIO_THRESH should be >= 0.9'
                image_shape = input_data['left_img'].shape[:2]
                boxes2d_image, no_neg_z_valids = box_utils.boxes3d_kitti_camera_to_imageboxes(
                    box_utils.boxes3d_lidar_to_kitti_camera(
                        gt_info['gt_boxes'][gt_info['gt_boxes_mask']], input_data['calib'], pseudo_lidar=True),
                    input_data['calib'],
                    return_neg_z_mask=True,
                    fix_neg_z_bug=True
                )
                boxes2d_inside = np.zeros_like(boxes2d_image)
                boxes2d_inside[:, 0] = np.clip(
                    boxes2d_image[:, 0], a_min=0, a_max=image_shape[1] - 1)
                boxes2d_inside[:, 1] = np.clip(
                    boxes2d_image[:, 1], a_min=0, a_max=image_shape[0] - 1)
                boxes2d_inside[:, 2] = np.clip(
                    boxes2d_image[:, 2], a_min=0, a_max=image_shape[1] - 1)
                boxes2d_inside[:, 3] = np.clip(
                    boxes2d_image[:, 3], a_min=0, a_max=image_shape[0] - 1)
                clip_box_area = (boxes2d_inside[:, 2] - boxes2d_inside[:, 0]) * (
                    boxes2d_inside[:, 3] - boxes2d_inside[:, 1])
                full_box_area = (
                    boxes2d_image[:, 2] - boxes2d_image[:, 0]) * (boxes2d_image[:, 3] - boxes2d_image[:, 1])
                clip_ratio = 1 - clip_box_area / full_box_area
                valid_mask &= clip_ratio < area_2d_ratio_threshold

            if gt_truncated_threshold is not None:
                gt_truncated = gt_info['gt_truncated'][gt_info['gt_boxes_mask']]
                valid_mask &= gt_truncated < gt_truncated_threshold

            cared_mask = gt_info['gt_boxes_mask'].copy()
            gt_info['gt_boxes_mask'][cared_mask] = valid_mask

            frame_data = {
                'frame_valid': frame_valid,
                'input_data': input_data,
                'gt_info': gt_info
            }
            return (tag, frame_data)

        with futures.ThreadPoolExecutor(4) as executor:
            infos = executor.map(process_single_frame, data_list)
        re_infos = list(infos)
        return re_infos

    def random_world_rotation(self, data_list, config=None):
        rot_range = config['WORLD_ROT_ANGLE']
        if not isinstance(rot_range, list):
            rot_range = [-rot_range, rot_range]
        rot_angle = np.random.uniform(rot_range[0], rot_range[1])

        def process_single_frame(data_tuple):
            tag, frame_data = data_tuple[0], data_tuple[1]
            frame_valid = frame_data['frame_valid']
            if not frame_valid:
                return (tag, frame_data)

            self.pre_world_transformation(frame_data)
            input_data, gt_info = frame_data['input_data'], frame_data['gt_info']

            gt_boxes, points, T = augmentor_utils.global_rotation(
                gt_info['gt_boxes'], input_data['points'], rot_range=rot_angle,
                return_trans_mat=True
            )

            gt_info['gt_boxes'] = gt_boxes
            # points_for2d is fixed since images do not support rotation
            input_data['points'] = points
            # note that random T is the inverse transformation matrix
            input_data['random_T'] = np.matmul(
                input_data.get('random_T', np.eye(4)), T)

            frame_data = {
                'frame_valid': frame_valid,
                'input_data': input_data,
                'gt_info': gt_info
            }
            return (tag, frame_data)

        with futures.ThreadPoolExecutor(4) as executor:
            infos = executor.map(process_single_frame, data_list)
        re_infos = list(infos)
        return re_infos

    def random_world_scaling(self, data_list, config=None):
        scale_range = config['WORLD_SCALE_RANGE']
        if scale_range[1] - scale_range[0] < 1e-3:
            return data_list
        scale_factor = np.random.uniform(scale_range[0], scale_range[1])

        def process_single_frame(data_tuple):
            tag, frame_data = data_tuple[0], data_tuple[1]
            frame_valid = frame_data['frame_valid']
            if not frame_valid:
                return (tag, frame_data)

            self.pre_world_transformation(frame_data)
            input_data, gt_info = frame_data['input_data'], frame_data['gt_info']

            gt_boxes, points, T = augmentor_utils.global_scaling(
                gt_info['gt_boxes'], input_data['points'], scale_factor,
                return_trans_mat=True
            )
            gt_info['gt_boxes'] = gt_boxes
            input_data['points'] = points
            # note that random T is the inverse transformation matrix
            input_data['random_T'] = np.matmul(
                input_data.get('random_T', np.eye(4)), T)

            frame_data = {
                'frame_valid': frame_valid,
                'input_data': input_data,
                'gt_info': gt_info
            }
            return (tag, frame_data)

        with futures.ThreadPoolExecutor(4) as executor:
            infos = executor.map(process_single_frame, data_list)
        re_infos = list(infos)
        return re_infos

    def forward(self, batch_dict):
        # [(prev2, value), ..., (next, value)]
        batch_list = [(k, v) for k, v in batch_dict.items()]
        for cur_augmentor in self.data_augmentor_queue:
            batch_list = cur_augmentor(data_list=batch_list)
  
        # Check if data augmentation is correct
        # import matplotlib.image as mpimg
        # for tag, frame_data in batch_list:
        #     left_img = frame_data['input_data']['left_img'].astype(np.uint8)
        #     right_img = frame_data['input_data']['left_img'].astype(np.uint8)
        #     points = frame_data['input_data']['points']
        #     ori_points = frame_data['input_data']['ori_points']
        #     gt_boxes = frame_data['gt_info']['gt_boxes']
        #     print(left_img.shape, right_img.shape, ori_points.shape, points.shape)
        #     mpimg.imsave('debug/aug_show/left_{}.png'.format(tag), left_img)
        #     mpimg.imsave('debug/aug_show/right_{}.png'.format(tag), right_img)
        #     points.tofile('debug/aug_show/points_{}.bin'.format(tag))
        #     ori_points.tofile('debug/aug_show/ori_points_{}.bin'.format(tag))
        #     np.save('debug/aug_show/gt_boxes_{}.npy'.format(tag), gt_boxes)

        for (_, batch_data) in batch_list:
            if not batch_data['frame_valid']:
                continue
            if 'gt_boxes' in batch_data['gt_info']:
                batch_data['gt_info']['gt_boxes'][:, 6] = common_utils.limit_period(
                    batch_data['gt_info']['gt_boxes'][:, 6], offset=0.5, period=2 * np.pi)
            if 'road_plane' in batch_data['input_data']:
                batch_data['input_data'].pop('road_plane')
            if 'gt_boxes_mask' in batch_data['gt_info']:
                gt_boxes_mask = batch_data['gt_info']['gt_boxes_mask']
                for key in ['gt_names', 'gt_boxes', 'gt_truncated', 'gt_occluded', 'gt_difficulty', 'gt_index', 'gt_boxes_no3daug', 'object_id']:
                    if key in batch_data['gt_info']:
                        batch_data['gt_info'][key] = batch_data['gt_info'][key][gt_boxes_mask]
                batch_data['gt_info'].pop('gt_boxes_mask')
            # if 'calib' in batch_data['input_data']:
            #     batch_data['input_data'].pop('calib')
            # if 'calib_ori' in batch_data['input_data']:
            #     batch_data['input_data'].pop('calib_ori')

        # batch_dict = {k: v for (k, v) in batch_list}
        # import pickle
        # with open('batch_dict.pkl', 'wb') as f:
        #     pickle.dump(batch_dict, f)
        # exit()


        return batch_dict
