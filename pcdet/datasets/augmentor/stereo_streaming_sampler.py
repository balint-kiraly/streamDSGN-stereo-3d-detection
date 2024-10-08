# Modified from OpenPCDet. https://github.com/open-mmlab/OpenPCDet
# DataBaseSampler is used for cut-and-paste augmentation for LiDAR Point clouds.

import pickle

from pcdet.utils import box_utils
from pcdet.ops.iou3d_nms import iou3d_nms_utils
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils

from skimage import io
import cv2
import torch
import copy

import numpy as np


def warp(img, bbox, target_bbox):
    bbox = bbox.flatten()
    center = tuple(bbox.reshape(2, 2).mean(axis=0))
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    angle = 0.
    rect = center, (w, h), angle
    box = cv2.boxPoints(rect)

    target_bbox = target_bbox.flatten()
    center = tuple(target_bbox.reshape(2, 2).mean(axis=0))
    w, h = target_bbox[2] - target_bbox[0], target_bbox[3] - target_bbox[1]
    angle = 0.
    rect = center, (w, h), angle
    target_box = cv2.boxPoints(rect)

    m = cv2.getPerspectiveTransform(box, target_box)
    warped = cv2.warpPerspective(img, m, (np.ceil(target_bbox[2]).astype(
        int), np.ceil(target_bbox[3]).astype(int)), flags=cv2.INTER_LINEAR)
    return warped


def query_bbox_overlaps(boxes, query_boxes):
    if isinstance(boxes, np.ndarray):
        boxes = torch.from_numpy(boxes)
        query_boxes = torch.from_numpy(query_boxes)
        # If input is ndarray, turn the overlaps back to ndarray when return
        def out_fn(x): return x.numpy()
    else:
        def out_fn(x): return x

    box_areas = (boxes[:, 2] - boxes[:, 0] + 1) * \
        (boxes[:, 3] - boxes[:, 1] + 1)
    query_areas = (query_boxes[:, 2] - query_boxes[:, 0] + 1) * \
        (query_boxes[:, 3] - query_boxes[:, 1] + 1)

    iw = (torch.min(boxes[:, 2:3], query_boxes[:, 2:3].t(
    )) - torch.max(boxes[:, 0:1], query_boxes[:, 0:1].t()) + 1).clamp(min=0)
    ih = (torch.min(boxes[:, 3:4], query_boxes[:, 3:4].t(
    )) - torch.max(boxes[:, 1:2], query_boxes[:, 1:2].t()) + 1).clamp(min=0)
    # ua = box_areas.view(-1, 1) + query_areas.view(1, -1) - iw * ih
    # return how much overlap of the query boxes instead of full mask
    overlaps = iw * ih / query_areas.view(1, -1)
    return out_fn(overlaps)


class StreamingSampler(object):
    def __init__(self, root_path, sampler_cfg, class_names, logger=None):
        self.root_path = root_path
        self.class_names = class_names
        self.sampler_cfg = sampler_cfg
        self.logger = logger

        self.db_infos = {}
        for class_name in class_names:
            self.db_infos[class_name] = []

        for db_info_path in sampler_cfg.DB_INFO_PATH:
            db_info_path = self.root_path.resolve() / db_info_path
            with open(str(db_info_path), 'rb') as f:
                infos = pickle.load(f)
                [self.db_infos[cur_class].extend(
                    infos[cur_class]) for cur_class in class_names]

        db_info_frameid_path = self.root_path.resolve() / sampler_cfg.DB_WITH_FRAME_ID
        with open(str(db_info_frameid_path), 'rb') as f:
            self.db_infos_with_frameid = pickle.load(f)

        '''
            ## Note that not every frame has a label, you should check it before using it.
            db_info:
                scene:
                    sample_idx:
                        Car: 
                            len_db_info: int
                            db_info: [{name, path, box3d_lidar}, ..., {}]
                        Pedestrian:
                        Cyclist:
        '''

        # self.db_infos -> Car, Pedestrian, Cyclist
        for func_name, val in sampler_cfg.PREPARE.items():
            self.db_infos = getattr(self, func_name)(self.db_infos, val)

        # for sample data
        self.sample_groups = {}
        self.sample_class_num = {}
        self.limit_whole_scene = sampler_cfg.get(
            'LIMIT_WHOLE_SCENE', False)  # True
        self.filter_occlusion_overlap = sampler_cfg.get(
            'filter_occlusion_overlap', 1.)
        self.far_to_near = getattr(self.sampler_cfg, 'far_to_near', False)
        self.stop_epoch = getattr(self.sampler_cfg, 'stop_epoch', None)

        self.frame_stride = self.sampler_cfg.get('FRAME_STRIDE', 1)
        self.all_sample_tag = self.sampler_cfg.get(
            'ALL_SAMPLE_TAG', {'token': 0})

        self.box3d_supervision = self.sampler_cfg.get(
            'BOX3D_SUPERVISION', 'token')

        for x in sampler_cfg.SAMPLE_GROUPS:
            class_name, sample_num = x.split(':')
            if class_name not in class_names:
                continue
            self.sample_class_num[class_name] = sample_num
            self.sample_groups[class_name] = {
                'sample_num': sample_num,
                'pointer': len(self.db_infos[class_name]),
                'indices': np.arange(len(self.db_infos[class_name])),
            }

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['logger']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)

    def set_epoch(self, cur_epoch):
        self.epoch = cur_epoch

    def filter_by_difficulty(self, db_infos, removed_difficulty):
        new_db_infos = {}
        for key, dinfos in db_infos.items():

            new_db_infos[key] = [
                info for info in dinfos
                if info['difficulty'] not in removed_difficulty
            ]
            if self.logger is not None:
                self.logger.info('Database filter by difficulty %s: %d => %d' % (
                    key, len(dinfos), len(new_db_infos[key])))
        return new_db_infos

    def filter_by_min_points(self, db_infos, min_gt_points_list):
        for name_num in min_gt_points_list:
            name, min_num = name_num.split(':')
            min_num = int(min_num)
            if min_num > 0 and name in db_infos.keys():
                filtered_infos = []
                for info in db_infos[name]:
                    if info['num_points_in_gt'] >= min_num:
                        filtered_infos.append(info)

                if self.logger is not None:
                    self.logger.info('Database filter by min points %s: %d => %d' %
                                     (name, len(db_infos[name]), len(filtered_infos)))
                db_infos[name] = filtered_infos

        return db_infos

    def sample_with_fixed_number(self, class_name, sample_group):
        """
        Args:
            class_name:
            sample_group:  # dict:
        Returns:

        """
        # sample_num = cfg(7) - num_gt
        sample_num, pointer, indices = int(
            sample_group['sample_num']), sample_group['pointer'], sample_group['indices']
        if pointer >= len(self.db_infos[class_name]):
            indices = np.random.permutation(
                len(self.db_infos[class_name]))  # permutation 排列
            pointer = 0
        # pointer = 500
        sampled_dict = [self.db_infos[class_name][idx]
                        for idx in indices[pointer: pointer + sample_num]]
        pointer += sample_num  # 更新pointer是为了不重复采样？indices采样编号列表，pointer采样指示器

        sample_group['pointer'] = pointer
        sample_group['indices'] = indices
        return sampled_dict

    def remove_duplicate_ids(self, all_sampled_dict):
        """
        Args:
            all_sampled_dict: dict = {prev2: sampled_dict, ..., next: sampled_dict}
        Returns:

        """
        nested_dict = {k: [x['this_obj_id'] for x in v]
                       for k, v in all_sampled_dict.items()}
        ret_sampled_dict = {}
        for k, v in nested_dict.items():
            mask = [v.count(item) == 1 for item in v]
            filtered_sampled_list = [x for x, y in zip(
                all_sampled_dict[k], mask) if y]
            ret_sampled_dict[k] = filtered_sampled_list
        return ret_sampled_dict

    def sample_with_all_frame_tag(self, sampled_dict):
        """
        Args:
            sampled_dict: list = [{single_simple}, ...],
        Returns:
            all_sampled_dict: dict = {prev2: sampled_dict, ..., next: sampled_dict}
        """
        all_sampled_dict = {}
        for tag in self.all_sample_tag.keys():
            all_sampled_dict[tag] = []

        for single_simple in sampled_dict:
            scene = single_simple['scene']
            sample_idx = single_simple['sample_idx']
            name = single_simple['name']  # Car, ...
            this_obj_id = single_simple['this_obj_id']
            obj_valid = []
            tmp_sampled_dict = {}

            for tag, factor in self.all_sample_tag.items():
                tag_simple_idx = f'{int(sample_idx) + factor * self.frame_stride:06d}'
                tag_samples = self.db_infos_with_frameid[scene].get(
                    tag_simple_idx, None)
                if tag_samples is None:
                    obj_valid.append(False)
                    continue
                tag_name_samples = self.db_infos_with_frameid[scene][tag_simple_idx].get(
                    name, None)
                if tag_name_samples is None:
                    obj_valid.append(False)
                    continue
                for tag_name_obj_samples in tag_name_samples['db_info']:
                    if tag_name_obj_samples['this_obj_id'] == this_obj_id:
                        tmp_sampled_dict[tag] = tag_name_obj_samples
                if tmp_sampled_dict.get(tag, None):
                    obj_valid.append(True)
                else:
                    obj_valid.append(False)
            # We only need objects that exist continuously
            if all(obj_valid):
                for tag in all_sampled_dict.keys():
                    all_sampled_dict[tag].append(tmp_sampled_dict[tag])
            else:
                tmp_sampled_dict.clear()

        all_sampled_dict = self.remove_duplicate_ids(all_sampled_dict)
        # all_id_dict = {k:[(x['scene'], x['sample_idx'], x['this_obj_id']) for x in v] for k,v in all_sampled_dict.items()}
        # for k, v in all_id_dict.items():
        #     print(v)
        return all_sampled_dict

    def get_valid_mask(self, existed_boxes, sampled_boxes):
        iou1 = iou3d_nms_utils.boxes_bev_iou_cpu(
            sampled_boxes[:, 0:7], existed_boxes[:, 0:7])
        iou2 = iou3d_nms_utils.boxes_bev_iou_cpu(
            sampled_boxes[:, 0:7], sampled_boxes[:, 0:7])
        iou2[range(sampled_boxes.shape[0]), range(
            sampled_boxes.shape[0])] = 0  # ingore self iou
        iou1 = iou1 if iou1.shape[1] > 0 else iou2
        valid_mask = ((iou1.max(axis=1) + iou2.max(axis=1))
                      == 0)  # filter overlaps bboxes
        return valid_mask

    def check_sample_valid(self, all_sampled_dict, exist_boxes_dict):
        """
        Args:
            all_sampled_dict: dict = {prev2: sampled_dict, ..., next: sampled_dict}
            exist_boxes_dict: dict = {prev2: gt_boxes, ..., next: gt_boxes}
        Returns:

        """
        for tag in exist_boxes_dict.keys():
            # When the number of samples is empty, skip this round of data augmentation.
            if len(all_sampled_dict[tag]) == 0:
                print(f'all_sampled_dict[{tag}] is empty !!!!!')
                valid_sampled_dict = {k: [] for k in exist_boxes_dict.keys()}
                sampled_boxes = np.array([], dtype=np.float32).reshape(-1, 7)
                valid_sampled_boxes_dict = {
                    k: sampled_boxes for k in exist_boxes_dict.keys()}
                return valid_sampled_dict, valid_sampled_boxes_dict

        valid_mask_list = []
        sampled_boxes_dict = {}
        for tag, gt_boxes in exist_boxes_dict.items():
            # get sampled boxes
            sampled_boxes = np.stack(
                [x['box3d_lidar'] for x in all_sampled_dict[tag]], axis=0).astype(np.float32)
            if self.sampler_cfg.get('DATABASE_WITH_FAKELIDAR', False):
                sampled_boxes = box_utils.boxes3d_kitti_fakelidar_to_lidar(
                    sampled_boxes)
            sampled_boxes_dict[tag] = sampled_boxes

            valid_mask = self.get_valid_mask(gt_boxes, sampled_boxes)
            valid_mask_list.append(valid_mask)

        # We must ensure that there is no overlap in all consecutive frames
        all_valid_mask = np.logical_and.reduce(valid_mask_list).nonzero()[0]
        valid_sampled_dict = {k: [v[x] for x in all_valid_mask]
                              for k, v in all_sampled_dict.items()}
        valid_sampled_boxes_dict = {k: v[all_valid_mask]
                                    for k, v in sampled_boxes_dict.items()}
        # all_id_dict = {k:[(x['scene'], x['sample_idx'], x['this_obj_id']) for x in v] for k,v in valid_sampled_dict.items()}
        return valid_sampled_dict, valid_sampled_boxes_dict

    @staticmethod
    def put_boxes_on_road_planes(gt_boxes, road_planes, calib):
        """
        Only validate in KITTIDataset
        Args:
            gt_boxes: (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
            road_planes: [a, b, c, d]
            calib:

        Returns:
        """
        a, b, c, d = road_planes
        center_cam = calib.lidar_pseudo_to_rect(gt_boxes[:, 0:3])
        # height at the point [x, 0, z], direction is upward [a, b, c]~=[0, -1, 1]
        cur_height_cam = (-d - a * center_cam[:, 0] - c * center_cam[:, 2]) / b
        # set the height of the box to be zero plane
        center_cam[:, 1] = cur_height_cam
        cur_lidar_height = calib.rect_to_lidar_pseudo(center_cam)[:, 2]
        mv_height = gt_boxes[:, 2] - gt_boxes[:, 5] / 2 - cur_lidar_height
        gt_boxes[:, 2] -= mv_height  # lidar view
        return gt_boxes, mv_height

    @staticmethod
    def put_boxes_on_pseudo_road_planes(gt_boxes, points, calib, pre_gt_boxes):
        from scipy import stats
        import scipy.interpolate as interpolate

        # if len(pre_gt_boxes) < 3:
        #     use_linear_regression = True
        #     res = stats.linregress(pre_gt_boxes[:, 0], pre_gt_boxes[:, 2])
        # else:
        #     use_linear_regression = False
        use_linear_regression = False

        # use already existed boxes as interpolation of their avg height from ground planes
        if len(pre_gt_boxes) < 2:
            use_interpolate_ground_plane = False
        else:
            pre_gt_boxes = pre_gt_boxes.copy()
            sort_idxs = pre_gt_boxes[:, 0].argsort()
            pre_gt_boxes = pre_gt_boxes[sort_idxs]

            # # avoid closest boxes
            # too_close_boxes = pre_gt_boxes[1:, 0] - pre_gt_boxes[:-1, 0] > 1.5
            # too_close_boxes = np.concatenate([[True], too_close_boxes], axis=0)
            # pre_gt_boxes = pre_gt_boxes[too_close_boxes]

            if len(pre_gt_boxes) < 2:
                use_interpolate_ground_plane = False
            else:
                # , bounds_error=False, fill_value="extrapolate")
                linear_planes = interpolate.interp1d(
                    pre_gt_boxes[:, 0], pre_gt_boxes[:, 2])
                use_interpolate_ground_plane = True

        # if not use_interpolate_ground_plane: # otherwise use the minimum points
        # (x, y, z, l, w, h, theta) in lidar coords, (x, y, z, l, h, w, theta) in rect coord
        bev_boxes = gt_boxes.copy()
        bev_boxes[:, 5] = 1000.  # height -> max
        bev_boxes[:, 2] = -100.
        # bev_boxes[:, [3, 4]] = bev_boxes[:, [3, 4]].clip(min=2.)
        point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
            torch.from_numpy(points[:, 0:3]), torch.from_numpy(
                bev_boxes[:, :7])
        ).numpy()

        mv_height = np.zeros((len(gt_boxes)))
        for i in range(len(gt_boxes)):
            # interpolation
            if use_interpolate_ground_plane and linear_planes.x.min() < gt_boxes[i, 0] < linear_planes.x.max():
                mv_height[i] = gt_boxes[i, 2] - linear_planes(gt_boxes[i, 0])
            else:
                if point_indices[i].sum() > 0:
                    bev_min_height = points[point_indices[i] > 0].min(0)[2]
                else:
                    # find nearest points
                    nearest_point = points[np.linalg.norm(
                        points[:, :2] - gt_boxes[i:i+1, :2], axis=-1).argmin()]
                    point_indice = np.linalg.norm(
                        points[:, :2] - nearest_point[None, :2], axis=-1) < 2.  # 2 meters minimum
                    bev_min_height = points[point_indice].min(0)[2]
                mv_height[i] = gt_boxes[i, 2] - bev_min_height

            # if use_linear_regression: # fix large gap with linear regression
            #     regress_z = gt_boxes[i, 2] - (res.intercept + res.slope * gt_boxes[i, 0])
            #     if mv_height[i] - regress_z > 2.:
            #         mv_height[i] = regress_z

        gt_boxes[:, 2] -= mv_height  # lidar view
        return gt_boxes, mv_height

    def check_occlusion_overlap_in_camera_coord(self, frame_data, sampled_gt_boxes, total_valid_sampled_dict):
        # sorted by distance
        if getattr(self, 'far_to_near', False):
            sorted_inds = (-sampled_gt_boxes[:, 0]).argsort()  # near to far !
        else:
            sorted_inds = (sampled_gt_boxes[:, 0]).argsort()  # near to far !
        sampled_gt_boxes = sampled_gt_boxes[sorted_inds]
        total_valid_sampled_dict = [
            total_valid_sampled_dict[idx] for idx in sorted_inds]

        if self.sampler_cfg.get('USE_ROAD_PLANE', False):
            if 'road_plane' in frame_data['input_data']:
                sampled_gt_boxes, mv_height = self.put_boxes_on_road_planes(
                    sampled_gt_boxes, frame_data['input_data']['road_plane'], frame_data['input_data']['calib']
                )
                # data_dict.pop('calib')
                frame_data['input_data'].pop('road_plane')
            else:
                # sampled_gt_boxes, mv_height = self.put_boxes_on_pseudo_road_planes(
                #     sampled_gt_boxes, points, frame_data['input_data']['calib'], frame_data['input_data']['gt_boxes']
                # )
                raise ValueError(
                    'Please use the pre-computed road planes for better results.')

        left_img = frame_data['input_data']['left_img']
        right_img = frame_data['input_data']['right_img']

        calib = frame_data['input_data']['calib']
        sampled_gt_box_corners = box_utils.boxes_to_corners_3d(
            sampled_gt_boxes)  # already move down to road
        N, _, _ = sampled_gt_box_corners.shape
        sampled_gt_box_corners_rect = calib.lidar_pseudo_to_rect(
            sampled_gt_box_corners.reshape(-1, 3))
        left_pts_img, left_pts_depth = calib.rect_to_img(
            sampled_gt_box_corners_rect)  # left
        right_pts_img, right_pts_depth = calib.rect_to_img(
            sampled_gt_box_corners_rect, right=True)  # left

        left_pts_img = left_pts_img.reshape(N, 8, 2)
        right_pts_img = right_pts_img.reshape(N, 8, 2)
        left_bbox_img = np.concatenate([left_pts_img.min(
            axis=1), left_pts_img.max(axis=1)], axis=1)  # slightly larger bbox
        right_bbox_img = np.concatenate(
            [right_pts_img.min(axis=1), right_pts_img.max(axis=1)], axis=1)

        # # move shifts in 2D
        # shift_2d = (left_bbox_img - ori_left_bbox_img)
        # left_bbox_2d += shift_2d

        left_bbox_img_int = left_bbox_img.astype(int)
        right_bbox_img_int = right_bbox_img.astype(int)

        left_bbox_img_int[:, [0, 2]] = left_bbox_img_int[:,
                                                         [0, 2]].clip(min=0, max=left_img.shape[1] - 1)
        left_bbox_img_int[:, [1, 3]] = left_bbox_img_int[:,
                                                         [1, 3]].clip(min=0, max=left_img.shape[0] - 1)
        right_bbox_img_int[:, [0, 2]] = right_bbox_img_int[:, [
            0, 2]].clip(min=0, max=right_img.shape[1] - 1)
        right_bbox_img_int[:, [1, 3]] = right_bbox_img_int[:, [
            1, 3]].clip(min=0, max=right_img.shape[0] - 1)

        left_cropped_bbox = left_bbox_img - left_bbox_img_int[:, [0, 1, 0, 1]]
        right_cropped_bbox = right_bbox_img - \
            right_bbox_img_int[:, [0, 1, 0, 1]]

        sampled_mask = np.zeros((0,), dtype=bool)
        check_overlap_boxes = np.zeros((0, 4), dtype=float)
        check_overlap_boxes = np.append(
            check_overlap_boxes, frame_data['gt_info']['gt_annots_boxes_2d'], axis=0)

        sampled_crop_box_list = []
        left_warped_img_list = []
        left_max_bbox_h_list = []
        left_max_bbox_w_list = []
        right_warped_img_list = []
        right_max_bbox_h_list = []
        right_max_bbox_w_list = []

        for idx, info in enumerate(total_valid_sampled_dict):
            # read images
            cropped_left_img = io.imread(
                self.root_path / info['cropped_left_img_path'])
            cropped_right_img = io.imread(
                self.root_path / info['cropped_right_img_path'])
            cropped_left_bbox = info['cropped_left_bbox']
            cropped_right_bbox = info['cropped_right_bbox']
            cropped_left_bbox[[2, 3]] -= 1  # fix bug of prepare dataset
            cropped_right_bbox[[2, 3]] -= 1

            # if cropped_left_bbox[0] < 0. or cropped_left_bbox[1] < 0. or cropped_left_bbox[2] >= cropped_left_img.shape[1] - 1 or cropped_left_bbox[3] >= cropped_left_img.shape[0] - 1:
            #     continue
            left_warped_img = warp(
                cropped_left_img, cropped_left_bbox, left_cropped_bbox[idx])
            left_warped_img_list.append(left_warped_img)
            left_max_bbox_h = min(
                left_warped_img.shape[0], left_img.shape[0]-left_bbox_img_int[idx, 1])
            left_max_bbox_h_list.append(left_max_bbox_h)
            left_max_bbox_w = min(
                left_warped_img.shape[1], left_img.shape[1]-left_bbox_img_int[idx, 0])
            left_max_bbox_w_list.append(left_max_bbox_w)
            right_warped_img = warp(
                cropped_right_img, cropped_right_bbox, right_cropped_bbox[idx])
            right_warped_img_list.append(right_warped_img)
            right_max_bbox_h = min(
                right_warped_img.shape[0], right_img.shape[0]-right_bbox_img_int[idx, 1])
            right_max_bbox_h_list.append(right_max_bbox_h)
            right_max_bbox_w = min(
                right_warped_img.shape[1], right_img.shape[1]-right_bbox_img_int[idx, 0])
            right_max_bbox_w_list.append(right_max_bbox_w)

            sampled_crop_box = np.asarray([[left_bbox_img_int[idx, 0], left_bbox_img_int[idx, 1],
                                          left_bbox_img_int[idx, 0]+left_max_bbox_w, left_bbox_img_int[idx, 1]+left_max_bbox_h]], dtype=float)
            sampled_crop_box_list.append(sampled_crop_box)
            if self.filter_occlusion_overlap < 1.:
                overlap_with_fg = query_bbox_overlaps(
                    sampled_crop_box, check_overlap_boxes)
                # print(overlap_with_fg)
                if np.prod(overlap_with_fg.shape) > 0 and min(overlap_with_fg.max(), 1.) > self.filter_occlusion_overlap:
                    sampled_mask = np.append(sampled_mask, False)
                else:
                    sampled_mask = np.append(sampled_mask, True)

        ret_dict = {
            'sampled_gt_boxes': sampled_gt_boxes,
            'total_valid_sampled_dict': total_valid_sampled_dict,
            'mv_height': mv_height,
            'left_img': left_img,
            'right_img': right_img,
            'sampled_mask': sampled_mask,
            'check_overlap_boxes': check_overlap_boxes,
            'sampled_crop_box_list': sampled_crop_box_list,
            'left_bbox_img_int': left_bbox_img_int,
            'right_bbox_img_int': right_bbox_img_int,
            'left_warped_img_list': left_warped_img_list,
            'left_max_bbox_h_list': left_max_bbox_h_list,
            'left_max_bbox_w_list': left_max_bbox_w_list,
            'right_warped_img_list': right_warped_img_list,
            'right_max_bbox_h_list': right_max_bbox_h_list,
            'right_max_bbox_w_list': right_max_bbox_w_list,
        }
        return ret_dict

    def add_sampled_boxes_to_scene_v2(self, frame_data, prepare_data, sampled_mask):
        gt_boxes_mask = frame_data['gt_info']['gt_boxes_mask']
        gt_boxes = frame_data['gt_info']['gt_boxes'][gt_boxes_mask]
        gt_names = frame_data['gt_info']['gt_names'][gt_boxes_mask]
        points = frame_data['input_data']['points']
        calib = frame_data['input_data']['calib']
        frame_data['input_data']['ori_points'] = copy.deepcopy(points)

        gt_difficulty = frame_data['gt_info']['gt_difficulty'][gt_boxes_mask]
        gt_occluded = frame_data['gt_info']['gt_occluded'][gt_boxes_mask]
        gt_index = frame_data['gt_info']['gt_index'][gt_boxes_mask]
        gt_truncated = frame_data['gt_info']['gt_truncated'][gt_boxes_mask]
        gt_object_id = frame_data['gt_info']['object_id'][gt_boxes_mask]

        sampled_gt_boxes = prepare_data['sampled_gt_boxes']
        total_valid_sampled_dict = prepare_data['total_valid_sampled_dict']
        mv_height = prepare_data['mv_height']
        left_img = prepare_data['left_img']
        right_img = prepare_data['right_img']

        points_img, points_depth = calib.rect_to_img(
            calib.lidar_pseudo_to_rect(points))
        if self.sampler_cfg.get('remove_overlapped', True):
            non_overlapped_mask = np.ones(len(points_img), dtype=bool)

        obj_points_list = []
        obj_points_img_list = []
        sampled_gt_indices = []
        sampled_gt_difficulty = []
        sample_gt_object_id = []
        check_overlap_boxes = prepare_data['check_overlap_boxes']
        sampled_crop_box_list = prepare_data['sampled_crop_box_list']
        left_bbox_img_int = prepare_data['left_bbox_img_int']
        right_bbox_img_int = prepare_data['right_bbox_img_int']
        left_warped_img_list = prepare_data['left_warped_img_list']
        left_max_bbox_h_list = prepare_data['left_max_bbox_h_list']
        left_max_bbox_w_list = prepare_data['left_max_bbox_w_list']
        right_warped_img_list = prepare_data['right_warped_img_list']
        right_max_bbox_h_list = prepare_data['right_max_bbox_h_list']
        right_max_bbox_w_list = prepare_data['right_max_bbox_w_list']

        for idx, info in enumerate(total_valid_sampled_dict):
            if self.filter_occlusion_overlap < 1.:
                if not sampled_mask[idx]:
                    continue
            # add obj point to point cloud
            file_path = self.root_path / info['path']
            obj_points = np.fromfile(str(file_path), dtype=np.float32).reshape(
                [-1, self.sampler_cfg.NUM_POINT_FEATURES])
            obj_points[:, :3] += info['box3d_lidar'][:3]
            if self.sampler_cfg.get('USE_ROAD_PLANE', False):
                # mv height
                obj_points[:, 2] -= mv_height[idx]

            check_overlap_boxes = np.append(
                check_overlap_boxes, sampled_crop_box_list[idx], axis=0)
            obj_points_list.append(obj_points)
            obj_points_img_list.append(calib.rect_to_img(
                calib.lidar_pseudo_to_rect(obj_points[:, :3]))[0])
            sampled_gt_indices.append(info['gt_index'])
            sampled_gt_difficulty.append(info['difficulty'])
            sample_gt_object_id.append(str(int(info['this_obj_id']) + 1000))
            # put cropped obj img to origin img
            left_warped_img = left_warped_img_list[idx]
            left_max_bbox_h = left_max_bbox_h_list[idx]
            left_max_bbox_w = left_max_bbox_w_list[idx]
            left_img[left_bbox_img_int[idx, 1]:left_bbox_img_int[idx, 1]+left_max_bbox_h, left_bbox_img_int[idx, 0]:left_bbox_img_int[idx, 0]+left_max_bbox_w] = left_warped_img[:left_max_bbox_h, :left_max_bbox_w]
            right_warped_img = right_warped_img_list[idx]
            right_max_bbox_h = right_max_bbox_h_list[idx]
            right_max_bbox_w = right_max_bbox_w_list[idx]
            right_img[right_bbox_img_int[idx, 1]:right_bbox_img_int[idx, 1]+right_max_bbox_h, right_bbox_img_int[idx, 0]:right_bbox_img_int[idx, 0]+right_max_bbox_w] = right_warped_img[:right_max_bbox_h, :right_max_bbox_w]

            if self.sampler_cfg.get('remove_overlapped', True):
                non_overlapped_mask &= ((points_img[:, 0] <= left_bbox_img_int[idx, 0]) | (points_img[:, 0] >= left_bbox_img_int[idx, 2]) |
                                        (points_img[:, 1] < left_bbox_img_int[idx, 1]) | (points_img[:, 1] >= left_bbox_img_int[idx, 3]))
                for j in range(len(obj_points_list) - 1):  # exclude itself
                    non_overlapped_obj_mask = ((obj_points_img_list[j][:, 0] <= left_bbox_img_int[idx, 0]) | (obj_points_img_list[j][:, 0] >= left_bbox_img_int[idx, 2]) |
                                               (obj_points_img_list[j][:, 1] < left_bbox_img_int[idx, 1]) | (obj_points_img_list[j][:, 1] >= left_bbox_img_int[idx, 3]))
                    obj_points_img_list[j] = obj_points_img_list[j][non_overlapped_obj_mask]
                    obj_points_list[j] = obj_points_list[j][non_overlapped_obj_mask]

        if self.sampler_cfg.get('remove_overlapped', True):
            points = points[non_overlapped_mask]

        sampled_gt_boxes = sampled_gt_boxes[sampled_mask]
        total_valid_sampled_dict = [total_valid_sampled_dict[idx]
                                    for idx in np.nonzero(sampled_mask)[0]]
        if len(obj_points_list) == 0:
            return frame_data

        obj_points = np.concatenate(obj_points_list, axis=0)
        sampled_gt_names = np.array([x['name']
                                    for x in total_valid_sampled_dict])
        large_sampled_gt_boxes = box_utils.enlarge_box3d(
            sampled_gt_boxes[:, 0:7], extra_width=self.sampler_cfg.REMOVE_EXTRA_WIDTH)
        points = box_utils.remove_points_in_boxes3d(
            points, large_sampled_gt_boxes)
        points = np.concatenate([obj_points[:, :3], points], axis=0)
        gt_names = np.concatenate([gt_names, sampled_gt_names], axis=0)
        gt_boxes = np.concatenate([gt_boxes, sampled_gt_boxes], axis=0)

        frame_data['gt_info']['gt_boxes'] = gt_boxes
        frame_data['gt_info']['gt_names'] = gt_names
        frame_data['input_data']['points'] = points
        frame_data['input_data']['left_img'] = left_img
        frame_data['input_data']['right_img'] = right_img
        frame_data['gt_info']['gt_boxes_mask'] = np.ones(
            len(gt_boxes), dtype=bool)
        frame_data['gt_info']['gt_difficulty'] = np.concatenate(
            (gt_difficulty, np.array(sampled_gt_difficulty)))
        frame_data['gt_info']['gt_occluded'] = np.concatenate(
            (gt_occluded, np.zeros(len(gt_boxes) - len(gt_occluded))))
        frame_data['gt_info']['gt_truncated'] = np.concatenate(
            (gt_truncated, np.zeros(len(gt_boxes) - len(gt_truncated))))
        frame_data['gt_info']['gt_index'] = np.concatenate(
            (gt_index, np.array(sampled_gt_indices)))
        frame_data['gt_info']['object_id'] = np.concatenate(
            (gt_object_id, np.array(sample_gt_object_id)))
        return frame_data

    def add_sampled_boxes_to_scene(self, frame_data, sampled_gt_boxes, total_valid_sampled_dict):
        # sorted by distance
        if getattr(self, 'far_to_near', False):
            sorted_inds = (-sampled_gt_boxes[:, 0]).argsort()  # near to far !
        else:
            sorted_inds = (sampled_gt_boxes[:, 0]).argsort()  # near to far !
        sampled_gt_boxes = sampled_gt_boxes[sorted_inds]
        total_valid_sampled_dict = [
            total_valid_sampled_dict[idx] for idx in sorted_inds]

        gt_boxes_mask = frame_data['gt_info']['gt_boxes_mask']
        gt_boxes = frame_data['gt_info']['gt_boxes'][gt_boxes_mask]
        gt_names = frame_data['gt_info']['gt_names'][gt_boxes_mask]
        points = frame_data['input_data']['points']
        frame_data['input_data']['ori_points'] = copy.deepcopy(points)

        gt_difficulty = frame_data['gt_info']['gt_difficulty'][gt_boxes_mask]
        gt_occluded = frame_data['gt_info']['gt_occluded'][gt_boxes_mask]
        gt_index = frame_data['gt_info']['gt_index'][gt_boxes_mask]
        gt_truncated = frame_data['gt_info']['gt_truncated'][gt_boxes_mask]
        gt_object_id = frame_data['gt_info']['object_id'][gt_boxes_mask]

        if self.sampler_cfg.get('USE_ROAD_PLANE', False):
            if 'road_plane' in frame_data['input_data']:
                sampled_gt_boxes, mv_height = self.put_boxes_on_road_planes(
                    sampled_gt_boxes, frame_data['input_data']['road_plane'], frame_data['input_data']['calib']
                )
                # data_dict.pop('calib')
                frame_data['input_data'].pop('road_plane')
            else:
                sampled_gt_boxes, mv_height = self.put_boxes_on_pseudo_road_planes(
                    sampled_gt_boxes, points, frame_data['input_data']['calib'], frame_data['gt_info']['gt_boxes']
                )
                raise ValueError(
                    'Please use the pre-computed road planes for better results.')

        left_img = frame_data['input_data']['left_img']
        right_img = frame_data['input_data']['right_img']

        calib = frame_data['input_data']['calib']
        sampled_gt_box_corners = box_utils.boxes_to_corners_3d(
            sampled_gt_boxes)  # already move down to road
        N, _, _ = sampled_gt_box_corners.shape
        sampled_gt_box_corners_rect = calib.lidar_pseudo_to_rect(
            sampled_gt_box_corners.reshape(-1, 3))
        left_pts_img, left_pts_depth = calib.rect_to_img(
            sampled_gt_box_corners_rect)  # left
        right_pts_img, right_pts_depth = calib.rect_to_img(
            sampled_gt_box_corners_rect, right=True)  # left

        left_pts_img = left_pts_img.reshape(N, 8, 2)
        right_pts_img = right_pts_img.reshape(N, 8, 2)

        left_bbox_img = np.concatenate([left_pts_img.min(
            axis=1), left_pts_img.max(axis=1)], axis=1)  # slightly larger bbox
        right_bbox_img = np.concatenate(
            [right_pts_img.min(axis=1), right_pts_img.max(axis=1)], axis=1)

        # # move shifts in 2D
        # shift_2d = (left_bbox_img - ori_left_bbox_img)
        # left_bbox_2d += shift_2d

        left_bbox_img_int = left_bbox_img.astype(int)
        right_bbox_img_int = right_bbox_img.astype(int)

        left_bbox_img_int[:, [0, 2]] = left_bbox_img_int[:,
                                                         [0, 2]].clip(min=0, max=left_img.shape[1] - 1)
        left_bbox_img_int[:, [1, 3]] = left_bbox_img_int[:,
                                                         [1, 3]].clip(min=0, max=left_img.shape[0] - 1)
        right_bbox_img_int[:, [0, 2]] = right_bbox_img_int[:, [
            0, 2]].clip(min=0, max=right_img.shape[1] - 1)
        right_bbox_img_int[:, [1, 3]] = right_bbox_img_int[:, [
            1, 3]].clip(min=0, max=right_img.shape[0] - 1)

        left_cropped_bbox = left_bbox_img - left_bbox_img_int[:, [0, 1, 0, 1]]
        right_cropped_bbox = right_bbox_img - \
            right_bbox_img_int[:, [0, 1, 0, 1]]

        points_img, points_depth = calib.rect_to_img(
            calib.lidar_pseudo_to_rect(points))
        if self.sampler_cfg.get('remove_overlapped', True):
            non_overlapped_mask = np.ones(len(points_img), dtype=bool)

        obj_points_list = []
        obj_points_img_list = []
        sampled_gt_indices = []
        sample_gt_object_id = []
        sampled_gt_difficulty = []
        sampled_mask = np.zeros((0,), dtype=bool)
        check_overlap_boxes = np.zeros((0, 4), dtype=float)
        check_overlap_boxes = np.append(
            check_overlap_boxes, frame_data['gt_info']['gt_annots_boxes_2d'], axis=0)

        for idx, info in enumerate(total_valid_sampled_dict):
            file_path = self.root_path / info['path']
            # read points
            obj_points = np.fromfile(str(file_path), dtype=np.float32).reshape(
                [-1, self.sampler_cfg.NUM_POINT_FEATURES])
            obj_points[:, :3] += info['box3d_lidar'][:3]
            if self.sampler_cfg.get('USE_ROAD_PLANE', False):
                # mv height
                obj_points[:, 2] -= mv_height[idx]

            # read images
            cropped_left_img = io.imread(
                self.root_path / info['cropped_left_img_path'])
            cropped_right_img = io.imread(
                self.root_path / info['cropped_right_img_path'])
            cropped_left_bbox = info['cropped_left_bbox']
            cropped_right_bbox = info['cropped_right_bbox']
            cropped_left_bbox[[2, 3]] -= 1  # fix bug of prepare dataset
            cropped_right_bbox[[2, 3]] -= 1

            # if cropped_left_bbox[0] < 0. or cropped_left_bbox[1] < 0. or cropped_left_bbox[2] >= cropped_left_img.shape[1] - 1 or cropped_left_bbox[3] >= cropped_left_img.shape[0] - 1:
            #     continue
            left_warped_img = warp(
                cropped_left_img, cropped_left_bbox, left_cropped_bbox[idx])
            max_bbox_h = min(
                left_warped_img.shape[0], left_img.shape[0]-left_bbox_img_int[idx, 1])
            max_bbox_w = min(
                left_warped_img.shape[1], left_img.shape[1]-left_bbox_img_int[idx, 0])

            sampled_crop_box = np.asarray([[left_bbox_img_int[idx, 0], left_bbox_img_int[idx, 1],
                                          left_bbox_img_int[idx, 0]+max_bbox_w, left_bbox_img_int[idx, 1]+max_bbox_h]], dtype=float)
            if self.filter_occlusion_overlap < 1.:
                overlap_with_fg = query_bbox_overlaps(
                    sampled_crop_box, check_overlap_boxes)
                # print(overlap_with_fg)
                if np.prod(overlap_with_fg.shape) > 0 and min(overlap_with_fg.max(), 1.) > self.filter_occlusion_overlap:
                    sampled_mask = np.append(sampled_mask, False)
                    continue

            sampled_mask = np.append(sampled_mask, True)
            check_overlap_boxes = np.append(
                check_overlap_boxes, sampled_crop_box, axis=0)

            obj_points_list.append(obj_points)
            obj_points_img_list.append(calib.rect_to_img(
                calib.lidar_pseudo_to_rect(obj_points[:, :3]))[0])
            sampled_gt_indices.append(info['gt_index'])
            sample_gt_object_id.append(str(int(info['this_obj_id']) + 1000))
            sampled_gt_difficulty.append(info['difficulty'])

            left_img[left_bbox_img_int[idx, 1]:left_bbox_img_int[idx, 1]+max_bbox_h, left_bbox_img_int[idx, 0]:left_bbox_img_int[idx, 0]+max_bbox_w] = left_warped_img[:max_bbox_h, :max_bbox_w]
            # print(f'left: origin size {left_warped_img.shape[1]}x{left_warped_img.shape[0]} final cropped box size{max_bbox_w}x{max_bbox_h}')

            if self.sampler_cfg.get('remove_overlapped', True):
                non_overlapped_mask &= ((points_img[:, 0] <= left_bbox_img_int[idx, 0]) | (points_img[:, 0] >= left_bbox_img_int[idx, 2]) |
                                        (points_img[:, 1] < left_bbox_img_int[idx, 1]) | (points_img[:, 1] >= left_bbox_img_int[idx, 3]))
                for j in range(len(obj_points_list) - 1):  # exclude itself
                    non_overlapped_obj_mask = ((obj_points_img_list[j][:, 0] <= left_bbox_img_int[idx, 0]) | (obj_points_img_list[j][:, 0] >= left_bbox_img_int[idx, 2]) |
                                               (obj_points_img_list[j][:, 1] < left_bbox_img_int[idx, 1]) | (obj_points_img_list[j][:, 1] >= left_bbox_img_int[idx, 3]))
                    obj_points_img_list[j] = obj_points_img_list[j][non_overlapped_obj_mask]
                    obj_points_list[j] = obj_points_list[j][non_overlapped_obj_mask]

            right_warped_img = warp(
                cropped_right_img, cropped_right_bbox, right_cropped_bbox[idx])
            max_bbox_h = min(
                right_warped_img.shape[0], right_img.shape[0]-right_bbox_img_int[idx, 1])
            max_bbox_w = min(
                right_warped_img.shape[1], right_img.shape[1]-right_bbox_img_int[idx, 0])
            right_img[right_bbox_img_int[idx, 1]:right_bbox_img_int[idx, 1]+max_bbox_h, right_bbox_img_int[idx, 0]:right_bbox_img_int[idx, 0]+max_bbox_w] = right_warped_img[:max_bbox_h, :max_bbox_w]
            # print(f'right: origin size {right_warped_img.shape[1]}x{right_warped_img.shape[0]} final cropped box size{max_bbox_w}x{max_bbox_h}')

        if self.sampler_cfg.get('remove_overlapped', True):
            points = points[non_overlapped_mask]

        sampled_gt_boxes = sampled_gt_boxes[sampled_mask]
        total_valid_sampled_dict = [total_valid_sampled_dict[idx]
                                    for idx in np.nonzero(sampled_mask)[0]]

        if len(obj_points_list) == 0:
            return frame_data

        obj_points = np.concatenate(obj_points_list, axis=0)
        sampled_gt_names = np.array([x['name']
                                    for x in total_valid_sampled_dict])

        large_sampled_gt_boxes = box_utils.enlarge_box3d(
            sampled_gt_boxes[:,
                             0:7], extra_width=self.sampler_cfg.REMOVE_EXTRA_WIDTH
        )
        points = box_utils.remove_points_in_boxes3d(
            points, large_sampled_gt_boxes)
        points = np.concatenate([obj_points[:, :3], points], axis=0)
        gt_names = np.concatenate([gt_names, sampled_gt_names], axis=0)
        gt_boxes = np.concatenate([gt_boxes, sampled_gt_boxes], axis=0)

        frame_data['gt_info']['gt_boxes'] = gt_boxes
        frame_data['gt_info']['gt_names'] = gt_names
        frame_data['input_data']['points'] = points
        frame_data['input_data']['left_img'] = left_img
        frame_data['input_data']['right_img'] = right_img

        frame_data['gt_info']['gt_boxes_mask'] = np.ones(
            len(gt_boxes), dtype=bool)
        frame_data['gt_info']['gt_difficulty'] = np.concatenate(
            (gt_difficulty, np.array(sampled_gt_difficulty)))
        frame_data['gt_info']['gt_occluded'] = np.concatenate(
            (gt_occluded, np.zeros(len(gt_boxes) - len(gt_occluded))))
        frame_data['gt_info']['gt_truncated'] = np.concatenate(
            (gt_truncated, np.zeros(len(gt_boxes) - len(gt_truncated))))
        frame_data['gt_info']['gt_index'] = np.concatenate(
            (gt_index, np.array(sampled_gt_indices)))
        frame_data['gt_info']['object_id'] = np.concatenate(
            (gt_object_id, np.array(sample_gt_object_id)))
        # frame_data['gt_info']['gt_annots_boxes_2d'] = np.concatenate([data_dict['gt_annots_boxes_2d'], left_bbox_2d[sampled_mask]]) # TODO(hack) projected 3d boxes is not 2d bbox
        # frame_data['gt_info']['gt_boxes_2d_ignored'] = np.concatenate(
        #     [frame_data['gt_info']['gt_boxes_2d_ignored'], left_bbox_img[sampled_mask]])  # TODO(hack) projected 3d boxes is not 2d bbox

        return frame_data

    def __call__(self, data_list):
        """
        Args:
            data_list:
                [(prev2, frame_data), ..., (next, frame_data)]
                frame_data:{
                    frame_valid:
                    input_data:
                    gt_info:
                }
        Returns:

        """
        data_dict = dict(data_list)
        if data_dict[self.box3d_supervision]['input_data']['calib'].flipped:
            print('flipped, skip gt_sampling')
            return data_list

        if not all([x[1]['frame_valid'] for x in data_list]):
            return data_list

        if np.random.rand() > self.sampler_cfg.get('ratio', 0.6):
            return data_list

        if self.stop_epoch is not None and self.epoch >= self.stop_epoch:
            print("Remove ground-truth data sampling at last 5 epochs.")
            return data_list

        existed_boxes_dict = {}
        num_gt_boxes = {}
        for i in range(len(data_list)):
            existed_boxes_dict[data_list[i][0]
                               ] = data_list[i][1]['gt_info']['gt_boxes']
            num_gt_boxes[data_list[i][0]
                         ] = data_list[i][1]['gt_info']['gt_boxes'].shape[0]

        total_valid_sampled_list = []
        '''
            total_valid_sampled_list: [Car, Pedestrian, Cyclist]
                Car: {prev2, ..., next}
        '''

        gt_names = data_dict[self.box3d_supervision]['gt_info']['gt_names'].astype(
            str)
        for class_name, sample_group in self.sample_groups.items():  # Car
            '''
            sample_group:
                sample_num: 7 (config files)
                pointer: len(self.db_infos[class_name]),
                indices': np.arange(len(self.db_infos[class_name]))
            '''
            if self.limit_whole_scene:
                num_gt = np.sum(class_name == gt_names)
                sample_group['sample_num'] = str(
                    int(self.sample_class_num[class_name]) - num_gt)
            if int(sample_group['sample_num']) > 0:
                sampled_dict = self.sample_with_fixed_number(
                    class_name, sample_group)
                all_sampled_dict = self.sample_with_all_frame_tag(sampled_dict)
                valid_sampled_dict, valid_sampled_boxes_dict = self.check_sample_valid(
                    all_sampled_dict, existed_boxes_dict)
                # update exitsed boxed
                for tag, sampled_boxes in valid_sampled_boxes_dict.items():
                    existed_boxes_dict[tag] = np.concatenate(
                        (existed_boxes_dict[tag], sampled_boxes), axis=0)
                # sorted with class name
                total_valid_sampled_list.append(valid_sampled_dict)

        # convert list to dict
        total_valid_sampled_dict = {}
        sampled_gt_boxes = {}
        '''
            total_valid_sampled_dict = {prev2, ..., next}
                token -> list: [Car, Pedestrian, Cyclist]
        '''
        for tag in data_dict.keys():
            total_valid_sampled_dict[tag] = [x[tag]
                                             for x in total_valid_sampled_list]
        '''
            total_valid_sampled_dict = {prev2, ..., next}
                token -> list: [single object, ...]
        '''
        for tag, v in total_valid_sampled_dict.items():
            sampled_gt_boxes[tag] = existed_boxes_dict[tag][num_gt_boxes[tag]:, :]
            total_valid_sampled_dict[tag] = [
                item for sublist in v for item in sublist]

        if total_valid_sampled_dict[self.box3d_supervision].__len__() > 0:
            if self.sampler_cfg.get('use_samplerv2', False):
                prepare_data_dict = {}
                occ_overlap_mask = []
                for tag, frame_data in data_list:
                    prepare_data = self.check_occlusion_overlap_in_camera_coord(
                        frame_data, sampled_gt_boxes[tag], total_valid_sampled_dict[tag])
                    prepare_data_dict[tag] = prepare_data
                    occ_overlap_mask.append(prepare_data['sampled_mask'])
                occ_overlap_mask = np.all(occ_overlap_mask, axis=0)
                aug_data_list = []
                for tag, frame_data in data_list:
                    aug_data = self.add_sampled_boxes_to_scene_v2(
                        frame_data, prepare_data_dict[tag], occ_overlap_mask)
                    aug_data_list.append((tag, aug_data))
                return aug_data_list
            else:
                aug_data_list = []
                for tag, frame_data in data_list:
                    aug_data = self.add_sampled_boxes_to_scene(
                        frame_data, sampled_gt_boxes[tag], total_valid_sampled_dict[tag])
                    aug_data_list.append((tag, aug_data))
                return aug_data_list
        else:
            return data_list
