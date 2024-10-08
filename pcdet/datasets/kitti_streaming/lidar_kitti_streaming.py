# Modified from OpenPCDet. https://github.com/open-mmlab/OpenPCDet
# LiDAR KITTI Pytorch Dataset (for training LiDAR models like SECOND)

import copy
import json
import pickle
import numpy as np
import torch
from skimage import io

from pcdet.utils import box_utils, calibration_kitti_tracking, common_utils, tracking3d_kitti
from pcdet.datasets.lidar_dataset_template import DatasetTemplate
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils


class LiDARKittiStreaming(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]
        self.root_split_path = self.root_path / \
            ('training' if self.split != 'test' else 'testing')
        self.prepare_data_path = self.root_path / dataset_cfg.PREPARE_PATH
        split_file = self.prepare_data_path / \
            'ImageSets' / (self.split + '.json')
        assert split_file.exists()
        with open(split_file, 'r') as f:
            self.sample_id_list = json.load(f)
        self.kitti_infos = []
        self.include_kitti_data(self.mode)

    def include_kitti_data(self, mode):
        if self.logger is not None:
            self.logger.info('Loading KITTI streaming dataset')
        kitti_infos = []
        for info_path in self.dataset_cfg.INFO_PATH[mode]:
            info_path = self.prepare_data_path / info_path
            if not info_path.exists():
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                kitti_infos.extend(infos)

        self.kitti_infos.extend(kitti_infos)

        if self.logger is not None:
            self.logger.info('Total samples for KITTI streaming dataset: %d' %
                             (len(kitti_infos)))

    def set_split(self, split):
        super().__init__(
            dataset_cfg=self.dataset_cfg, class_names=self.class_names, training=self.training, root_path=self.root_path, logger=self.logger
        )
        self.split = split
        self.root_split_path = self.root_path / \
            ('training' if self.split != 'test' else 'testing')
        split_dir = self.prepare_data_path / \
            'ImageSets' / (self.split + '.json')
        assert split_dir.exists()
        with open(split_dir, 'r') as f:
            self.sample_id_list = json.load(f)

    def get_lidar(self, scene, filename):
        scene = scene.split('_')[0]
        lidar_file = self.root_split_path / 'velodyne' / \
            ('%s' % scene) / ('%s.bin' % filename)
        assert lidar_file.exists()
        return np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 4)

    def get_image_shape(self, scene, filename):
        scene = scene.split('_')[0]
        img_file = self.root_split_path / 'image_02' / \
            ('%s' % scene) / ('%s.png' % filename)
        assert img_file.exists()
        return np.array(io.imread(img_file).shape[:2], dtype=np.int32)

    def get_label(self, scene, filename):
        scene = scene.split('_')[0]
        label_file = self.root_split_path / 'label_02' / ('%s.txt' % scene)
        assert label_file.exists()
        return tracking3d_kitti.get_objects_from_label(label_file, filename)

    def get_calib(self, scene):
        scene = scene.split('_')[0]
        calib_file = self.root_split_path / 'calib' / ('%s.txt' % scene)
        assert calib_file.exists()
        return calibration_kitti_tracking.Calibration(calib_file)

    def get_image(self, scene, filename, image_id=2):
        scene = scene.split('_')[0]
        img_file = self.root_split_path / \
            ('image_0%s' % image_id) / scene / ('%s.png' % filename)
        assert img_file.exists()
        return io.imread(img_file).copy()

    def get_road_plane(self, scene, filename):
        scene = scene.split('_')[0]
        plane_file = self.root_split_path / 'planes' / \
            scene / ('%s.txt' % filename)
        if not plane_file.exists():
            return None

        with open(plane_file, 'r') as f:
            lines = f.readlines()
        lines = [float(i) for i in lines[3].split()]
        plane = np.asarray(lines)

        # Ensure normal is always facing up, this is in the rectified camera coordinate
        if plane[1] > 0:
            plane = -plane

        norm = np.linalg.norm(plane[0:3])
        plane = plane / norm
        return plane

    @staticmethod
    def get_fov_flag(pts_rect, img_shape, calib):
        """
        Args:
            pts_rect:
            img_shape:
            calib:

        Returns:

        """
        pts_img, pts_rect_depth = calib.rect_to_img(pts_rect)
        val_flag_1 = np.logical_and(
            pts_img[:, 0] >= 0, pts_img[:, 0] < img_shape[1])
        val_flag_2 = np.logical_and(
            pts_img[:, 1] >= 0, pts_img[:, 1] < img_shape[0])
        val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
        pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)

        return pts_valid_flag

    def get_infos(self, num_workers=4, count_inside_pts=True, sample_id_list=None):
        '''
            info -> dict:
            {
                calib：
                sample_idx -> dict:
                {
                    scene,
                    frame_tag
                },    
                infos -> dict:
                {
                    prev2,
                    prev,
                    token,
                    next -> dict：
                    {
                        frame_valid,
                        frame_idx,
                        image_shape,
                        annos
                    } 
                }
            }

        '''
        import concurrent.futures as futures

        def get_annos(scene, filename, calib, image_shape):
            obj_list = self.get_label(scene, filename)
            has_label = len(obj_list) > 0
            annotations = {}
            annotations['has_label'] = has_label
            annotations['scene'] = scene
            annotations['frame_id'] = filename
            annotations['name'] = np.array([obj.cls_type for obj in obj_list])
            # annotations['name_with_ignored'] = copy.deepcopy(
            #     annotations['name'])
            annotations['object_id'] = np.array(
                [obj.object_id for obj in obj_list])
            annotations['truncated'] = np.array(
                [obj.truncation for obj in obj_list])
            annotations['occluded'] = np.array(
                [obj.occlusion for obj in obj_list])
            annotations['alpha'] = np.array([obj.alpha for obj in obj_list])
            annotations['bbox'] = np.concatenate([obj.box2d.reshape(
                1, 4) for obj in obj_list], axis=0) if has_label else np.array([]).reshape(-1, 4)

            # bbox_with_ignored is only used for 2D detection heads, we dont need it
            # annotations['bbox_with_ignored'] = copy.deepcopy(
            #     annotations['bbox'])
            annotations['dimensions'] = np.array(
                [[obj.l, obj.h, obj.w] for obj in obj_list]) if has_label else np.array([]).reshape(-1, 3)  # lhw(camera) format
            annotations['location'] = np.concatenate([obj.loc.reshape(
                1, 3) for obj in obj_list], axis=0) if has_label else np.array([]).reshape(-1, 3)
            annotations['rotation_y'] = np.array([obj.ry for obj in obj_list])
            annotations['score'] = np.array([obj.score for obj in obj_list])
            annotations['difficulty'] = np.array(
                [obj.level for obj in obj_list], np.int32)

            annotations = common_utils.drop_info_with_name_and_keep_ignored(
                annotations, 'DontCare', ignored_keys=['has_label', 'scene', 'frame_id'])

            num_objects = len(
                [obj.cls_type for obj in obj_list if obj.cls_type != 'DontCare'])
            num_gt = len(annotations['name'])
            index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
            annotations['index_in_label'] = np.array(index, dtype=np.int32)

            if has_label:
                loc = annotations['location']
                dims = annotations['dimensions']
                rots = annotations['rotation_y']
                loc_lidar = calib.rect_to_lidar(loc)
                l, h, w = dims[:, 0:1], dims[:, 1:2], dims[:, 2:3]
                loc_lidar[:, 2] += h[:, 0] / 2
                gt_boxes_lidar = np.concatenate(
                    [loc_lidar, l, w, h, -(np.pi / 2 + rots[..., np.newaxis])], axis=1)
            else:
                gt_boxes_lidar = np.array([]).reshape(-1, 7)
            annotations['gt_boxes_lidar'] = gt_boxes_lidar

            if count_inside_pts:
                points = self.get_lidar(scene, filename)
                calib = self.get_calib(scene)
                pts_rect = calib.lidar_to_rect(points[:, 0:3])

                fov_flag = self.get_fov_flag(pts_rect, image_shape, calib)
                pts_fov = points[fov_flag]
                corners_lidar = box_utils.boxes_to_corners_3d(gt_boxes_lidar)
                num_points_in_gt = -np.ones(num_gt, dtype=np.int32)

                for k in range(num_objects):
                    flag = box_utils.in_hull(pts_fov[:, 0:3], corners_lidar[k])
                    num_points_in_gt[k] = flag.sum()
                annotations['num_points_in_gt'] = num_points_in_gt
            return annotations

        def process_single_frame(sample_idx_dict):
            print('%s sample_idx: %s' % (self.split, sample_idx_dict))
            info = {}
            info['sample_idx'] = sample_idx_dict
            info['infos'] = {}

            # get calib
            calib = self.get_calib(sample_idx_dict['scene'])  # 0001
            P2 = np.concatenate(
                [calib.P2, np.array([[0., 0., 0., 1.]])], axis=0)
            R0_4x4 = np.zeros([4, 4], dtype=calib.R0.dtype)
            R0_4x4[3, 3] = 1.
            R0_4x4[:3, :3] = calib.R0
            V2C_4x4 = np.concatenate(
                [calib.V2C, np.array([[0., 0., 0., 1.]])], axis=0)
            calib_info = {'P2': P2, 'R0_rect': R0_4x4,
                          'Tr_velo_to_cam': V2C_4x4}
            info['calib'] = calib_info

            for tag in sample_idx_dict['frame_tag']:  # token, prev2, prev, next
                info['infos'][tag] = {}

            for tag in sample_idx_dict['frame_tag']:
                # file does not exist
                if sample_idx_dict['frame_tag'][tag] == '':
                    info['infos'][tag]['frame_valid'] = False
                    continue
                info['infos'][tag]['frame_valid'] = True
                info['infos'][tag]['frame_idx'] = sample_idx_dict['frame_tag'][tag]
                info['infos'][tag]['image_shape'] = self.get_image_shape(
                    sample_idx_dict['scene'], sample_idx_dict['frame_tag'][tag])
                info['infos'][tag]['annos'] = get_annos(
                    sample_idx_dict['scene'], sample_idx_dict['frame_tag'][tag], calib, info['infos'][tag]['image_shape'])
            return info

        sample_id_list = sample_id_list if sample_id_list is not None else self.sample_id_list
        with futures.ThreadPoolExecutor(num_workers) as executor:
            infos = executor.map(process_single_frame, sample_id_list)
        return list(infos)

    def crop_images(self, scene, img_idx, gt_boxes):
        left_img = self.get_image(scene, img_idx, 2)
        right_img = self.get_image(scene, img_idx, 3)
        calib = self.get_calib(scene)

        gt_box_corners = box_utils.boxes_to_corners_3d(gt_boxes)
        N, _, _ = gt_box_corners.shape
        gt_box_corners_rect = calib.lidar_to_rect(
            gt_box_corners.reshape(-1, 3))
        left_pts_img, left_pts_depth = calib.rect_to_img(
            gt_box_corners_rect)  # left
        right_pts_img, right_pts_depth = calib.rect_to_img(
            gt_box_corners_rect, right=True)  # left

        left_pts_img = left_pts_img.reshape(N, 8, 2)
        right_pts_img = right_pts_img.reshape(N, 8, 2)

        left_bbox_img = np.concatenate([left_pts_img.min(axis=1).clip(
            min=0.), left_pts_img.max(axis=1).clip(min=0.) + 1], axis=1)  # slightly larger bbox
        right_bbox_img = np.concatenate([right_pts_img.min(axis=1).clip(
            min=0.), right_pts_img.max(axis=1).clip(min=0.) + 1], axis=1)

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

        return left_img, right_img, left_bbox_img, right_bbox_img, left_bbox_img_int, right_bbox_img_int

    def create_groundtruth_database(self, info_path=None, used_classes=None, split='train', image_crops=False):
        # if used_classes is None:
        #     used_classes = ['Car', 'Pedestrian', 'Cyclist']

        database_save_path = self.root_path / \
            ('gt_database_%s' % split)
        if database_save_path.exists():
            import os
            os.system(f'rm -r {str(database_save_path)}')
            database_save_path.mkdir(parents=True, exist_ok=True)
        else:
            database_save_path.mkdir(parents=True, exist_ok=True)

        if image_crops:
            db_info_save_path = self.root_path / \
                ('stereo_kitti_dbinfos_%s.pkl' % split)
            db_info_with_frameid_save_path = self.root_path / \
                ('stereo_kitti_dbinfos_with_frameid_%s.pkl' % split)
            database_image_save_path = database_save_path / 'images'
            database_image_save_path.mkdir(parents=True, exist_ok=True)
        else:
            db_info_save_path = self.root_path / \
                ('kitti_dbinfos_%s.pkl' % split)
            db_info_with_frameid_save_path = self.root_path / \
                ('kitti_dbinfos_with_frameid_%s.pkl' % split)
        database_lidar_save_path = database_save_path / 'lidar'
        database_lidar_save_path.mkdir(parents=True, exist_ok=True)

        with open(info_path, 'rb') as f:
            infos = pickle.load(f)

        all_db_infos = {}
        for k in range(len(infos)):  # len of train set
            info = infos[k]
            scene = info['sample_idx']['scene']
            token_sample_idx = info['sample_idx']['frame_tag']['token']
            print('gt_database sample: %d/%d, scene: %s, frame_idx: %s' %
                  (k + 1, len(infos), scene, token_sample_idx))

            obj_id_list = info['infos']['token']['annos']['object_id']
            token_annos = info['infos']['token']['annos']
            index_in_label = token_annos['index_in_label']

            has_label, gt_boxes, names, difficulty, bbox = \
                token_annos['has_label'], token_annos['gt_boxes_lidar'], token_annos['name'], \
                token_annos['difficulty'], token_annos['bbox']

            if not has_label:
                # for k in used_classes:
                #     all_db_infos[k].append(
                #         {'has_label': False,
                #             'scene': scene,
                #             'sample_idx': token_sample_idx,
                #             'db_info': None})
                continue

            points = self.get_lidar(scene, token_sample_idx)
            point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
                torch.from_numpy(points[:, 0:3]), torch.from_numpy(gt_boxes)).numpy()  # (nboxes, npoints)

            if image_crops:
                left_img, right_img, left_bbox_img, right_bbox_img, left_bbox_img_int, right_bbox_img_int = \
                    self.crop_images(scene, token_sample_idx, gt_boxes)

            single_frame_db_info = {}
            for idx, this_obj_id in enumerate(obj_id_list):
                db_info = {}

                # Save the cropped point cloud
                save_points_path = database_lidar_save_path / \
                    (f'{scene}_{token_sample_idx}_{names[idx]}_{this_obj_id}.bin')
                gt_points = points[point_indices[idx] > 0]
                gt_points[:, :3] -= gt_boxes[idx, :3]
                with open(save_points_path, 'w') as f:
                    gt_points.tofile(f)

                if image_crops:
                    save_left_path = database_image_save_path / \
                        (f'{scene}_{token_sample_idx}_{names[idx]}_{this_obj_id}_left.jpg')
                    cropped_left_img = left_img[left_bbox_img_int[idx, 1]: left_bbox_img_int[idx,
                                                                                             3]+1, left_bbox_img_int[idx, 0]: left_bbox_img_int[idx, 2]+1]
                    cropped_left_bbox = left_bbox_img[idx] - \
                        left_bbox_img_int[idx, [0, 1, 0, 1]]
                    io.imsave(str(save_left_path), cropped_left_img)

                    save_right_path = database_image_save_path / \
                        (f'{scene}_{token_sample_idx}_{names[idx]}_{this_obj_id}_right.jpg')
                    cropped_right_img = right_img[right_bbox_img_int[idx, 1]: right_bbox_img_int[idx,
                                                                                                 3]+1, right_bbox_img_int[idx, 0]: right_bbox_img_int[idx, 2]+1]
                    cropped_right_bbox = right_bbox_img[idx] - \
                        right_bbox_img_int[idx, [0, 1, 0, 1]]
                    io.imsave(str(save_right_path), cropped_right_img)

                if (used_classes is None) or names[idx] in used_classes:
                    db_path = str(save_points_path.relative_to(
                        self.root_path))  # gt_database/xxxxx.bin
                    db_info = {'scene': scene, 'sample_idx': token_sample_idx, 'name': names[idx], 
                               'path': db_path, 'gt_index': index_in_label[idx],
                               'this_obj_id': this_obj_id, 'box3d_lidar': gt_boxes[idx],
                               'num_points_in_gt': gt_points.shape[0], 'difficulty': difficulty[idx],
                               'bbox': bbox[idx], 'score': token_annos['score'][idx], 'annos_index': idx}

                    if image_crops:
                        db_info.update(cropped_left_bbox=cropped_left_bbox,
                                       cropped_left_img_path=str(
                                           save_left_path.relative_to(self.root_path)),
                                       cropped_right_bbox=cropped_right_bbox,
                                       cropped_right_img_path=str(save_right_path.relative_to(self.root_path)))

                    if names[idx] in all_db_infos:
                        all_db_infos[names[idx]].append(db_info)
                    else:
                        all_db_infos[names[idx]] = [db_info]

        for k, v in all_db_infos.items():
            print('Database %s: %d' % (k, len(v)))
        with open(db_info_save_path, 'wb') as f:
            pickle.dump(all_db_infos, f)
        
        '''
        ori:
            Car: [] -> list
            Pedestrian: []
            Cyclist: []
        new:
            scene:
                sample_idx:
                    Car:
                        len_db_info: int
                        db_info: [] -> list
                    Pedestrian:
                    Cyclist:
        '''
        new_db_info = {}
        for name, value in all_db_infos.items():
            for db_info in value:
                scene = db_info['scene']
                sample_idx = db_info['sample_idx']
                if scene not in new_db_info:
                    new_db_info[scene] = {}
                if sample_idx not in new_db_info[scene]:
                    new_db_info[scene][sample_idx] = {}
                if name not in new_db_info[scene][sample_idx]:
                    new_db_info[scene][sample_idx][name] = {
                        'db_info': []
                    }
                new_db_info[scene][sample_idx][name]['db_info'].append(db_info)
        
        for k,v in new_db_info.items():
            # scene
            for k1, v1 in v.items():
                # sample_idx
                for k2, v2 in v1.items():
                    # Car
                    v2['len_db_info'] = len(v2['db_info'])
        
        with open(db_info_with_frameid_save_path, 'wb') as f:
            pickle.dump(new_db_info, f)
                


def create_kitti_infos(dataset_cfg, class_names, data_path, save_path, workers=4, create_infos=True, create_gt_database=True, image_crops=False):
    dataset = LiDARKittiStreaming(
        dataset_cfg=dataset_cfg, class_names=class_names, root_path=data_path, training=False)
    train_split, val_split, sample_train_split, sample_val_split = 'train', 'val', 'train_sample', 'val_sample'
    sample_stride = dataset_cfg.get('SAMPLE_STRIDE', 1)
    prepare_path = dataset_cfg.get('PREPARE_PATH', f'{sample_stride}')
    prepare_data_path = save_path / prepare_path

    train_filename = prepare_data_path / ('kitti_infos_%s.pkl' % train_split)
    val_filename = prepare_data_path / ('kitti_infos_%s.pkl' % val_split)
    sample_train_filename = prepare_data_path / \
        ('kitti_infos_%s.pkl' % sample_train_split)
    sample_val_filename = prepare_data_path / \
        ('kitti_infos_%s.pkl' % sample_val_split)

    print('---------------Start to generate data infos---------------')
    if create_infos:
        dataset.set_split(train_split)
        kitti_infos_train = dataset.get_infos(
            num_workers=workers, count_inside_pts=False)
        with open(train_filename, 'wb') as f:
            pickle.dump(kitti_infos_train, f)
        print('Kitti info train file is saved to %s' % train_filename)

        dataset.set_split(val_split)
        kitti_infos_val = dataset.get_infos(
            num_workers=workers, count_inside_pts=False)
        with open(val_filename, 'wb') as f:
            pickle.dump(kitti_infos_val, f)
        print('Kitti info val file is saved to %s' % val_filename)

        dataset.set_split(sample_train_split)
        kitti_infos_split = dataset.get_infos(
            num_workers=workers, count_inside_pts=False)
        with open(sample_train_filename, 'wb') as f:
            pickle.dump(kitti_infos_split, f)
        print('Kitti info val file is saved to %s' % sample_train_filename)

        dataset.set_split(sample_val_split)
        kitti_infos_split = dataset.get_infos(
            num_workers=workers, count_inside_pts=False)
        with open(sample_val_filename, 'wb') as f:
            pickle.dump(kitti_infos_split, f)
        print('Kitti info val file is saved to %s' % sample_val_filename)

    if create_gt_database:

        print('---------------Start create groundtruth database for data augmentation---------------')
        dataset.set_split(train_split)
        dataset.create_groundtruth_database(
            train_filename, split=train_split, image_crops=image_crops)

        print('---------------Start create groundtruth database for data augmentation---------------')
        dataset.set_split(sample_train_split)
        dataset.create_groundtruth_database(
            sample_train_filename, split=sample_train_split, image_crops=image_crops)

    print('---------------Data preparation Done---------------')


if __name__ == '__main__':
    import yaml
    from pathlib import Path
    from easydict import EasyDict
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('command', type=str)
    parser.add_argument(
        '--cfg', type=str, default='./configs/lidar/dataset_configs/kitti_tracking_streaming-token_prev2_prev_next.yaml')
    parser.add_argument('--image_crops', action='store_true')
    parser.add_argument('--workers', type=int, default=32)
    args = parser.parse_args()

    if args.command == 'create_kitti_infos':
        dataset_cfg = EasyDict(
            yaml.load(open(args.cfg), Loader=yaml.FullLoader))
        ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()

        create_kitti_infos(
            dataset_cfg=dataset_cfg,
            class_names=['Car', 'Pedestrian', 'Cyclist'],
            data_path=ROOT_DIR / 'data' / 'kitti_tracking',
            save_path=ROOT_DIR / 'data' / 'kitti_tracking',
            create_infos=True,
            create_gt_database=False,
            workers=args.workers,
        )
    elif args.command == 'create_gt_database_only':
        dataset_cfg = EasyDict(
            yaml.load(open(args.cfg), Loader=yaml.FullLoader))
        ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()

        create_kitti_infos(
            dataset_cfg=dataset_cfg,
            class_names=['Car', 'Pedestrian', 'Cyclist'],
            data_path=ROOT_DIR / 'data' / 'kitti_tracking',
            save_path=ROOT_DIR / 'data' / 'kitti_tracking',
            create_infos=False,
            create_gt_database=True,
            workers=args.workers,
            image_crops=args.image_crops,
        )
