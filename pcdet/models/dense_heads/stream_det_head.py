"""A variant of anchor_head_single.

The differences are as follows:
* two more options: num_convs, GN
* apply two split convs for regression outputs and classification outputs
* when num_convs == 0, this module should be almost the same as anchor_head_single
* in conv_box/cls, the kernel size is modified to 3 instead of 1
"""

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from collections import Counter

from pcdet.utils.common_utils import dist_reduce_mean
from .anchor_stream_head_template import AnchorStreamHeadTemplate
from pcdet.ops.iou3d_nms.iou3d_nms_utils import boxes_iou_bev


def convbn(in_planes, out_planes, kernel_size, stride, pad, dilation, gn=False, groups=32):
    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=dilation if dilation > 1 else pad, dilation=dilation, bias=False),
                         nn.BatchNorm2d(out_planes) if not gn else nn.GroupNorm(groups, out_planes))


class StreamDetHead(AnchorStreamHeadTemplate):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range,
                 predict_boxes_when_training=True, **kwargs):
        super().__init__(
            model_cfg=model_cfg, num_class=num_class, class_names=class_names, grid_size=grid_size, point_cloud_range=point_cloud_range,
            predict_boxes_when_training=predict_boxes_when_training
        )

        self.num_anchors_per_location = sum(self.num_anchors_per_location)
        self.num_convs = model_cfg.get('NUM_CONVS', 2)
        self.num_filters = model_cfg.get('NUM_FILTERS', 64)
        self.GN = model_cfg.GN
        self.xyz_for_angles = getattr(model_cfg, 'xyz_for_angles', False)
        self.hwl_for_angles = getattr(model_cfg, 'hwl_for_angles', False)

        if self.num_convs > 0:
            self.rpn3d_cls_convs = []
            self.rpn3d_bbox_convs = []
            self.rpn3d_cls_convs.append(
                nn.Sequential(
                    convbn(input_channels, self.num_filters,
                           3, 1, 1, 1, gn=self.GN),
                    nn.ReLU(inplace=True))
            )
            self.rpn3d_bbox_convs.append(
                nn.Sequential(
                    convbn(input_channels, self.num_filters,
                           3, 1, 1, 1, gn=self.GN),
                    nn.ReLU(inplace=True))
            )
            for _ in range(self.num_convs-1):
                self.rpn3d_cls_convs.append(
                    nn.Sequential(
                        convbn(self.num_filters, self.num_filters,
                               3, 1, 1, 1, gn=self.GN),
                        nn.ReLU(inplace=True))
                )
                self.rpn3d_bbox_convs.append(
                    nn.Sequential(
                        convbn(self.num_filters, self.num_filters,
                               3, 1, 1, 1, gn=self.GN),
                        nn.ReLU(inplace=True))
                )
            assert len(self.rpn3d_cls_convs) == self.num_convs
            assert len(self.rpn3d_bbox_convs) == self.num_convs
            self.rpn3d_cls_convs = nn.Sequential(*self.rpn3d_cls_convs)
            self.rpn3d_bbox_convs = nn.Sequential(*self.rpn3d_bbox_convs)

        cls_feature_channels = self.num_filters
        cls_groups = 1

        self.conv_cls = nn.Conv2d(
            cls_feature_channels, self.num_anchors_per_location * self.num_class,
            kernel_size=3, padding=1, stride=1, groups=cls_groups
        )
        if self.xyz_for_angles and self.hwl_for_angles:
            box_dim = self.num_anchors_per_location * self.box_coder.code_size
        elif not self.xyz_for_angles and not self.hwl_for_angles:
            box_dim = self.num_class * 6 + self.num_anchors_per_location * \
                (self.box_coder.code_size - 6)
        else:
            box_dim = self.num_class * 3 + self.num_anchors_per_location * \
                (self.box_coder.code_size - 3)
        self.conv_box = nn.Conv2d(
            self.num_filters, box_dim,
            kernel_size=3, padding=1, stride=1
        )

        self.num_angles = self.num_anchors_per_location // self.num_class

        if self.model_cfg.get('USE_DIRECTION_CLASSIFIER', None) is not None:
            self.conv_dir_cls = nn.Conv2d(
                cls_feature_channels,
                self.num_anchors_per_location * self.model_cfg.NUM_DIR_BINS,
                kernel_size=1,
                groups=cls_groups
            )
        else:
            self.conv_dir_cls = None

        self.box3d_supervision = model_cfg.get('BOX3D_SUPERVISION', 'token')
        self.history_tag = self.model_cfg.get('HISTORY_TAG', None)
        self.init_weights()

    def init_weights(self):
        pi = 0.01
        nn.init.normal_(self.conv_cls.weight, std=0.1)
        nn.init.normal_(self.conv_box.weight, std=0.02)
        nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))
        # nn.init.normal_(self.conv_box.weight, mean=0, std=0.001)

    def get_anchor(self, batch_size):
        if isinstance(self.anchors, list):
            if self.use_multihead:
                anchors = torch.cat(
                    [anchor.permute(3, 4, 0, 1, 2, 5).contiguous().view(-1, anchor.shape[-1]) for anchor in
                     self.anchors], dim=0)
            else:
                anchors = torch.cat(self.anchors, dim=-3)
        else:
            anchors = self.anchors
        anchors = anchors.view(
            1, -1, anchors.shape[-1]).repeat(batch_size, 1, 1)
        return anchors

    def get_trend_layer_loss(self):
        def get_pos_ind(box_cls_labels):
            positives = box_cls_labels > 0
            reg_weights = positives.float()
            pos_normalizer = positives.sum(1, keepdim=True).float()
            if self.reduce_avg_factor:
                pos_normalizer = dist_reduce_mean(
                    pos_normalizer.view(-1).mean())
            reg_weights /= torch.clamp(pos_normalizer, min=self.clamp_value)
            pos_inds = reg_weights > 0
            return pos_inds, reg_weights

        def get_xyz_sin_difference(boxes1, boxes2, dim=6):
            assert dim != -1
            rad_pred_encoding = torch.sin(
                boxes1[..., dim:dim + 1]) * torch.cos(boxes2[..., dim:dim + 1])
            rad_tg_encoding = torch.cos(
                boxes1[..., dim:dim + 1]) * torch.sin(boxes2[..., dim:dim + 1])
            # (x, y, z, sin(pred_rad-tg_rad))
            boxes1 = torch.cat(
                [boxes1[..., :3], rad_pred_encoding, boxes1[..., dim + 1:]], dim=-1)
            boxes2 = torch.cat(
                [boxes2[..., :3], rad_tg_encoding, boxes2[..., dim + 1:]], dim=-1)
            return boxes1, boxes2

        tb_dict = {
            'trend_loss': 0.,
            'delta_loc_loss': 0.,
            'delta_velo_loss': 0.,
        }

        if 'token' not in self.forward_ret_dict['history_reg_target']:
            return 0., tb_dict

        box_cls_labels = self.forward_ret_dict['box_cls_labels']
        box_preds = self.forward_ret_dict['box_preds']
        box_reg_targets = self.forward_ret_dict['box_reg_targets']
        batch_size = int(box_preds.shape[0])
        box_preds = box_preds.view(batch_size, -1,
                                   box_preds.shape[-1] // self.num_anchors_per_location if not self.use_multihead else
                                   box_preds.shape[-1])

        anchors = self.get_anchor(batch_size)
        pos_inds, reg_weights = get_pos_ind(
            self.forward_ret_dict['box_cls_labels'])
        decoded_box_preds = self.box_coder.decode_torch(
            box_preds[pos_inds], anchors[pos_inds])
        decoded_reg_targets = self.box_coder.decode_torch(
            box_reg_targets[pos_inds], anchors[pos_inds])
        reg_weights = reg_weights[pos_inds]

        if self.model_cfg.LOSS_CONFIG.get('SAMPLE_BALANCING', False):
            pass


        token_boxes_id = self.forward_ret_dict['history_reg_target']['token']['object_id']
        token_boxes_tg = self.forward_ret_dict['history_reg_target']['token']['gt_boxes'][:, :7]

        # ignore some category
        trend_loss_category = self.model_cfg.LOSS_CONFIG.get('CALCULATE_TREND_CATEGORY', [1, 2, 3])
        labels_fg = box_cls_labels[pos_inds]
        category_mask = torch.zeros_like(labels_fg, dtype=torch.bool)
        for category in trend_loss_category:
            category_mask |= (labels_fg == category)
        decoded_reg_targets = decoded_reg_targets[category_mask]
        decoded_box_preds = decoded_box_preds[category_mask]
        reg_weights = reg_weights[category_mask]

        # The index in decoded_box_preds should consistent with the index in token_boxes_tg
        matching_matrix = boxes_iou_bev(decoded_reg_targets, token_boxes_tg)
        if matching_matrix.numel() == 0:
            return 0., tb_dict
        max_values, max_indices = torch.max(matching_matrix, dim=1)
        matching_mask = max_values > 0.
        valid_decoded_box_preds = decoded_box_preds[matching_mask]
        valid_decoded_reg_targets = decoded_reg_targets[matching_mask]
        valid_max_indices = max_indices[matching_mask]
        selected_token_tg = torch.index_select(
            token_boxes_tg, 0, valid_max_indices)

        # If the edge length difference is too large, it means that they are not the same object.
        consistency_edge_mask = (
            valid_decoded_reg_targets[:, 3:6] - selected_token_tg[:, 3:6]).abs().sum(dim=-1) < 0.1
        valid_decoded_box_preds = valid_decoded_box_preds[consistency_edge_mask]
        selected_token_tg = selected_token_tg[consistency_edge_mask]

        valid_decoded_box_preds_sin, selected_token_tg_sin1 = get_xyz_sin_difference(
            valid_decoded_box_preds, selected_token_tg)
        delta_loc1 = valid_decoded_box_preds_sin - selected_token_tg_sin1

        trend_loss_src = torch.zeros_like(selected_token_tg)
        if 'prev' in self.forward_ret_dict['history_reg_target']:
            selected_prev_tg = torch.index_select(
                self.forward_ret_dict['history_reg_target']['prev']['gt_boxes'][:, :7], 0, valid_max_indices)
            selected_prev_tg = selected_prev_tg[consistency_edge_mask]
            selected_token_tg_sin2, selected_prev_tg_sin1 = get_xyz_sin_difference(
                selected_token_tg, selected_prev_tg)
            delta_loc2 = selected_token_tg_sin2 - selected_prev_tg_sin1
            delta_loc_loss_src = self.trend_loss_func(
                delta_loc1, delta_loc2, reg_weights[matching_mask][consistency_edge_mask])
            trend_loss_src = delta_loc_loss_src
            tb_dict.update(delta_loc_loss=(
                delta_loc_loss_src.sum() / batch_size).item())

        adap_bal_hyparam = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS.get(
            'adap_bal_hyparam', 0.)
        bal_hyparam = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS.get(
            'bal_hyparam', 0.)
        assert not (adap_bal_hyparam > 0. and bal_hyparam > 0.), 'ada_bal_hyparam and bal_hyparam cannot both be greater than zero!'
        if 'prev' in self.forward_ret_dict['history_reg_target'] and 'prev2' in self.forward_ret_dict['history_reg_target'] and (adap_bal_hyparam > 0. or bal_hyparam > 0.):
            selected_prev2_tg = torch.index_select(
                self.forward_ret_dict['history_reg_target']['prev2']['gt_boxes'][:, :7], 0, valid_max_indices)
            selected_prev2_tg = selected_prev2_tg[consistency_edge_mask]
            selected_prev_tg_sin2, selected_prev2_tg_sin1 = get_xyz_sin_difference(
                selected_prev_tg, selected_prev2_tg)
            delta_loc3 = selected_prev_tg_sin2 - selected_prev2_tg_sin1
            delta_velocity1 = delta_loc1 - delta_loc2
            delta_velocity2 = delta_loc2 - delta_loc3
            delta_velocity_loss_src = self.trend_loss_func(
                delta_velocity1, delta_velocity2, reg_weights[matching_mask][consistency_edge_mask])

            if adap_bal_hyparam > 0.:  # does not work well!
                delta_velocity2_log = torch.log(torch.clamp(torch.abs(delta_velocity2), min=1e-8, max=1e-1))
                norm_min_value = torch.log(torch.tensor(1e-8))
                norm_max_value = torch.log(torch.tensor(1e-1))
                delta_velocity2_norm = (delta_velocity2_log - norm_min_value) / (norm_max_value - norm_min_value)
                adaptive_balance_factor = adap_bal_hyparam * delta_velocity2_norm
        
                trend_loss_src = (1 - adaptive_balance_factor) * delta_loc_loss_src + \
                    adaptive_balance_factor * delta_velocity_loss_src    
            elif bal_hyparam > 0.:
                trend_loss_src = delta_loc_loss_src + bal_hyparam * delta_velocity_loss_src
            else:
                raise NotImplementedError
            
            tb_dict.update(delta_velo_loss=(
                delta_velocity_loss_src.sum() / batch_size).item())

        trend_loss = trend_loss_src.sum() / batch_size
        trend_loss_weight = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS.get(
            'trend_weight', 0.)
        trend_loss *= trend_loss_weight
        tb_dict.update(trend_loss=trend_loss.item())
        return trend_loss, tb_dict

    def get_loss(self, batch_dict, tb_dict=None):
        if tb_dict is None:
            tb_dict = {}

        supervision_dict = batch_dict[self.box3d_supervision]
        targets_dict = self.assign_targets(
            gt_boxes=supervision_dict['gt_boxes'],
            data_dict=supervision_dict,
        )
        self.forward_ret_dict.update(targets_dict)

        cls_loss, tb_dict_cls = self.get_cls_layer_loss()
        box_loss, tb_dict_box = self.get_box_reg_layer_loss()
        tb_dict.update(tb_dict_cls)
        tb_dict.update(tb_dict_box)
        rpn_loss = cls_loss + box_loss

        if self.model_cfg.LOSS_CONFIG.get('TREND_LOSS_TYPE', None) and self.history_tag:
            # Only supports batchsize==1, delete non-consecutive objects
            history_reg_target = {}
            common_object_id = batch_dict['token']['object_id'][0]
            object_id_arr = {}
            gt_boxes_arr = {}
            for k in ['token', *self.history_tag]:
                if batch_dict[k] is not None:
                    common_object_id = np.intersect1d(
                        common_object_id, batch_dict[k]['object_id'][0])
                    object_id_arr[k] = batch_dict[k]['object_id'][0]
                    gt_boxes_arr[k] = batch_dict[k]['gt_boxes'].squeeze(dim=0)
                    assert object_id_arr[k].shape[0] == gt_boxes_arr[k].shape[
                        0], f'object_id_arr[{k}].shape[0]=={object_id_arr[k].shape[0]} and gt_boxes_arr[{k}].shape[0]=={gt_boxes_arr[k].shape[0]} are not equal!!!'
            sorted_common_object_id = [
                x for x in batch_dict['token']['object_id'][0] if x in common_object_id]
            if len(sorted_common_object_id) > 0:
                object_id_to_gt_boxes = {key: dict(zip(id_array, gt_array)) for key, id_array, gt_array in zip(
                    object_id_arr.keys(), object_id_arr.values(), gt_boxes_arr.values())}
                sorted_id_arr = {key: np.array(
                    [id_ for id_ in sorted_common_object_id]) for key in object_id_arr.keys()}
                sorted_gt_arr = {key: torch.stack(
                    ([dict_[id_] for id_ in sorted_common_object_id]), dim=0) for key, dict_ in object_id_to_gt_boxes.items()}
                history_reg_target = {k: {'object_id': sorted_id_arr[k], 'gt_boxes': sorted_gt_arr[k]} for k in [
                    *self.history_tag, 'token'] if k in object_id_arr}

            for k, v in history_reg_target.items():
                history_tg_dict = self.assign_targets(
                    gt_boxes=v['gt_boxes'].unsqueeze(0))
                history_reg_target[k].update(history_tg_dict)

            # for k, v in history_reg_target.items():
            #     print(k)
            #     print(history_reg_target[k].keys())
            #     print(history_reg_target[k]['gt_boxes'].shape)
            #     print(history_reg_target[k]['box_reg_targets'].shape)

            # all_equal = all(np.array_equal(object_id_arr['token'], arr) for k, arr in object_id_arr.items())
            # if not all_equal:
            #     for k in ['token', *self.history_tag]:
            #         print(k, object_id_arr[k], object_id_arr[k].shape)
            #         print(k, gt_boxes_arr[k], gt_boxes_arr[k].shape)
            #         print('*********')
            #         print(k, sorted_id_arr[k], sorted_id_arr[k].shape)
            #         print(k, sorted_gt_arr[k], sorted_gt_arr[k].shape)
            #     print('====================')
            # print('###########################')

            self.forward_ret_dict.update(history_reg_target=history_reg_target)
            trend_loss, tb_dict_trend = self.get_trend_layer_loss()
            tb_dict.update(tb_dict_trend)
            try:
                # There is a very small probability that this implementation will cause CUDA errors
                rpn_loss += trend_loss
            except:
                print(
                    '================= RuntimeError: CUDA error: invalid configuration argument =================')
                print(
                    '================= trend_loss is not added to rpn_loss =================')

        if 'imitation_features_pairs' in self.forward_ret_dict:
            npairs = len(self.forward_ret_dict["imitation_features_pairs"])
            if npairs > 0:
                if 'gt_boxes' in supervision_dict:
                    self.forward_ret_dict['gt_boxes'] = supervision_dict['gt_boxes']
                for feature_pairs in self.forward_ret_dict["imitation_features_pairs"]:
                    features_preds = feature_pairs['pred']
                    features_targets = feature_pairs['gt']
                    tag = (
                        '_' + feature_pairs['student_feature_name']) if npairs > 1 else ''
                    imitation_loss, tb_dict_imitation = self.get_imitation_reg_layer_loss(
                        features_preds=features_preds,
                        features_targets=features_targets,
                        imitation_cfg=feature_pairs['config'])
                    tb_dict_imitation = {k + tag: v for k,
                                        v in tb_dict_imitation.items()}
                    tb_dict.update(tb_dict_imitation)
                    rpn_loss += imitation_loss
        tb_dict['rpn_loss'] = rpn_loss.item()
        return rpn_loss, tb_dict

    def forward(self, data_dict):
        # NOTE: clear forward ret dict to avoid potential bugs
        self.forward_ret_dict.clear()
        spatial_features_2d = data_dict['spatial_features_2d']

        if self.do_feature_imitation and self.training:
            self.forward_ret_dict['imitation_features_pairs'] = []
            imitation_conv_layers = [self.conv_imitation] if len(
                self.imitation_configs) == 1 else self.conv_imitation
            for cfg, imitation_conv in zip(self.imitation_configs, imitation_conv_layers):
                teacher_feature_name = cfg.teacher_feature_layer
                student_feature_name = cfg.student_feature_layer
                if teacher_feature_name not in data_dict or student_feature_name not in data_dict:
                    continue
                self.forward_ret_dict['imitation_features_pairs'].append(
                    dict(
                        config=cfg,
                        teacher_feature_name=teacher_feature_name,
                        student_feature_name=student_feature_name,
                        gt=data_dict[teacher_feature_name],
                        pred=imitation_conv(data_dict[student_feature_name])
                    )
                )

            # for k in data_dict:
            #     if k in ["lidar_batch_cls_preds", "lidar_batch_box_preds"]:
            #         self.forward_ret_dict[k] = data_dict[k]

        cls_features = spatial_features_2d
        reg_features = spatial_features_2d
        if self.num_convs > 0:
            cls_features = self.rpn3d_cls_convs(cls_features)
            reg_features = self.rpn3d_bbox_convs(reg_features)
        box_preds = self.conv_box(reg_features)
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]

        data_dict['reg_features'] = reg_features

        cls_preds = self.conv_cls(cls_features)
        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]

        if not self.xyz_for_angles or not self.hwl_for_angles:
            # TODO: here we assume that for each class, there are only anchors with difference angles
            if self.xyz_for_angles:
                xyz_dim = self.num_anchors_per_location * 3
                xyz_shapes = (
                    self.num_class, self.num_anchors_per_location // self.num_class, 3)
            else:
                xyz_dim = self.num_class * 3
                xyz_shapes = (self.num_class, 1, 3)
            if self.hwl_for_angles:
                hwl_dim = self.num_anchors_per_location * 3
                hwl_shapes = (
                    self.num_class, self.num_anchors_per_location // self.num_class, 3)
            else:
                hwl_dim = self.num_class * 3
                hwl_shapes = (self.num_class, 1, 3)
            rot_dim = self.num_anchors_per_location * \
                (self.box_coder.code_size - 6)
            rot_shapes = (self.num_class, self.num_anchors_per_location //
                          self.num_class, (self.box_coder.code_size - 6))
            assert box_preds.shape[-1] == xyz_dim + hwl_dim + rot_dim
            xyz_preds, hwl_preds, rot_preds = torch.split(
                box_preds, [xyz_dim, hwl_dim, rot_dim], dim=-1)
            # anchors [Nz, Ny, Nx, N_cls*N_size=3*1, N_rot, 7]
            xyz_preds = xyz_preds.view(*xyz_preds.shape[:3], *xyz_shapes)
            hwl_preds = hwl_preds.view(*hwl_preds.shape[:3], *hwl_shapes)
            rot_preds = rot_preds.view(*rot_preds.shape[:3], *rot_shapes)
            # expand xyz and hwl
            if not self.xyz_for_angles:
                xyz_preds = xyz_preds.repeat(
                    1, 1, 1, 1, rot_preds.shape[4] // xyz_preds.shape[4], 1)
            if not self.hwl_for_angles:
                hwl_preds = hwl_preds.repeat(
                    1, 1, 1, 1, rot_preds.shape[4] // hwl_preds.shape[4], 1)
            box_preds = torch.cat([xyz_preds, hwl_preds, rot_preds], dim=-1)
            box_preds = box_preds.view(*box_preds.shape[:3], -1)

        self.forward_ret_dict['cls_preds'] = cls_preds
        self.forward_ret_dict['box_preds'] = box_preds

        if 'valids' in data_dict:
            self.forward_ret_dict['valids'] = data_dict['valids'].any(1)

        if self.conv_dir_cls is not None:
            dir_cls_preds = self.conv_dir_cls(cls_features)
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
            self.forward_ret_dict['dir_cls_preds'] = dir_cls_preds
        else:
            dir_cls_preds = None
        if not self.training or self.predict_boxes_when_training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=data_dict['batch_size'],
                cls_preds=cls_preds, box_preds=box_preds, dir_cls_preds=dir_cls_preds
            )
            data_dict['batch_cls_preds'] = batch_cls_preds
            data_dict['batch_box_preds'] = batch_box_preds
            # TODO: check the code here, we add sigmoid in the generate predicted boxes, so set normalized to be True
            data_dict['cls_preds_normalized'] = False

        return data_dict
