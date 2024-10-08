# 2D detection head based on mmdetection.


import numpy as np
import torch
import torch.nn as nn

from mmdet.models.builder import build_head
from mmdet.core import bbox2result



class StreamMMDet2DHead(nn.Module):
    def __init__(self, model_cfg):
        super(StreamMMDet2DHead, self).__init__()
        self.bbox_head = build_head(model_cfg.cfg)
        self.bbox_head.init_weights()
        self.use_3d_center = model_cfg.use_3d_center
        self.box2d_supervision = model_cfg.get('BOX2D_SUPERVISION', 'token')
        self.get_box2d_when_inference = model_cfg.get('GET_BOX2D_WHEN_INFERENCE', False)

    def get_loss(self, batch_dict, tb_dict):
        data_dict = batch_dict['token']
        supervision_dict = batch_dict[self.box2d_supervision]
        if 'mv_imgs' in data_dict:
            img_metas = [{
                'image': data_dict['mv_imgs'],
                'img_shape': list(data_dict['mv_imgs'][i].shape[1:]) + [3],
                'pad_shape': list(data_dict['mv_imgs'][i].shape[1:]) + [3],
            } for i in range(len(data_dict['mv_imgs']))]
        else:
            img_metas = [{
                # for debug
                "image": (data_dict['left_img'][i] if 'left_img' in data_dict else data_dict['left_imgs'][i, -1]),
                "img_shape": list((data_dict['left_img'][i] if 'left_img' in data_dict else data_dict['left_imgs'][i, -1]).shape[1:3]) + [3],
                "pad_shape": list((data_dict['left_img'][i] if 'left_img' in data_dict else data_dict['left_imgs'][i, -1]).shape[1:3]) + [3]}
                for i in range(len((data_dict['left_img' if 'left_img' in data_dict else 'left_imgs'])))]

        gt_boxes_2d, gt_labels = torch.split(
            supervision_dict['gt_boxes_2d'], (4, 1), dim=2)
        gt_labels = gt_labels[..., 0].long()
        if self.use_3d_center:
            gt_boxes_2d = torch.cat(
                [gt_boxes_2d, supervision_dict['gt_centers_2d']], dim=-1)

        gt_boxes_2d = torch.unbind(gt_boxes_2d)
        gt_labels = torch.unbind(gt_labels - 1)  # a list of [N] tensors
        gt_bboxes_2d_ignore = torch.unbind(
            supervision_dict['gt_boxes_2d_ignored']) if 'gt_boxes_2d_ignored' in supervision_dict else None  # a list of [N, 4] tensors

        losses = self.bbox_head.forward_train(data_dict['sem_features'], img_metas, gt_boxes_2d,
                                              gt_labels, gt_bboxes_2d_ignore)

        for k, v in losses.items():
            if not isinstance(v, (list, tuple)) and len(v.shape) == 0:
                _sum_loss = v
            else:
                _sum_loss = sum(_loss for _loss in v)
            assert len(_sum_loss.shape) == 0
            losses[k] = _sum_loss.sum()
            tb_dict['rpn2d_' + k] = losses[k].item()
        loss_sum = sum([v for _, v in losses.items()])
        return loss_sum, tb_dict

    def forward(self, batch_dict):
        if not self.training and self.get_box2d_when_inference:
            data_dict = batch_dict['token']
            if 'mv_imgs' in data_dict:
                img_metas = [{
                    'image': data_dict['mv_imgs'],
                    'img_shape': list(data_dict['mv_imgs'][i].shape[1:]) + [3],
                    'pad_shape': list(data_dict['mv_imgs'][i].shape[1:]) + [3],
                    "scale_factor": 1.0,  # TODO: scale factor from dataset
                } for i in range(len(data_dict['mv_imgs']))]
            else:
                img_metas = [{
                    "img_shape": list((data_dict['left_img'][i] if 'left_img' in data_dict else data_dict['left_imgs'][i, -1]).shape[1:3]) + [3],
                    "pad_shape": list((data_dict['left_img'][i] if 'left_img' in data_dict else data_dict['left_imgs'][i, -1]).shape[1:3]) + [3],
                    "scale_factor": 1.0}  # TODO: scale factor from dataset
                    for i in range(len(data_dict['left_img' if 'left_img' in data_dict else 'left_imgs']))]

                outs = self.bbox_head(data_dict['sem_features'])
                data_dict['head_outs'] = outs

                try:
                    bbox_list = self.bbox_head.get_bboxes(
                        *outs, img_metas, rescale=False)  # TODO: rescale

                    bbox_results = [
                        bbox2result(det_bboxes, det_labels,
                                    self.bbox_head.num_classes)
                        for det_bboxes, det_labels in bbox_list
                    ]

                    data_dict['boxes_2d_pred'] = []
                    for bbox_result in bbox_results:
                        pred_dict = {}
                        pred_dict['pred_boxes_2d'] = np.concatenate(
                            [x[:, :-1] for x in bbox_result])
                        pred_dict['pred_scores_2d'] = np.concatenate(
                            [x[:, -1] for x in bbox_result])
                        pred_dict['pred_labels_2d'] = np.concatenate(
                            [[cls_id + 1] * len(x) for cls_id, x in enumerate(bbox_result)]).astype(np.int64)
                        data_dict['boxes_2d_pred'].append(pred_dict)
                except NotImplementedError:
                    print("not implemented get_bboxes, skip")

        return batch_dict

