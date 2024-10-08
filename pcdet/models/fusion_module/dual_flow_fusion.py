import torch
import torch.nn as nn

from pcdet.models.fusion_module.submodule import BaseConv


class SptialAttention(nn.Module):
    def __init__(self, model_cfg):
        super(SptialAttention, self).__init__()

        self.model_cfg = model_cfg
        self.use_deform = self.model_cfg.get('USE_DEFORM', False)

        padding = 3 if self.model_cfg.kernel_size == 7 else 1
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=self.model_cfg.kernel_size,
                                      stride=1, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [B, C, H, W]
        max_x = torch.max(x, dim=1, keepdim=True)[0]
        avg_x = torch.mean(x, dim=1, keepdim=True)
        x = torch.cat([max_x, avg_x], dim=1)
        x = self.spatial_conv(x)
        x = self.sigmoid(x)
        return x


class DualFlowFusion(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.GN = model_cfg.get('GN', False)
        self.model_cfg = model_cfg
        output_channels = self.model_cfg.get('OUTPUT_CHANNELS', input_channels)
        self.history_tag = self.model_cfg.get(
            'HISTORY_TAG', None)  # prev2, prev
        self.fusion_features_name = self.model_cfg.get(
            'HISTORY_FEATURES_NAME', None)
        assert self.history_tag is not None, 'You need to specify HISTORY_TAG!'
        assert self.fusion_features_name is not None, 'You need to specify HISTORY_FEATURES_NAME!'
        assert len(self.fusion_features_name) == 1, 'Only support one fusion feature!'

        self.use_deform = self.model_cfg.get('USE_DEFORM', False)
        if len(self.history_tag) == 1:
            self.token_branch_conv = nn.Sequential(
                BaseConv(input_channels,
                         output_channels // 2,
                         ksize=1,
                         stride=1,
                         gn=self.GN,
                         deform=self.use_deform))
        elif len(self.history_tag) == 2:
            self.token_branch_conv = nn.Sequential(
                BaseConv(input_channels,
                         output_channels // 2,
                         ksize=1,
                         stride=1,
                         gn=self.GN,
                         deform=self.use_deform))
            self.history_branch_conv = nn.Sequential(
                BaseConv(input_channels,
                         output_channels // (2 * len(self.history_tag)),
                         ksize=1,
                         stride=1,
                         gn=self.GN,
                         deform=self.use_deform))
        else:
            raise NotImplementedError

        # self.att_block = __all__[self.model_cfg.ATT_BLOCK.NAME](
        #     self.model_cfg.ATT_BLOCK.ATT_CFG) if self.model_cfg.ATT_BLOCK is not None else None

        self.attn_cfg = self.model_cfg.get('ATTN_CFG', None)
        if self.attn_cfg is not None:
            self.spatial_attn = SptialAttention(self.attn_cfg)

        self.use_kd = self.model_cfg.get('USE_KD', False)

        self.num_bev_features = output_channels

    def forward(self, batch_dict):
        # a list of prev2, prev features: [prev2, prev]
        # prev = (this_sample_idx, feature); feature dict: {name: feature tensor}
        history_features = batch_dict['history_features']
        if len(history_features) == 0:
            return batch_dict
        
        # name = self.fusion_features_name[0]
        # cur_features = batch_dict[name]
        # last_features = history_features[0][1][name]

        name = self.fusion_features_name[0]
        dynamic_flow_features = []
        if len(history_features) == 1:
            dynamic_flow_features.append(
                self.token_branch_conv(history_features[0][1][name]))
        elif len(history_features) == 2:
            for x in history_features:
                dynamic_flow_features.append(
                    self.history_branch_conv(x[1][name]))
        else:
            raise NotImplementedError
        
        cur_feature = batch_dict[name]
        dynamic_flow_features.append(
            self.token_branch_conv(cur_feature))
        dynamic_flow_features = torch.cat(dynamic_flow_features, dim=1)
        if self.use_kd is not None:
            batch_dict['student_feature'] = cur_feature
        # if self.att_block is not None:
        #     dynamic_flow_features = self.att_block(dynamic_flow_features)

        # static flow features fusion
        batch_dict[name] = dynamic_flow_features + batch_dict[name]
        return batch_dict


if __name__ == '__main__':
    from easydict import EasyDict
    model_cfg = EasyDict()
    model_cfg.HISTORY_TAG = ['prev']
    model_cfg.HISTORY_FEATURES_NAME = ['spatial_features']

    batch_dict = {}
    batch_dict['history_features'] = [
        (idx, {'spatial_features': torch.randn(1, 96, 288, 256).half().cuda()}) for idx in range(1)]
    batch_dict['spatial_features'] = torch.randn(1, 96, 288, 256).half().cuda()

    from thop import profile, clever_format
    model = DualFlowFusion(model_cfg, 96).half().cuda()
    flops, params = profile(model, inputs=(batch_dict,))
    flops, params = clever_format([flops, params], "%.3f")
    print(f"计算量：{flops}")
    print(f"参数量：{params}")

    import time
    spend_time = []
    for i in range(500):
        torch.cuda.synchronize()
        t1 = time.time()
        out_dict = model(batch_dict)
        torch.cuda.synchronize()
        t2 = time.time()
        print(t2-t1)
        spend_time.append(t2-t1)
        # out_dict.clear()

    print('total time: {}, avg time: {}'.format(
        sum(spend_time[1:]), sum(spend_time[1:]) / len(spend_time[1:])))