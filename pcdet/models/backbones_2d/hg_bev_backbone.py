# Hourglass BEV backbone (same as DSGN. https://arxiv.org/abs/2001.03398)

import torch.nn as nn

from pcdet.models.backbones_3d_stereo.submodule import convbn, hourglass2d, hourglass2d_large_kernel


class HgBEVBackbone(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_channels = model_cfg.num_channels
        self.GN = model_cfg.GN
        hourglass_cfg = model_cfg.get('hourglass', None)
        self.rpn3d_conv2 = nn.Sequential(
            convbn(input_channels, self.num_channels, 3, 1, 1, 1, gn=self.GN),
            nn.ReLU(inplace=True))
        
        if hourglass_cfg is None:
            self.rpn3d_conv3 = hourglass2d(self.num_channels, gn=self.GN)
        else:
            if hourglass_cfg.get('use_large_kernel', False):
                self.rpn3d_conv3 = hourglass2d_large_kernel(self.num_channels, gn=self.GN, model_cfg=hourglass_cfg)
            else:
                self.rpn3d_conv3 = hourglass2d(self.num_channels, gn=self.GN, model_cfg=hourglass_cfg)
        self.num_bev_features = self.num_channels

    def forward(self, data_dict):
        spatial_features = data_dict['spatial_features']
        x = self.rpn3d_conv2(spatial_features)
        x = self.rpn3d_conv3(x, None, None)[0]
        data_dict['spatial_features_2d'] = x

        return data_dict
    

if __name__ == '__main__':
    import torch
    from easydict import EasyDict
    from thop import profile, clever_format
    hourglass = {
            'use_large_kernel': True, # False
            'kernel_list': [3,5,3,5],
            'stride_list': [2,1,2,1],
            'padding_list': [1,2,1,2],
            'dilation_list': [1,1,1,1]
    }
    # hourglass=None

    model_cfg = EasyDict(num_channels=64, GN=False, hourglass=hourglass)
    print(model_cfg)
    model = HgBEVBackbone(model_cfg, 96)
    print(model)
    indata = torch.randn(1, 96, 288, 256)
    indict = {'spatial_features': indata}
    flops, params = profile(model, inputs=(indict,))
    flops, params = clever_format([flops, params], "%.3f")
    print(f"计算量：{flops}")
    print(f"参数量：{params}")

    # import time
    # spend_time = []
    # model = model.cuda()
    # indict = {'spatial_features': indata.cuda()}
    # for i in range(50):
    #     start = time.time()
    #     outdata = model(indict)
    #     spend_time.append(time.time() - start)
    #     print(f"iter {i} time: {spend_time[-1]}")
    # print(f"average time: {sum(spend_time[1:]) / len(spend_time[1:])}")