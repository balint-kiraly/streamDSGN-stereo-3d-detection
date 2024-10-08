import torch
import torch.nn as nn

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math

from pcdet.models.fusion_module.submodule import BaseConv


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LKA(nn.Module):
    def __init__(self, dim, k_size):
        super().__init__()

        if k_size == 7:
            self.conv0 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
            self.conv_spatial = nn.Conv2d(
                dim, dim, 3, stride=1, padding=2, groups=dim, dilation=2)
        elif k_size == 11:
            self.conv0 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
            self.conv_spatial = nn.Conv2d(
                dim, dim, 5, stride=1, padding=4, groups=dim, dilation=2)
        elif k_size == 23:
            self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
            self.conv_spatial = nn.Conv2d(
                dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        elif k_size == 35:
            self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
            self.conv_spatial = nn.Conv2d(
                dim, dim, 11, stride=1, padding=15, groups=dim, dilation=3)
        elif k_size == 41:
            self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
            self.conv_spatial = nn.Conv2d(
                dim, dim, 13, stride=1, padding=18, groups=dim, dilation=3)  
        elif k_size == 53:
            self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
            self.conv_spatial = nn.Conv2d(
                dim, dim, 17, stride=1, padding=24, groups=dim, dilation=3)

        # self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        # self.conv_spatial = nn.Conv2d(
        #     dim, dim, k_size, stride=1, padding=2, groups=dim, dilation=2)
        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)
        return u * attn


class LSKA(nn.Module):
    def __init__(self, dim, k_size):
        super().__init__()

        self.k_size = k_size

        if k_size == 7:
            self.conv0h = nn.Conv2d(dim, dim, kernel_size=(
                1, 3), stride=(1, 1), padding=(0, (3-1)//2), groups=dim)
            self.conv0v = nn.Conv2d(dim, dim, kernel_size=(
                3, 1), stride=(1, 1), padding=((3-1)//2, 0), groups=dim)
            self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(
                1, 3), stride=(1, 1), padding=(0, 2), groups=dim, dilation=2)
            self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(
                3, 1), stride=(1, 1), padding=(2, 0), groups=dim, dilation=2)
        elif k_size == 11:
            self.conv0h = nn.Conv2d(dim, dim, kernel_size=(
                1, 3), stride=(1, 1), padding=(0, (3-1)//2), groups=dim)
            self.conv0v = nn.Conv2d(dim, dim, kernel_size=(
                3, 1), stride=(1, 1), padding=((3-1)//2, 0), groups=dim)
            self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(
                1, 5), stride=(1, 1), padding=(0, 4), groups=dim, dilation=2)
            self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(
                5, 1), stride=(1, 1), padding=(4, 0), groups=dim, dilation=2)
        elif k_size == 23:
            self.conv0h = nn.Conv2d(dim, dim, kernel_size=(
                1, 5), stride=(1, 1), padding=(0, (5-1)//2), groups=dim)
            self.conv0v = nn.Conv2d(dim, dim, kernel_size=(
                5, 1), stride=(1, 1), padding=((5-1)//2, 0), groups=dim)
            self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(
                1, 7), stride=(1, 1), padding=(0, 9), groups=dim, dilation=3)
            self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(
                7, 1), stride=(1, 1), padding=(9, 0), groups=dim, dilation=3)
        elif k_size == 35:
            self.conv0h = nn.Conv2d(dim, dim, kernel_size=(
                1, 5), stride=(1, 1), padding=(0, (5-1)//2), groups=dim)
            self.conv0v = nn.Conv2d(dim, dim, kernel_size=(
                5, 1), stride=(1, 1), padding=((5-1)//2, 0), groups=dim)
            self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(
                1, 11), stride=(1, 1), padding=(0, 15), groups=dim, dilation=3)
            self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(
                11, 1), stride=(1, 1), padding=(15, 0), groups=dim, dilation=3)
        elif k_size == 41:
            self.conv0h = nn.Conv2d(dim, dim, kernel_size=(
                1, 5), stride=(1, 1), padding=(0, (5-1)//2), groups=dim)
            self.conv0v = nn.Conv2d(dim, dim, kernel_size=(
                5, 1), stride=(1, 1), padding=((5-1)//2, 0), groups=dim)
            self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(
                1, 13), stride=(1, 1), padding=(0, 18), groups=dim, dilation=3)
            self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(
                13, 1), stride=(1, 1), padding=(18, 0), groups=dim, dilation=3)
        elif k_size == 53:
            self.conv0h = nn.Conv2d(dim, dim, kernel_size=(
                1, 5), stride=(1, 1), padding=(0, (5-1)//2), groups=dim)
            self.conv0v = nn.Conv2d(dim, dim, kernel_size=(
                5, 1), stride=(1, 1), padding=((5-1)//2, 0), groups=dim)
            self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(
                1, 17), stride=(1, 1), padding=(0, 24), groups=dim, dilation=3)
            self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(
                17, 1), stride=(1, 1), padding=(24, 0), groups=dim, dilation=3)

        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0h(x)
        attn = self.conv0v(attn)
        attn = self.conv_spatial_h(attn)
        attn = self.conv_spatial_v(attn)
        attn = self.conv1(attn)
        return u * attn


class Attention(nn.Module):
    def __init__(self, d_model, k_size, attn_type='LSKA'):
        super().__init__()

        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        if attn_type == 'LSKA':
            self.spatial_gating_unit = LSKA(d_model, k_size)
        elif attn_type == 'LKA':
            self.spatial_gating_unit = LKA(d_model, k_size)
        else:
            raise NotImplementedError
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x


class Block(nn.Module):
    def __init__(self, dim, k_size, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU, attn_type='LSKA'):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.attn = Attention(dim, k_size, attn_type)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = nn.BatchNorm2d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)
        layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1)
                               * self.attn(self.norm1(x)))
        x = x + \
            self.drop_path(
                self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x)))
        return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.BatchNorm2d(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = self.norm(x)
        return x, H, W


class VANBackbone(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg
        self.GN = model_cfg.get('GN', False)
        self.k_size = model_cfg.get('K_SIZE', 7)
        self.depths = model_cfg.get('DEPTHS', [3, 3])
        self.embed_dims = model_cfg.get('EMBED_DIMS', [128, 128])
        self.num_stages = len(self.embed_dims)
        self.mlp_ratios = model_cfg.get('MLP_RATIOS', [3, 4])
        self.drop_rate = model_cfg.get('DROP_RATE', 0.)
        self.drop_path_rate = model_cfg.get('DROP_PATH_RATE', 0.)
        self.num_channels = model_cfg.get('NUM_CHANNELS', 64)
        self.upsample_strides = model_cfg.get('UPSAMPLE_STRIDES', [2, 2])
        self.num_upsample_filters = model_cfg.get(
            'NUM_UPSAMPLE_FILTERS', [128, 64])
        self.attn_type = model_cfg.get('ATTN_TYPE', 'LSKA')

        self.multi_fusion_type = model_cfg.get('MULTI_FUSION_TYPE', 'SUM')
        if self.multi_fusion_type == 'CAT':
            self.num_bev_features = sum(self.num_upsample_filters)
        elif self.multi_fusion_type == 'SUM':
            self.num_bev_features = self.num_upsample_filters[-1]

        # build networks
        self.stem = BaseConv(input_channels, self.num_channels,
                             ksize=1, stride=1, gn=self.GN)

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate,
                                                sum(self.depths))]  # stochastic depth decay rule
        cur = 0
        for i in range(self.num_stages):
            patch_embed = OverlapPatchEmbed(patch_size=3 if i == 0 else 3,
                                            stride=2 if i == 0 else 2,
                                            in_chans=self.num_channels if i == 0 else self.embed_dims[
                                                i - 1],
                                            embed_dim=self.embed_dims[i])
            block = nn.ModuleList([Block(dim=self.embed_dims[i], k_size=self.k_size, mlp_ratio=self.mlp_ratios[i],
                                  drop=self.drop_rate, drop_path=dpr[cur + j], attn_type=self.attn_type) for j in range(self.depths[i])])
            norm = nn.LayerNorm(self.embed_dims[i])
            cur += self.depths[i]
            up_block = nn.Sequential(
                nn.ConvTranspose2d(self.embed_dims[i],
                                   self.num_upsample_filters[i],
                                   kernel_size=self.upsample_strides[i],
                                   stride=self.upsample_strides[i],
                                   bias=False),
                nn.BatchNorm2d(
                    self.num_upsample_filters[i]) if not self.GN else nn.GroupNorm(32, self.num_upsample_filters[i]),
                nn.ReLU())

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)
            setattr(self, f"up_block{i + 1}", up_block)

    def forward_features(self, x):
        B = x.shape[0]
        outs = []
        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            up_block = getattr(self, f"up_block{i + 1}")
            x, H, W = patch_embed(x)
            for blk in block:
                x = blk(x)
            x = x.flatten(2).transpose(1, 2)
            x = norm(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x)

        if self.multi_fusion_type == 'CAT':
            for i in range(self.num_stages):
                up_block = getattr(self, f"up_block{i + 1}")
                outs[i] = up_block(outs[i])
            ret_feat = torch.cat(outs, dim=1)

        elif self.multi_fusion_type == 'SUM':
            outs.reverse()
            x_up = None
            for i in range(self.num_stages):
                up_block = getattr(self, f"up_block{i + 1}")
                if x_up is not None:
                    outs[i] += x_up
                x_up = up_block(outs[i])
            ret_feat = x_up
        return ret_feat

    def forward(self, data_dict):
        spatial_features = data_dict['spatial_features']
        x = self.stem(spatial_features)
        x = self.forward_features(x)
        data_dict['spatial_features_2d'] = x
        return data_dict


if __name__ == '__main__':
    from easydict import EasyDict
    from thop import profile, clever_format

    model_cfg = EasyDict(
        NUM_CHANNELS=64,
        GN=False,
        K_SIZE=53,
        DEPTHS=[2, 2],
        EMBED_DIMS=[128, 128],
        MLP_RATIOS=[2, 2],
        DROP_RATE=0.,
        DROP_PATH_RATE=0.,
        UPSAMPLE_STRIDES=[2, 2],
        NUM_UPSAMPLE_FILTERS=[128, 64],
        MULTI_FUSION_TYPE='SUM',
        ATTN_TYPE='LKA'
    )
    model = VANBackbone(model_cfg, 96)
    print(model)
    indata = torch.randn(1, 96, 288, 256)
    indict = {'spatial_features': indata}
    flops, params = profile(model, inputs=(indict,))
    flops, params = clever_format([flops, params], "%.3f")
    print(f"计算量：{flops}")
    print(f"参数量：{params}")

    outdata = model(indict)

    import time
    spend_time = []
    model = model.cuda()
    indict = {'spatial_features': indata.cuda()}
    for i in range(50):
        start = time.time()
        outdata = model(indict)
        spend_time.append(time.time() - start)
        print(f"iter {i} time: {spend_time[-1]}")
    print(f"average time: {sum(spend_time[1:]) / len(spend_time[1:])}")


