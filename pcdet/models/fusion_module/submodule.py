# Partly reference from YOLOX
# https://github.com/Megvii-BaseDetection/YOLOX

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d


class SiLU(nn.Module):
    """export-friendly version of nn.SiLU()"""

    @staticmethod
    def forward(x):
        return x * F.sigmoid(x)


def get_activation(name="silu"):
    if name == "silu":
        module = SiLU()
    elif name == "relu":
        module = nn.ReLU()
    elif name == "lrelu":
        module = nn.LeakyReLU(0.1)
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module


class BaseConv(nn.Module):
    """A Conv2d -> Batchnorm -> silu/leaky relu block"""

    def __init__(self,
                 in_channels,
                 out_channels,
                 ksize,
                 stride,
                 padding=None,
                 dilation=1,
                 groups=1,
                 bias=False,
                 act="relu",
                 gn=False,
                 deform=False,
                 use_norm=True,
                 use_act=True,
                 ):
        super().__init__()
        # same padding
        if padding is None:
            padding = (ksize - 1) // 2
        if not deform:
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=ksize,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias)
        else:
            self.conv = DeformConv2d(
                in_channels,
                out_channels,
                kernel_size=ksize,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias)

        if gn and out_channels % 32 != 0:
            print('Cannot apply GN as the channels is not 32-divisible.')
            gn = False
        if use_norm:
            self.norm = nn.BatchNorm2d(
                out_channels) if not gn else nn.GroupNorm(32, out_channels)
        if use_act:
            self.act = get_activation(act)

    def forward(self, x):
        x = self.conv(x)
        if hasattr(self, "norm"):
            x = self.norm(x)
        if hasattr(self, "act"):
            x = self.act(x)
        return x

    def fuseforward(self, x):
        return self.act(self.conv(x))


class Focus(nn.Module):
    """Focus width and height information into channel space."""

    def __init__(self, in_channels, out_channels, ksize=1, stride=1, act="relu"):
        super().__init__()
        self.conv = BaseConv(in_channels * 4, out_channels,
                             ksize, stride, act=act)

    def forward(self, x):
        # shape of x (b,c,w,h) -> y(b,4c,w/2,h/2)
        patch_top_left = x[..., ::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = F.concat(
            (patch_top_left, patch_bot_left,
             patch_top_right, patch_bot_right,), axis=1,
        )
        return self.conv(x)
