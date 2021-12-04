import numpy as np
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import fusion_strategy

import fusion_strategy

def _make_divisible(v, divisor, min_value=None):

    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def hard_sigmoid(x, inplace: bool = False):
    if inplace:
        return x.add_(3.).clamp_(0., 6.).div_(6.)
    else:
        return F.relu6(x + 3.) / 6.

class SqueezeExcite(nn.Module):
    def __init__(self, in_chs, se_ratio=0.25, reduced_base_chs=None,
                 act_layer=nn.ReLU, gate_fn=hard_sigmoid, divisor=4, **_):
        super(SqueezeExcite, self).__init__()
        self.gate_fn = gate_fn
        reduced_chs = _make_divisible((reduced_base_chs or in_chs) * se_ratio, divisor)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        self.act1 = act_layer(inplace=True)
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)

    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        x = x * self.gate_fn(x_se)
        return x

class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels*(ratio-1)

        self.primary_conv = nn.Sequential(
            nn.ReflectionPad2d(kernel_size//2),
            nn.Conv2d(inp, init_channels, kernel_size, stride, 0, bias=False),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.ReflectionPad2d(dw_size//2),
            nn.Conv2d(init_channels, new_channels, dw_size, 1, 0, groups=init_channels, bias=False),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )
        

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1,x2], dim=1)
        return out[:,:self.oup,:,:]


class GhostBottleneck(nn.Module):
    """ Ghost bottleneck w/ optional SE"""

    def __init__(self, in_chs, mid_chs, out_chs, dw_kernel_size=3,
                 stride=1, act_layer=nn.ReLU, se_ratio=0.):
        super(GhostBottleneck, self).__init__()
        has_se = se_ratio is not None and se_ratio > 0.
        self.stride = stride

        # Point-wise expansion
        self.ghost1 = GhostModule(in_chs, mid_chs, relu=True)

        # Squeeze-and-excitation
        if has_se:
            self.se = SqueezeExcite(mid_chs, se_ratio=se_ratio)
        else:
            self.se = None

        # Point-wise linear projection
        self.ghost2 = GhostModule(mid_chs, out_chs, relu=False)

        # shortcut
        if (in_chs == out_chs and self.stride == 1):
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.ReflectionPad2d((dw_kernel_size - 1) // 2),
                nn.Conv2d(in_chs, in_chs, dw_kernel_size, stride=stride,
                          padding=0, groups=in_chs, bias=False),
                nn.BatchNorm2d(in_chs),
                nn.Conv2d(in_chs, out_chs, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_chs),
            )

    def forward(self, x):
        residual = x

        # 1st ghost bottleneck
        x = self.ghost1(x)

        # Depth-wise convolution
        if self.stride > 1:
            x = self.conv_dw(x)
            x = self.bn_dw(x)

        # Squeeze-and-excitation
        if self.se is not None:
            x = self.se(x)

        # 2nd ghost bottleneck
        x = self.ghost2(x)

        x += self.shortcut(residual)
        x = torch.cat([x,residual],1);
        return x


# Dense Block unit
class DenseBlock(torch.nn.Module):
    def __init__(self, in_channels, kernel_size, stride):
        super(DenseBlock, self).__init__()
        out_channels_def = 16;
        denseblock = []
        denseblock += [GhostBottleneck(in_channels, in_channels*2, out_channels_def,kernel_size,1),
                       GhostBottleneck(in_channels*2, in_channels*2, out_channels_def, kernel_size, 1),
                       GhostBottleneck(in_channels*3, in_channels*2, out_channels_def, kernel_size, 1,se_ratio=0.25)]
        self.denseblock = nn.Sequential(*denseblock)

    def forward(self, x):
        out = self.denseblock(x)
        return out

# GhostFusion network
class GhostFusion_net(nn.Module):
    def __init__(self, input_nc=1, output_nc=1):
        super(GhostFusion_net, self).__init__()
        denseblock = DenseBlock
        nb_filter = [16, 64, 32, 16]
        kernel_size = 3
        stride = 1

        # encoder
        self.conv1 = GhostModule(input_nc, nb_filter[0], dw_size=kernel_size, stride=stride)
        self.DB1 = denseblock(nb_filter[0], kernel_size, stride)

        # decoder
        self.conv2 = GhostModule(nb_filter[1], nb_filter[1], dw_size=kernel_size, stride=stride)
        self.conv3 = GhostModule(nb_filter[1], nb_filter[2], dw_size=kernel_size, stride=stride)
        self.conv4 = GhostModule(nb_filter[2], nb_filter[3], dw_size=kernel_size, stride=stride)
        self.conv5 = GhostModule(nb_filter[3], output_nc, dw_size=kernel_size, stride=stride)
  
    def encoder(self, input):
        x1 = self.conv1(input)
        x_DB = self.DB1(x1)
        return [x_DB]

    def fusion(self, en1, en2, strategy_type='attention_weight'):
        # addition
        if strategy_type is 'L1':
         # attention weight
            fusion_function = fusion_strategy.L1Fusion
        elif (strategy_type is 'AVG'):
            fusion_function = fusion_strategy.AVGfusion
        elif (strategy_type is 'MAX'):
            fusion_function = fusion_strategy.MAXfusion
        elif (strategy_type is 'AGL1'):
            fusion_function = fusion_strategy.AGL1Fusion
        else:
            fusion_function = fusion_strategy.SCFusion

        f_0 = fusion_function(en1[0], en2[0])
        return [f_0]

    def decoder(self, f_en):
        x2 = self.conv2(f_en[0])
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        output = self.conv5(x4)

        return [output]
