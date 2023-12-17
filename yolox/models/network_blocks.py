#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class SiLU(nn.Module):
    """export-friendly version of nn.SiLU()"""

    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


def get_activation(name="silu", inplace=True):
    if name == "silu":
        module = nn.SiLU(inplace=inplace)
    elif name == "relu":
        module = nn.ReLU(inplace=inplace)
    elif name == "lrelu":
        module = nn.LeakyReLU(0.1, inplace=inplace)
    elif name == "hard_swish":
        module = nn.Hardswish(inplace=inplace)
    elif name == "":
        module = nn.Identity()
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module


def channel_shuffle(x, groups=2):  # shuffle channel
    # RESHAPE----->transpose------->Flatten
    B, C, H, W = x.size()
    out = x.view(B, groups, C // groups, H, W).permute(0, 2, 1, 3, 4).contiguous()
    out = out.view(B, C, H, W)
    return out


class BaseConv(nn.Module):
    """A Conv2d -> Batchnorm -> silu/leaky relu block"""

    def __init__(
            self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="silu"
    ):
        super().__init__()
        # same padding
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class DWConv(nn.Module):
    """Depthwise Conv + Conv"""

    def __init__(self, in_channels, out_channels, ksize, stride=1, act="silu"):
        super().__init__()
        self.dconv = BaseConv(
            in_channels,
            in_channels,
            ksize=ksize,
            stride=stride,
            groups=in_channels,
            act=act,
        )
        self.pconv = BaseConv(
            in_channels, out_channels, ksize=1, stride=1, groups=1, act=act
        )

    def forward(self, x):
        x = self.dconv(x)
        return self.pconv(x)


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(
            self,
            in_channels,
            out_channels,
            shortcut=True,
            expansion=0.5,
            depthwise=False,
            act="silu",
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        Conv = DWConv if depthwise else BaseConv
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = Conv(hidden_channels, out_channels, 3, stride=1, act=act)
        self.use_add = shortcut and in_channels == out_channels

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        if self.use_add:
            y = y + x
        return y


class ResLayer(nn.Module):
    """Residual layer with `in_channels` inputs."""

    def __init__(self, in_channels: int):
        super().__init__()
        mid_channels = in_channels // 2
        self.layer1 = BaseConv(
            in_channels, mid_channels, ksize=1, stride=1, act="lrelu"
        )
        self.layer2 = BaseConv(
            mid_channels, in_channels, ksize=3, stride=1, act="lrelu"
        )

    def forward(self, x):
        out = self.layer2(self.layer1(x))
        return x + out


class SPPBottleneck(nn.Module):
    """Spatial pyramid pooling layer used in YOLOv3-SPP"""

    def __init__(
            self, in_channels, out_channels, kernel_sizes=(5, 9, 13), activation="silu"
    ):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=activation)
        self.m = nn.ModuleList(
            [
                nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2)
                for ks in kernel_sizes
            ]
        )
        conv2_channels = hidden_channels * (len(kernel_sizes) + 1)
        self.conv2 = BaseConv(conv2_channels, out_channels, 1, stride=1, act=activation)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat([x] + [m(x) for m in self.m], dim=1)
        x = self.conv2(x)
        return x


class SPPF(nn.Module):
    def __init__(
            self, in_channels, out_channels, k=5, activation="silu"
    ):
        super(SPPF, self).__init__()
        hidden_channels = in_channels // 2
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=activation)
        self.conv2 = BaseConv(hidden_channels * 4, out_channels, 1, stride=1, act=activation)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)  # 窗口大小为5的maxpool

    def forward(self, x):
        x = self.conv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.conv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class CSPLayer(nn.Module):
    """C3 in yolov5, CSP Bottleneck with 3 convolutions"""

    def __init__(
            self,
            in_channels,
            out_channels,
            n=1,
            shortcut=True,
            expansion=0.5,
            depthwise=False,
            act="silu",
    ):
        """
        Args:
            in_channels (int): input channels.
            out_channels (int): output channels.
            n (int): number of Bottlenecks. Default value: 1.
        """
        # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        hidden_channels = int(out_channels * expansion)  # hidden channels
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv3 = BaseConv(2 * hidden_channels, out_channels, 1, stride=1, act=act)
        module_list = [
            Bottleneck(
                hidden_channels, hidden_channels, shortcut, 1.0, depthwise, act=act
            )
            for _ in range(n)
        ]
        self.m = nn.Sequential(*module_list)

    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x_1 = self.m(x_1)
        x = torch.cat((x_1, x_2), dim=1)
        return self.conv3(x)


class Focus(nn.Module):
    """Focus width and height information into channel space."""

    def __init__(self, in_channels, out_channels, ksize=1, stride=1, act="silu"):
        super().__init__()
        self.conv = BaseConv(in_channels * 4, out_channels, ksize, stride, act=act)

    def forward(self, x):
        # shape of x (b,c,w,h) -> y (b,4c,w/2,h/2)
        patch_top_left = x[..., ::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = torch.cat(
            (
                patch_top_left,
                patch_bot_left,
                patch_top_right,
                patch_bot_right,
            ),
            dim=1,
        )
        return self.conv(x)


# -----------------------some new module---------------------------------------------
class ChannelAttention(nn.Module):
    # Channel-attention module https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc1/configs/rtmdet
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.act(self.fc(self.pool(x)))


class SpatialAttention(nn.Module):
    # Spatial-attention module
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        return x * self.act(self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)))


class CBAM(nn.Module):
    # Convolutional Block Attention Module
    def __init__(self, c1, kernel_size=7):  # ch_in, kernels
        super().__init__()
        self.channel_attention = ChannelAttention(c1)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        return self.spatial_attention(self.channel_attention(x))


# --------------------build shuffleNet----------------------------------
def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


class conv_bn_relu_maxpool(nn.Module):
    def __init__(self, c1, c2, act="silu"):  # ch_in, ch_out
        super(conv_bn_relu_maxpool, self).__init__()

        self.conv = BaseConv(c1, c2, ksize=3, stride=2, act=act)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

    def forward(self, x):
        return self.maxpool(self.conv(x))


class Shuffle_Block(nn.Module):
    """
    将relu激活函数修改为了silu激活函数
    """

    def __init__(self, inp, oup, stride):
        super(Shuffle_Block, self).__init__()

        if not (1 <= stride <= 3):
            raise ValueError('illegal stride value')
        self.stride = stride
        branch_features = oup // 2
        assert (self.stride != 1) or (inp == branch_features << 1)

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                self.depthwise_conv(inp, inp, kernel_size=3, stride=self.stride, padding=1),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.SiLU(inplace=True),
            )

        self.branch2 = nn.Sequential(
            nn.Conv2d(inp if (self.stride > 1) else branch_features,
                      branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.SiLU(inplace=True),
            self.depthwise_conv(branch_features, branch_features, kernel_size=3, stride=self.stride, padding=1),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.SiLU(inplace=True),
        )

    @staticmethod
    def depthwise_conv(i, o, kernel_size, stride=1, padding=0, bias=False):
        return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias, groups=i)

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = channel_shuffle(out, 2)

        return out


# ———————————————————————————shuffle block end——————————————————————————
# ---------------------build enhanced shuffle block-------------------------
class GhostConv(nn.Module):
    # Ghost Convolution https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k=3, s=1, g=1, act="hard_swish"):  # ch_in, ch_out, kernel, stride, groups
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = BaseConv(c1, c_, 1, s, g, act=act)
        self.cv2 = BaseConv(c_, c_, k, s, c_, act=act)

    def forward(self, x):
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), dim=1)


class ES_SEModule(nn.Module):
    def __init__(self, channel, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(
            in_channels=channel,
            out_channels=channel // reduction,
            kernel_size=1,
            stride=1,
            padding=0)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(
            in_channels=channel // reduction,
            out_channels=channel,
            kernel_size=1,
            stride=1,
            padding=0)
        self.hardsigmoid = nn.Hardsigmoid()

    def forward(self, x):
        identity = x
        x = self.avg_pool(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.hardsigmoid(x)
        out = identity * x
        return out


class ES_Block(nn.Module):
    def __init__(self, c1, c2, k_sizes=(5, 9, 13), SE=False, act="hard_swish"):
        super(ES_Block, self).__init__()
        branch_channels = c1 // 2
        hidden_channels = branch_channels // (len(k_sizes) + 1)

        self.conv1 = BaseConv(branch_channels, hidden_channels, 1, 1, act=act)
        self.m = nn.ModuleList()
        for k_size in k_sizes:
            self.m.append(
                BaseConv(hidden_channels, hidden_channels, k_size, 1, groups=hidden_channels, act=act)
            )
        self.se = None
        if SE:
            self.se = ES_SEModule(branch_channels)
        self.pw = BaseConv(branch_channels, branch_channels, 1, 1, act=act)

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        x1 = self.conv1(x1)
        x1 = torch.cat([x1] + [m(x1) for m in self.m], dim=1)
        if self.se is not None:
            x1 = self.se(x1)
        x1 = self.pw(x1)
        x = torch.cat((x1, x2), dim=1)
        x = channel_shuffle(x, 2)
        return x


# --------------------------enhanced shuffle block end-------------------------------
# --------------------------build rep_vgg block-----------------------------------------
def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                        kernel_size=kernel_size, stride=stride, padding=padding, groups=groups,
                                        bias=False))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))

    return result


class SEBlock(nn.Module):

    def __init__(self, input_channels, internal_neurons):
        super(SEBlock, self).__init__()
        self.down = nn.Conv2d(in_channels=input_channels, out_channels=internal_neurons, kernel_size=1, stride=1,
                              bias=True)
        self.up = nn.Conv2d(in_channels=internal_neurons, out_channels=input_channels, kernel_size=1, stride=1,
                            bias=True)
        self.input_channels = input_channels

    def forward(self, inputs):
        x = F.avg_pool2d(inputs, kernel_size=inputs.size(3))
        x = self.down(x)
        x = F.relu(x)
        x = self.up(x)
        x = torch.sigmoid(x)
        x = x.view(-1, self.input_channels, 1, 1)
        return inputs * x


class RepVGGBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1, dilation=1, groups=1, padding_mode='zeros', deploy=False, use_se=False,
                 act="hard_swish"):
        super(RepVGGBlock, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels

        padding_11 = padding - kernel_size // 2

        self.nonlinearity = get_activation(act, inplace=True)  # act

        if use_se:
            self.se = SEBlock(out_channels, internal_neurons=out_channels // 16)
        else:
            self.se = nn.Identity()

        if deploy:  # 部署时只生成一个卷积算子
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding, dilation=dilation, groups=groups, bias=True,
                                         padding_mode=padding_mode)

        else:
            self.rbr_identity = nn.BatchNorm2d(  # BN
                num_features=in_channels) if out_channels == in_channels and stride == 1 else None
            self.rbr_dense = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                     stride=stride, padding=padding, groups=groups)  # 3 * 3Conv
            self.rbr_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride,
                                   padding=padding_11, groups=groups)  # 1 * 1Conv

    def get_equivalent_kernel_bias(self):  # 融合所有的卷积 和 恒等映射
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):  # 将1 * 1卷积扩展为3 * 3卷积
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):  # 融合conv和bn，返回新的conv权重和conv偏置
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):  # 融合conv + bn
            kernel = branch.conv.weight  # 获取卷积算子的去权重
            running_mean = branch.bn.running_mean  # 获取BN层的样本均值
            running_var = branch.bn.running_var  # 获取BN层的样本方差
            gamma = branch.bn.weight  # 获取BN层的样本gamma
            beta = branch.bn.bias  #
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)  # 将bn转换成3 * 3卷积
            if not hasattr(self, 'id_tensor'):  #
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3),
                                        dtype=np.float32)  # 假设它是一个3 * 3大小的卷积核，总共有in_channels个卷积核，每个卷积核的通道数为input_dim
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1  # 将所有的卷积核的，只在一个channel上将中间位置置为1.其余还是0
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor  # 等价于权重为1的 1 * 1大小的卷积
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)  # 给卷积核乘上的权重
        return kernel * t, beta - running_mean * gamma / std  # 返回新的权重和偏置，注意原卷积核的bias应该为0

    def forward(self, inputs):
        if hasattr(self, 'rbr_reparam'):  # 重参化
            return self.nonlinearity(self.se(self.rbr_reparam(inputs)))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.nonlinearity(self.se(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out))

    def fusevggforward(self, x):
        return self.nonlinearity(self.rbr_dense(x))


# -------------------------rep_vgg block end------------------------------------------
# -------------------------build DBB------------------------------------------
class BNAndPadLayer(nn.Module):
    def __init__(self,
                 pad_pixels,
                 num_features,
                 eps=1e-5,
                 momentum=0.1,
                 affine=True,
                 track_running_stats=True):
        super(BNAndPadLayer, self).__init__()
        self.bn = nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats)
        self.pad_pixels = pad_pixels

    def forward(self, input):
        output = self.bn(input)
        if self.pad_pixels > 0:
            if self.bn.affine:
                pad_values = self.bn.bias.detach() - self.bn.running_mean * self.bn.weight.detach() / torch.sqrt(
                    self.bn.running_var + self.bn.eps)
            else:
                pad_values = - self.bn.running_mean / torch.sqrt(self.bn.running_var + self.bn.eps)
            output = F.pad(output, [self.pad_pixels] * 4)
            pad_values = pad_values.view(1, -1, 1, 1)
            output[:, :, 0:self.pad_pixels, :] = pad_values
            output[:, :, -self.pad_pixels:, :] = pad_values
            output[:, :, :, 0:self.pad_pixels] = pad_values
            output[:, :, :, -self.pad_pixels:] = pad_values
        return output

    @property
    def weight(self):
        return self.bn.weight

    @property
    def bias(self):
        return self.bn.bias

    @property
    def running_mean(self):
        return self.bn.running_mean

    @property
    def running_var(self):
        return self.bn.running_var

    @property
    def eps(self):
        return self.bn.eps


def transI_fusebn(kernel, bn):
    gamma = bn.weight
    std = (bn.running_var + bn.eps).sqrt()
    return kernel * ((gamma / std).reshape(-1, 1, 1, 1)), bn.bias - bn.running_mean * gamma / std


def transIII_1x1_kxk(k1, b1, k2, b2, groups):
    if groups == 1:
        k = F.conv2d(k2, k1.permute(1, 0, 2, 3))  # k1 (c2,c1,x,x) k2(c3,c2,x,x)
        b_hat = (k2 * b1.reshape(1, -1, 1, 1)).sum((1, 2, 3))
    else:
        k_slices = []
        b_slices = []
        k1_T = k1.permute(1, 0, 2, 3)
        k1_group_width = k1.size(0) // groups
        k2_group_width = k2.size(0) // groups
        for g in range(groups):
            k1_T_slice = k1_T[:, g * k1_group_width:(g + 1) * k1_group_width, :, :]
            k2_slice = k2[g * k2_group_width:(g + 1) * k2_group_width, :, :, :]
            k_slices.append(F.conv2d(k2_slice, k1_T_slice))
            b_slices.append(
                (k2_slice * b1[g * k1_group_width:(g + 1) * k1_group_width].reshape(1, -1, 1, 1)).sum((1, 2, 3)))
        k, b_hat = transIV_depthconcat(k_slices, b_slices)
    return k, b_hat + b2


def transIV_depthconcat(kernels, biases):
    return torch.cat(kernels, dim=0), torch.cat(biases)


def transVI_multiscale(kernel):
    if kernel is None:
        return 0
    return torch.nn.functional.pad(kernel, [1, 1, 1, 1])


# 将avg转换为conv算子
def transV_avg(channels, kernel_size, groups):
    input_dim = channels // groups
    k = torch.zeros((channels, input_dim, kernel_size, kernel_size))  # (c2, c2, k, k)
    k[np.arange(channels), np.tile(np.arange(input_dim), groups), :, :] = 1.0 / kernel_size ** 2
    return k


def transII_addbranch(kernels, biases):
    return sum(kernels), sum(biases)


class DiverseBranchBlock(nn.Module):
    def __init__(self, c1, c2, k_size, s=1, p=1, g=1,
                 deploy=False, act=None, single_init=False):
        super(DiverseBranchBlock, self).__init__()
        self.deploy = deploy
        self.groups = g
        self.out_channels = c2
        self.kernel_size = k_size
        self.nonlinear = nn.Identity() if act is None else get_activation(act)

        if deploy:  # 重参后的dbb算子
            self.dbb_reparam = nn.Conv2d(c1, c2, k_size, s, padding=p, groups=g, bias=True)

        # 3*3 + bn分支
        self.dbb_origin = conv_bn(in_channels=c1, out_channels=c2, kernel_size=k_size,
                                  stride=s, padding=p, groups=g)
        # 1*1 + bn分支
        self.dbb_1x1 = conv_bn(in_channels=c1, out_channels=c2, kernel_size=1, stride=1,
                               padding=0, groups=g)
        # 1*1 + bn + pad + avg + bn
        self.dbb_avg = nn.Sequential()
        self.dbb_avg.add_module('conv', nn.Conv2d(in_channels=c1, out_channels=c2, kernel_size=1,
                                                  stride=1, padding=0, groups=g, bias=False))
        self.dbb_avg.add_module('bn', BNAndPadLayer(pad_pixels=p, num_features=c2))  # 进行BN和pad
        self.dbb_avg.add_module('avg', nn.AvgPool2d(kernel_size=k_size, stride=s, padding=0))
        self.dbb_avg.add_module('avgbn', nn.BatchNorm2d(c2))

        # 1*1 + bn + pad + 3*3 + bn
        self.dbb_1x1_kxk = nn.Sequential()
        self.dbb_1x1_kxk.add_module('conv1', nn.Conv2d(in_channels=c1, out_channels=c1,
                                                       kernel_size=1, stride=1, padding=0, groups=g, bias=False))
        self.dbb_1x1_kxk.add_module('bn1', BNAndPadLayer(pad_pixels=p, num_features=c1,
                                                         affine=True))
        self.dbb_1x1_kxk.add_module('conv2',
                                    nn.Conv2d(in_channels=c1, out_channels=c2,
                                              kernel_size=k_size, stride=s, padding=0, groups=g,
                                              bias=False))
        self.dbb_1x1_kxk.add_module('bn2', nn.BatchNorm2d(c2))
        if single_init:
            #   Initialize the bn.weight of dbb_origin as 1 and others as 0. This is not the default setting.
            self.single_init()

    def init_gamma(self, gamma_value):
        if hasattr(self, "dbb_origin"):
            torch.nn.init.constant_(self.dbb_origin.bn.weight, gamma_value)
        if hasattr(self, "dbb_1x1"):
            torch.nn.init.constant_(self.dbb_1x1.bn.weight, gamma_value)
        if hasattr(self, "dbb_avg"):
            torch.nn.init.constant_(self.dbb_avg.avgbn.weight, gamma_value)
        if hasattr(self, "dbb_1x1_kxk"):
            torch.nn.init.constant_(self.dbb_1x1_kxk.bn2.weight, gamma_value)

    def single_init(self):
        self.init_gamma(0.0)
        if hasattr(self, "dbb_origin"):
            torch.nn.init.constant_(self.dbb_origin.bn.weight, 1.0)

    def forward(self, inputs):

        if hasattr(self, 'dbb_reparam'):
            return self.nonlinear(self.dbb_reparam(inputs))

        out = self.dbb_origin(inputs)
        if hasattr(self, 'dbb_1x1'):
            out += self.dbb_1x1(inputs)
        out += self.dbb_avg(inputs)
        out += self.dbb_1x1_kxk(inputs)
        return self.nonlinear(out)

    def get_equivalent_kernel_bias(self):  # 获取重参后的权重 weight + bias
        # 融合 3 * 3 + bn
        k_origin, b_origin = transI_fusebn(self.dbb_origin.conv.weight, self.dbb_origin.bn)

        # 融合 1 * 1 + bn
        if hasattr(self, 'dbb_1x1'):
            k_1x1, b_1x1 = transI_fusebn(self.dbb_1x1.conv.weight, self.dbb_1x1.bn)
            k_1x1 = transVI_multiscale(k_1x1)  # 将1*1卷积核转换成3*3卷积核
        else:
            k_1x1, b_1x1 = 0, 0

        # 融合dbb_1x1_kxk分支
        k_1x1_kxk_first, b_1x1_kxk_first = transI_fusebn(self.dbb_1x1_kxk.conv1.weight, self.dbb_1x1_kxk.bn1)
        k_1x1_kxk_second, b_1x1_kxk_second = transI_fusebn(self.dbb_1x1_kxk.conv2.weight, self.dbb_1x1_kxk.bn2)
        k_1x1_kxk_merged, b_1x1_kxk_merged = transIII_1x1_kxk(k_1x1_kxk_first, b_1x1_kxk_first, k_1x1_kxk_second,
                                                              b_1x1_kxk_second, groups=self.groups)

        # 融合dbb_avg分支
        k_avg = transV_avg(self.out_channels, self.kernel_size, self.groups)  # 得到一个avg的conv算子
        k_1x1_avg_second, b_1x1_avg_second = transI_fusebn(k_avg.to(self.dbb_avg.avgbn.weight.device),
                                                           self.dbb_avg.avgbn)
        if hasattr(self.dbb_avg, 'conv'):
            k_1x1_avg_first, b_1x1_avg_first = transI_fusebn(self.dbb_avg.conv.weight, self.dbb_avg.bn)
            k_1x1_avg_merged, b_1x1_avg_merged = transIII_1x1_kxk(k_1x1_avg_first, b_1x1_avg_first, k_1x1_avg_second,
                                                                  b_1x1_avg_second, groups=self.groups)
        else:
            k_1x1_avg_merged, b_1x1_avg_merged = k_1x1_avg_second, b_1x1_avg_second

        return transII_addbranch((k_origin, k_1x1, k_1x1_kxk_merged, k_1x1_avg_merged),
                                 (b_origin, b_1x1, b_1x1_kxk_merged, b_1x1_avg_merged))

    def fuseforward(self, x):
        return self.nonlinear(self.dbb_origin(x))


class ES_DBB(nn.Module):
    def __init__(self, channels, k_size, SE=False, act="hard_swish"):
        super(ES_DBB, self).__init__()
        branch_channels = channels // 2
        self.conv1 = BaseConv(branch_channels, branch_channels, 1, 1, act=act)
        padding = k_size // 2
        self.dbb = DiverseBranchBlock(branch_channels, branch_channels, k_size, 1, p=padding, act=act)

        self.se = None
        if SE:
            self.se = ES_SEModule(branch_channels)

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        x2 = self.conv1(self.dbb(x2))
        if self.se is not None:
            x2 = self.se(x2)
        x = torch.cat((x1, x2), dim=1)
        x = channel_shuffle(x, 2)
        return x
