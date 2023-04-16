#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import torch
import torch.nn as nn

from .darknet import CSPDarknet, CSPDarknet_Ghost, ShuffleNet
from .network_blocks import BaseConv, CSPLayer, DWConv, CBAM, GhostConv, C3Ghost


class YOLOPAFPN(nn.Module):
    """
    YOLOv3 model. Darknet 53 is the default backbone of this model.
    """

    def __init__(
            self,
            depth=1.0,
            width=1.0,
            in_features=("dark3", "dark4", "dark5"),
            in_channels=[256, 512, 1024],
            depthwise=False,
            act="silu",
    ):
        super().__init__()
        self.backbone = CSPDarknet(depth, width, depthwise=depthwise, act=act)
        self.in_features = in_features
        self.in_channels = in_channels
        Conv = DWConv if depthwise else BaseConv

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.lateral_conv0 = BaseConv(
            int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act
        )
        self.C3_p4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )  # cat

        self.reduce_conv1 = BaseConv(
            int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act
        )
        self.C3_p3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[0] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv2 = Conv(
            int(in_channels[0] * width), int(in_channels[0] * width), 3, 2, act=act
        )
        self.C3_n3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv1 = Conv(
            int(in_channels[1] * width), int(in_channels[1] * width), 3, 2, act=act
        )
        self.C3_n4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[2] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

    def forward(self, input):
        """
        Args:
            inputs: input images.

        Returns:
            Tuple[Tensor]: FPN feature.
        """

        #  backbone
        out_features = self.backbone(input)
        features = [out_features[f] for f in self.in_features]
        [x2, x1, x0] = features

        fpn_out0 = self.lateral_conv0(x0)  # 1024->512/32
        f_out0 = self.upsample(fpn_out0)  # 512/16
        f_out0 = torch.cat([f_out0, x1], 1)  # 512->1024/16
        f_out0 = self.C3_p4(f_out0)  # 1024->512/16

        fpn_out1 = self.reduce_conv1(f_out0)  # 512->256/16
        f_out1 = self.upsample(fpn_out1)  # 256/8
        f_out1 = torch.cat([f_out1, x2], 1)  # 256->512/8
        pan_out2 = self.C3_p3(f_out1)  # 512->256/8

        p_out1 = self.bu_conv2(pan_out2)  # 256->256/16
        p_out1 = torch.cat([p_out1, fpn_out1], 1)  # 256->512/16
        pan_out1 = self.C3_n3(p_out1)  # 512->512/16

        p_out0 = self.bu_conv1(pan_out1)  # 512->512/32
        p_out0 = torch.cat([p_out0, fpn_out0], 1)  # 512->1024/32
        pan_out0 = self.C3_n4(p_out0)  # 1024->1024/32

        outputs = (pan_out2, pan_out1, pan_out0)
        return outputs


class YOLOPAFPN_P2(nn.Module):
    """
    new Backbone:
        Add the Bottleneck Transformer architecture
    new Neck:
        Add a detection layer for small targets
    """

    def __init__(
            self,
            depth=1.0,
            width=1.0,
            in_features=("dark2", "dark3", "dark4", "dark5"),  # add p2
            in_channels=[128, 256, 512, 1024],  # add p2
            depthwise=False,
            act="silu",
    ):
        super(YOLOPAFPN_P2, self).__init__()
        self.backbone = CSPDarknet(depth, width, in_features, depthwise, act)  # 替换主干网络
        self.in_features = in_features
        self.in_channels = in_channels
        Conv = DWConv if depthwise else BaseConv

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.cbams = nn.ModuleList()

        self.lateral_conv0 = BaseConv(
            int(in_channels[3] * width), int(in_channels[2] * width), 1, 1, act=act  # 1024 -> 512
        )
        self.C3_p4 = CSPLayer(
            int(2 * in_channels[2] * width),
            int(in_channels[2] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )  # 512 + 512 -> 512

        self.reduce_conv1 = BaseConv(
            int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act  # 512 -> 256
        )
        self.C3_p3 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )  # 256 + 256 -> 256

        # add p2
        self.reduce_conv2 = BaseConv(
            int(in_channels[1] * width), int(in_channels[1] * width), 1, 1, act=act  # 256 -> 256
        )
        self.C3_p2 = CSPLayer(
            int((in_channels[0] + in_channels[1]) * width),
            int(in_channels[1] * width),  # 256
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )  # 128 + 256 -> 256  p2_out

        self.bu_conv3 = Conv(
            int(in_channels[1] * width), int(in_channels[1] * width), 3, 2, act=act  # 64/32  256
        )
        self.C3_n2 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )  # 512 -> 256 p3_out

        self.bu_conv2 = Conv(
            int(in_channels[1] * width), int(in_channels[1] * width), 3, 2, act=act  # 32/16 256
        )
        self.C3_n3 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[2] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )  # 256 + 256 -> 512  p4_out

        # removed p5

        for ch in [256, 256, 512]:
            self.cbams.append(CBAM(int(ch * width)))

    def forward(self, input):
        out_features = self.backbone(input)
        features = [out_features[f] for f in self.in_features]
        [x3, x2, x1, x0] = features  # backbone out 128/4 256/8 512/16 1024/32

        fpn_out0 = self.lateral_conv0(x0)  # 1024->512/32
        f_out0 = self.upsample(fpn_out0)  # 512/16
        f_out0 = torch.cat([f_out0, x1], 1)  # 512->1024/16
        f_out0 = self.C3_p4(f_out0)  # 1024->512/16

        fpn_out1 = self.reduce_conv1(f_out0)  # 512->256/16
        f_out1 = self.upsample(fpn_out1)  # 256/8
        f_out1 = torch.cat([f_out1, x2], 1)  # 256->512/8
        f_out1 = self.C3_p3(f_out1)  # 512->256/8

        # g3 256->256/4  addition
        fpn_out2 = self.reduce_conv2(f_out1)  # 256->128/8
        f_out2 = self.upsample(fpn_out2)  # 128/4
        # x3 = self.gam_p2(x3)  # gam
        f_out2 = torch.cat([f_out2, x3], 1)  # 128->256/4
        pan_out3 = self.C3_p2(f_out2)  # 256->256/4  p2_out
        pan_out3 = self.cbams[0](pan_out3)

        # g4 256->256/8 addition
        p_out2 = self.bu_conv3(pan_out3)  # 128/8
        p_out2 = torch.cat([p_out2, fpn_out2], 1)  # 128->256/8
        pan_out2 = self.C3_n2(p_out2)  # 256/8  p3_out
        pan_out2 = self.cbams[1](pan_out2)

        # g5 256->512/16
        p_out1 = self.bu_conv2(pan_out2)  # 256->256/16
        p_out1 = torch.cat([p_out1, fpn_out1], 1)  # 256->512/16
        pan_out1 = self.C3_n3(p_out1)  # 512->512/16 p4_out
        pan_out1 = self.cbams[2](pan_out1)

        # removed p5
        # g6 512->1024/32
        # p_out0 = self.bu_conv1(pan_out1)  # 512->512/32
        # p_out0 = torch.cat([p_out0, fpn_out0], 1)  # 512->1024/32
        # pan_out0 = self.C3_n4(p_out0)  # 1024->1024/32

        outputs = (pan_out3, pan_out2, pan_out1)  # (p2,p3,p4)

        return outputs


class YOLOPAFPN_Ghost(YOLOPAFPN_P2):
    """
        new backbone with GhostNet
    """

    def __init__(
            self,
            depth=1.0,
            width=1.0,
            in_features=("dark2", "dark3", "dark4", "dark5"),  # add p2
            in_channels=[128, 256, 512, 1024],  # add p2
            depthwise=False,
            act="silu",
    ):
        super(YOLOPAFPN_Ghost, self).__init__(depth, width, in_features, in_channels, depthwise, act)
        self.backbone = CSPDarknet_Ghost(depth, width, in_features, depthwise, act)  # 替换主干网络

        Conv = DWConv if depthwise else GhostConv

        # neck 也要修改
        self.lateral_conv0 = GhostConv(
            int(in_channels[3] * width), int(in_channels[2] * width), 1, 1, act=act  # 1024 -> 512
        )
        self.C3_p4 = C3Ghost(
            int(2 * in_channels[2] * width),
            int(in_channels[2] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )  # 512 + 512 -> 512

        self.reduce_conv1 = GhostConv(
            int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act  # 512 -> 256
        )
        self.C3_p3 = C3Ghost(
            int(2 * in_channels[1] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )  # 256 + 256 -> 256

        # add p2
        self.reduce_conv2 = GhostConv(
            int(in_channels[1] * width), int(in_channels[1] * width), 1, 1, act=act  # 256 -> 256
        )
        self.C3_p2 = C3Ghost(
            int((in_channels[0] + in_channels[1]) * width),
            int(in_channels[1] * width),  # 256
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )  # 128 + 256 -> 256  p2_out

        self.bu_conv3 = Conv(
            int(in_channels[1] * width), int(in_channels[1] * width), 3, 2, act=act  # 64/32  256
        )
        self.C3_n2 = C3Ghost(
            int(2 * in_channels[1] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )  # 512 -> 256 p3_out

        self.bu_conv2 = Conv(
            int(in_channels[1] * width), int(in_channels[1] * width), 3, 2, act=act  # 32/16 256
        )
        self.C3_n3 = C3Ghost(
            int(2 * in_channels[1] * width),
            int(in_channels[2] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )  # 256 + 256 -> 512  p4_out

        # removed p5

        # cbam for p2_out p3_out p4_ou4
        for ch in [256, 256, 512]:
            self.cbams.append(CBAM(int(ch * width)))


class YOLOPAFPN_ShuffleNet(YOLOPAFPN_P2):
    """
            new backbone with shuffle net
        """

    def __init__(
            self,
            depth=1.0,
            width=1.0,
            in_features=("dark2", "dark3", "dark4", "dark5"),  # add p2
            in_channels=[128, 256, 512, 1024],  # add p2
            depthwise=False,
            act="silu",
    ):
        super(YOLOPAFPN_ShuffleNet, self).__init__(depth, width, in_features, in_channels, depthwise, act)
        self.backbone = ShuffleNet(width, in_features, act)  # 替换主干网络


class YOLOPAFPN_rP5(nn.Module):
    """
        cspDarkNet without of path 5.
        """

    def __init__(
            self,
            depth=1.0,
            width=1.0,
            in_features=("dark3", "dark4", "dark5"),
            in_channels=[256, 512, 1024],
            depthwise=False,
            act="silu",
    ):
        super(YOLOPAFPN_rP5, self).__init__()
        self.backbone = CSPDarknet(depth, width, depthwise=depthwise, act=act)
        self.in_features = in_features
        self.in_channels = in_channels
        Conv = DWConv if depthwise else BaseConv

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.lateral_conv0 = BaseConv(
            int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act
        )
        self.C3_p4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )  # cat

        self.reduce_conv1 = BaseConv(
            int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act
        )
        self.C3_p3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[0] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv2 = Conv(
            int(in_channels[0] * width), int(in_channels[0] * width), 3, 2, act=act
        )
        self.C3_n3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # no dark5

    def forward(self, input):
        """
        Args:
            inputs: input images.

        Returns:
            Tuple[Tensor]: FPN feature.
        """

        #  backbone
        out_features = self.backbone(input)
        features = [out_features[f] for f in self.in_features]
        [x2, x1, x0] = features  # dark3, dark4

        fpn_out0 = self.lateral_conv0(x0)  # 1024->512/32
        f_out0 = self.upsample(fpn_out0)  # 512/16
        f_out0 = torch.cat([f_out0, x1], 1)  # 512->1024/16
        f_out0 = self.C3_p4(f_out0)  # 1024->512/16

        fpn_out1 = self.reduce_conv1(f_out0)  # 512->256/16
        f_out1 = self.upsample(fpn_out1)  # 256/8
        f_out1 = torch.cat([f_out1, x2], 1)  # 256->512/8
        pan_out2 = self.C3_p3(f_out1)  # 512->256/8

        p_out1 = self.bu_conv2(pan_out2)  # 256->256/16
        p_out1 = torch.cat([p_out1, fpn_out1], 1)  # 256->512/16
        pan_out1 = self.C3_n3(p_out1)  # 512->512/16

        # no dark5
        # p_out0 = self.bu_conv1(pan_out1)  # 512->512/32
        # p_out0 = torch.cat([p_out0, fpn_out0], 1)  # 512->1024/32
        # pan_out0 = self.C3_n4(p_out0)  # 1024->1024/32

        outputs = (pan_out2, pan_out1)
        return outputs