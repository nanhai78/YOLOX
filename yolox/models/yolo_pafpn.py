#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import torch
import torch.nn as nn

from .darknet import CSPDarknet, ShuffleNet, CSPDarknet_Repvgg
from .network_blocks import BaseConv, CSPLayer, DWConv, CBAM, C3Ghost


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


class YOLOPAFPN_rP5(nn.Module):
    """
        移除检测层P5
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
        [x2, x1, x0] = features  # dark3, dark4, dark5

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

        outputs = (pan_out2, pan_out1)  # p3 p4
        return outputs


class YOLOPAFPN_P2(nn.Module):
    """
    增加P2检测层, 移除检测层P5
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

        self.lateral_conv0 = BaseConv(
            int(in_channels[3] * width), int(in_channels[2] * width), 1, 1, act=act
        )  # 512->256
        self.C3_p4 = CSPLayer(
            int(2 * in_channels[2] * width),
            int(in_channels[2] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )  # 512->256

        self.reduce_conv1 = BaseConv(
            int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act
        )  # 256->128
        self.C3_p3 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )  # 256->128

        # add p2
        self.reduce_conv2 = BaseConv(
            int(in_channels[1] * width), int(in_channels[1] * width), 1, 1, act=act
        )  # 128 -> 128
        self.C3_p2 = CSPLayer(
            int((in_channels[0] + in_channels[1]) * width),
            int(in_channels[1] * width),  # 256
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )  # 192->128

        self.bu_conv3 = Conv(
            int(in_channels[1] * width), int(in_channels[1] * width), 3, 2, act=act
        )  # 128->128
        self.C3_n2 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )  # 256->128

        self.bu_conv2 = Conv(
            int(in_channels[1] * width), int(in_channels[1] * width), 3, 2, act=act
        )  # 128->128
        self.C3_n3 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[2] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )  # 256->256

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
        f_out2 = torch.cat([f_out2, x3], 1)  # 128->256/4
        pan_out3 = self.C3_p2(f_out2)  # 256->256/4  p2_out

        # g4 256->256/8 addition
        p_out2 = self.bu_conv3(pan_out3)  # 128/8
        p_out2 = torch.cat([p_out2, fpn_out2], 1)  # 128->256/8
        pan_out2 = self.C3_n2(p_out2)  # 256/8  p3_out

        # g5 256->512/16
        p_out1 = self.bu_conv2(pan_out2)  # 256->256/16
        p_out1 = torch.cat([p_out1, fpn_out1], 1)  # 256->512/16
        pan_out1 = self.C3_n3(p_out1)  # 512->512/16 p4_out

        outputs = (pan_out3, pan_out2, pan_out1)  # (p2,p3,p4)

        return outputs


class YOLOPAFPN_P2_Cbam(nn.Module):
    """
    历史原因，改了类名，但是会导致权重也要改，所以在使用改回YOLOPAFPN_P2
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
        super(YOLOPAFPN_P2_Cbam, self).__init__()
        self.backbone = CSPDarknet(depth, width, in_features, depthwise, act)  # 替换主干网络
        self.in_features = in_features
        self.in_channels = in_channels
        Conv = DWConv if depthwise else BaseConv

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

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

        self.cbams = nn.ModuleList()
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

        outputs = (pan_out3, pan_out2, pan_out1)  # (p2,p3,p4)

        return outputs


class YOLO_Repvgg(YOLOPAFPN_rP5):
    """
    原类名为YOLO_Repvgg，如要使用权重，改回名称
    现类名YOLO_rP5_Rep
    """
    def __init__(
            self,
            depth=1.0,
            width=1.0,
            in_features=("dark3", "dark4", "dark5"),
            in_channels=[128, 256, 512],
            depthwise=False,
            act="silu",
            deploy=False
    ):
        super(YOLO_Repvgg, self).__init__(depth, width, in_features, in_channels, depthwise, act)

        self.backbone = CSPDarknet_Repvgg(depth, width, depthwise=depthwise, act=act, deploy=deploy)
        self.in_features = in_features
        self.in_channels = in_channels
        Conv = DWConv if depthwise else BaseConv

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.lateral_conv0 = BaseConv(
            in_channels[2], in_channels[0], 1, 1, act=act
        )  # 512->128
        self.C3_p4 = CSPLayer(
            in_channels[0] + in_channels[1],
            in_channels[0],
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )  # 384->128

        self.reduce_conv1 = BaseConv(
            in_channels[0], in_channels[0], 1, 1, act=act
        )  # 128->128
        self.C3_p3 = CSPLayer(
            in_channels[1],
            in_channels[0],
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )  # 256 -> 128

        # bottom-up conv
        self.bu_conv2 = Conv(
            in_channels[0], in_channels[0], 3, 2, act=act
        )  # 128->128
        self.C3_n3 = CSPLayer(
            in_channels[1],
            in_channels[0],
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )  # 256->128


class YOLO_Shuffle(nn.Module):
    def __init__(self,
                 in_features=("dark3", "dark4", "dark5"),
                 act="silu",
                 ):
        super(YOLO_Shuffle, self).__init__()
        self.backbone = ShuffleNet()
        self.in_features = in_features
        # p3(120) p4(232) p5(464)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.lateral_conv0 = BaseConv(
            464, 128, 1, 1, act=act
        )
        self.C3_p4 = CSPLayer(
            360,
            128,
            1,
            False,
            depthwise=False,
            act=act,
        )  # cat

        self.reduce_conv1 = BaseConv(
            128, 64, 1, 1, act=act
        )
        self.C3_p3 = CSPLayer(
            184,
            64,
            1,
            False,
            depthwise=False,
            act=act,
        )
        # bottom-up conv
        self.bu_conv2 = BaseConv(
            64, 64, 3, 2, act=act
        )
        self.C3_n3 = CSPLayer(
            128,
            128,
            1,
            False,
            depthwise=False,
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

        outputs = (pan_out2, pan_out1)  # p3(64) p4(128) p5(256)
        return outputs


class YOLO_Rep_P2(YOLOPAFPN_P2):
    def __init__(
            self,
            depth=1.0,
            width=1.0,
            in_features=("dark2", "dark3", "dark4", "dark5"),  # add p2
            in_channels=[64, 128, 256, 512],  # add p2
            depthwise=False,
            act="silu",
    ):
        super(YOLO_Rep_P2, self).__init__(depth, width, in_features, in_channels, depthwise, act)
        self.backbone = CSPDarknet_Repvgg(depth, width, in_features, depthwise, act)  # 替换主干网络
        self.in_features = in_features
        self.in_channels = in_channels
        Conv = DWConv if depthwise else BaseConv

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.cbams = nn.ModuleList()

        self.lateral_conv0 = BaseConv(
            512, 128, 1, 1, act=act
        )  # 512->128
        self.C3_p4 = CSPLayer(
            384,
            128,
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )  # 384 -> 128

        self.reduce_conv1 = BaseConv(
            128, 64, 1, 1, act=act  # 128->64
        )
        self.C3_p3 = CSPLayer(
            192,
            64,
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )  # 192 -> 64

        # add p2
        self.reduce_conv2 = BaseConv(
            64, 64, 1, 1, act=act  # 256 -> 256
        )
        self.C3_p2 = C3Ghost(
            128,
            64,  # 256
            round(3 * depth),
            False,
            depthwise=True,
            act=act,
        )  # 128->64

        self.bu_conv3 = DWConv(
            64, 64, 3, 2, act=act  # 64->64
        )
        self.C3_n2 = CSPLayer(
            128,
            128,
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )  # 512 -> 256 p3_out

        self.bu_conv2 = Conv(
            128, 128, 3, 2, act=act  # 32/16 256
        )
        self.C3_n3 = CSPLayer(
            192,
            128,
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )  # 192 -> 128
