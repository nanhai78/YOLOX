#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import contextlib
from copy import deepcopy
from typing import Sequence

import torch
import torch.nn as nn

__all__ = [
    "fuse_conv_and_bn",
    "fuse_model",
    "get_model_info",
    "replace_module",
    "freeze_module",
    "adjust_status",
]


def get_model_info(model: nn.Module, tsize: Sequence[int]) -> str:
    from thop import profile

    stride = 64
    img = torch.zeros((1, 3, stride, stride), device=next(model.parameters()).device)
    flops, params = profile(deepcopy(model), inputs=(img,), verbose=False)
    params /= 1e6
    flops /= 1e9
    flops *= tsize[0] * tsize[1] / stride / stride * 2  # Gflops
    info = "Params: {:.2f}M, Gflops: {:.2f}".format(params, flops)
    return info


def fuse_conv_and_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    """
    Fuse convolution and batchnorm layers.
    check more info on https://tehnokv.com/posts/fusing-batchnorm-and-conv/

    Args:
        conv (nn.Conv2d): convolution to fuse.
        bn (nn.BatchNorm2d): batchnorm to fuse.

    Returns:
        nn.Conv2d: fused convolution behaves the same as the input conv and bn.
    """
    fusedconv = (
        nn.Conv2d(
            conv.in_channels,
            conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            groups=conv.groups,
            bias=True,
        )
        .requires_grad_(False)
        .to(conv.weight.device)
    )

    # prepare filters
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

    # prepare spatial bias
    b_conv = (
        torch.zeros(conv.weight.size(0), device=conv.weight.device)
        if conv.bias is None
        else conv.bias
    )
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(
        torch.sqrt(bn.running_var + bn.eps)
    )
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fusedconv


def fuse_model(model: nn.Module) -> nn.Module:
    """fuse conv and bn in model

    Args:
        model (nn.Module): model to fuse

    Returns:
        nn.Module: fused model
    """
    from yolox.models.network_blocks import BaseConv, RepVGGBlock, Shuffle_Block
    from yolox.models.slim_neck import Conv
    print("Fusing layers...")
    for m in model.modules():
        if type(m) is BaseConv and hasattr(m, "bn"):
            print("Fuse BaseConv")
            m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
            delattr(m, "bn")  # remove batchnorm
            m.forward = m.fuseforward  # update forward
        elif type(m) is Conv and hasattr(m, "bn"):
            print("Fuse Conv")
            m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
            delattr(m, "bn")  # remove batchnorm
            m.forward = m.forward_fuse
        elif type(m) is RepVGGBlock:
            print("Fuse RepVGGBlock")
            if hasattr(m, 'rbr_1x1'):
                kernel, bias = m.get_equivalent_kernel_bias()  # 获得融合后的权重和偏置
                rbr_reparam = nn.Conv2d(in_channels=m.rbr_dense.conv.in_channels,  # 重参后的Conv模块
                                        out_channels=m.rbr_dense.conv.out_channels,
                                        kernel_size=m.rbr_dense.conv.kernel_size,
                                        stride=m.rbr_dense.conv.stride,
                                        padding=m.rbr_dense.conv.padding, dilation=m.rbr_dense.conv.dilation,
                                        groups=m.rbr_dense.conv.groups, bias=True)
                rbr_reparam.weight.data = kernel  # 给重参的Conv重新赋参
                rbr_reparam.bias.data = bias
                # for para in model.parameters():
                #     para.detach_()
                m.rbr_dense = rbr_reparam  #
                m.__delattr__('rbr_1x1')  # 去掉 1 * 1卷积模块
                if hasattr(m, 'rbr_identity'):  # 去掉 identity模块
                    m.__delattr__('rbr_identity')
                if hasattr(m, 'id_tensor'):
                    m.__delattr__('id_tensor')  # 去掉id_tensor
                m.deploy = True
                delattr(m, 'se') # 删除se模块
                m.forward = m.fusevggforward  # update forward
        elif type(m) is Shuffle_Block:
            print("Fuse Shuffle_Block")
            if hasattr(m, 'branch1'):  # 第一个分支的融合  3*3Conv + BN + 1*1卷积 + BN + Relu =>  3 * 3卷积 + 1*1卷积 + Relu
                re_branch1 = nn.Sequential(
                    nn.Conv2d(m.branch1[0].in_channels, m.branch1[0].out_channels,
                              kernel_size=m.branch1[0].kernel_size, stride=m.branch1[0].stride,
                              padding=m.branch1[0].padding, groups=m.branch1[0].groups),
                    nn.Conv2d(m.branch1[2].in_channels, m.branch1[2].out_channels,
                              kernel_size=m.branch1[2].kernel_size, stride=m.branch1[2].stride,
                              padding=m.branch1[2].padding, bias=False),
                    nn.ReLU(inplace=True),
                )
                re_branch1[0] = fuse_conv_and_bn(m.branch1[0], m.branch1[1])
                re_branch1[1] = fuse_conv_and_bn(m.branch1[2], m.branch1[3])
                # pdb.set_trace()
                # print(m.branch1[0])
                m.branch1 = re_branch1
            if hasattr(m, 'branch2'):
                re_branch2 = nn.Sequential(
                    nn.Conv2d(m.branch2[0].in_channels, m.branch2[0].out_channels,
                              kernel_size=m.branch2[0].kernel_size, stride=m.branch2[0].stride,
                              padding=m.branch2[0].padding, groups=m.branch2[0].groups),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(m.branch2[3].in_channels, m.branch2[3].out_channels,
                              kernel_size=m.branch2[3].kernel_size, stride=m.branch2[3].stride,
                              padding=m.branch2[3].padding, bias=False),
                    nn.Conv2d(m.branch2[5].in_channels, m.branch2[5].out_channels,
                              kernel_size=m.branch2[5].kernel_size, stride=m.branch2[5].stride,
                              padding=m.branch2[5].padding, groups=m.branch2[5].groups),
                    nn.ReLU(inplace=True),
                )
                re_branch2[0] = fuse_conv_and_bn(m.branch2[0], m.branch2[1])
                re_branch2[2] = fuse_conv_and_bn(m.branch2[3], m.branch2[4])
                re_branch2[3] = fuse_conv_and_bn(m.branch2[5], m.branch2[6])
                # pdb.set_trace()
                m.branch2 = re_branch2

    return model


def replace_module(module, replaced_module_type, new_module_type, replace_func=None) -> nn.Module:
    """
    Replace given type in module to a new type. mostly used in deploy.

    Args:
        module (nn.Module): model to apply replace operation.
        replaced_module_type (Type): module type to be replaced.
        new_module_type (Type)
        replace_func (function): python function to describe replace logic. Defalut value None.

    Returns:
        model (nn.Module): module that already been replaced.
    """

    def default_replace_func(replaced_module_type, new_module_type):
        return new_module_type()

    if replace_func is None:
        replace_func = default_replace_func

    model = module
    if isinstance(module, replaced_module_type):
        model = replace_func(replaced_module_type, new_module_type)
    else:  # recurrsively replace
        for name, child in module.named_children():
            new_child = replace_module(child, replaced_module_type, new_module_type)
            if new_child is not child:  # child is already replaced
                model.add_module(name, new_child)

    return model


def freeze_module(module: nn.Module, name=None) -> nn.Module:
    """freeze module inplace

    Args:
        module (nn.Module): module to freeze.
        name (str, optional): name to freeze. If not given, freeze the whole module.
            Note that fuzzy match is not supported. Defaults to None.

    Examples:
        freeze the backbone of model
        >>> freeze_moudle(model.backbone)

        or freeze the backbone of model by name
        >>> freeze_moudle(model, name="backbone")
    """
    for param_name, parameter in module.named_parameters():
        if name is None or name in param_name:
            parameter.requires_grad = False

    # ensure module like BN and dropout are freezed
    for module_name, sub_module in module.named_modules():
        # actually there are no needs to call eval for every single sub_module
        if name is None or name in module_name:
            sub_module.eval()

    return module


@contextlib.contextmanager
def adjust_status(module: nn.Module, training: bool = False) -> nn.Module:
    """Adjust module to training/eval mode temporarily.

    Args:
        module (nn.Module): module to adjust status.
        training (bool): training mode to set. True for train mode, False fro eval mode.

    Examples:
        >>> with adjust_status(model, training=False):
        ...     model(data)
    """
    status = {}

    def backup_status(module):
        for m in module.modules():
            # save prev status to dict
            status[m] = m.training
            m.training = training

    def recover_status(module):
        for m in module.modules():
            # recover prev status from dict
            m.training = status.pop(m)

    backup_status(module)
    yield module
    recover_status(module)
