import torch
import argparse
import os

"""
权重重参化
"""
parser = argparse.ArgumentParser(description='YOLOX Conversion')
parser.add_argument('--load', metavar='LOAD', help='path to the weights file')
parser.add_argument('--save', metavar='SAVE', help='path to the weights file')
parser.add_argument('-a', '--arch', metavar='ARCH', default='ResNet-18')


def get_model(
    depth=0.33,
    width=0.5,
    num_classes=1,
):
    from yolox.models import YOLOX, YOLOPAFPN_Rep, YOLOXHead
    in_channels = [256, 512, 1024]  # in channels for head
    # in_channels = [256, 256, 512]
    strides = [8, 16, 32]  # p2 p3 p4
    # strides = [4, 8, 16]
    backbone = YOLOPAFPN_Rep(depth, width)
    head = YOLOXHead(num_classes, width, strides=strides, in_channels=in_channels)
    model = YOLOX(backbone, head)
    return model


def convert():
    args = parser.parse_args()
    model = get_model()
    if os.path.isfile(args.load):
        print("=> loading checkpoint '{}'".format(args.load))
        ckpt = torch.load(args.load)
        model.load_state_dict(ckpt["model"])  # 模型加载权重
    else:
        print("=> no checkpoint found at {}".format(args.load))

    for module in model.modules():
        if hasattr(module, 'switch_to_deploy'):
            module.switch_to_deploy()
    if args.save is not None:
        ckpt = {
            "model": model.state_dict()
        }
        torch.save(ckpt, args.save)
        print("=> fuse finish '{}'".format(args.load))
    else:
        print("=> please confirm save path")


if __name__ == '__main__':
    convert()
