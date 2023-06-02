import torch
from yolox.models import YOLOX, YOLOPAFPN, YOLOXHead
import os


def repvgg_model_convert(model: torch.nn.Module, save_path=None, do_copy=True):
    import copy
    if do_copy:
        model = copy.deepcopy(model)
    for module in model.modules():
        if hasattr(module, 'switch_to_deploy'):
            module.switch_to_deploy()
    if save_path is not None:
        torch.save(model.state_dict(), save_path)
    return model


if __name__ == '__main__':
    # 加载训练模型
    depth = 0.33
    width = 0.5
    num_classes = 1
    strides = [8, 16]
    in_channels = [256, 256]

    backbone = YOLOPAFPN(depth, width)
    head = YOLOXHead(num_classes, width, strides=strides, in_channels=in_channels)
    train_model = YOLOX(backbone, head)
    # 加载权重
    load = ""
    save = ""

    if os.path.isfile(load):
        print("=> loading checkpoint '{}'".format(load))
        ckpt = torch.load(load)
        train_model.load_state_dict(ckpt)  # 将权重加载到训练好的模型上
        repvgg_model_convert(train_model, save)  # 模型进行转换，然后重新保存
    else:
        print("=> no checkpoint found at '{}'".format(load))
