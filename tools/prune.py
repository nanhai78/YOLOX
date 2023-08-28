# 剪枝
import torch
import torch.nn as nn
import numpy as np
from yolox.models.network_blocks import Bottleneck

from_layers = {  # key为当前层，value为上一层的名称

}
def get_model(ckpt_files,
              depth,
              width,
              num_classes
              ):
    from yolox.models import YOLOX, YOLOPAFPN, YOLOXHead

    in_channels = [256, 512, 1024]  # in channels for head
    strides = [8, 16, 32]
    backbone = YOLOPAFPN(depth, width)
    head = YOLOXHead(num_classes, width, strides=strides, in_channels=in_channels)
    model = YOLOX(backbone, head)
    model.cuda()
    model.eval()
    ckpt = torch.load(ckpt_files, map_location="cuda:0")
    model.load_state_dict(ckpt["model"])

    return model


def get_prune_model(
        depth,
        width,
        num_classes,
        mask_dict: dict,
):
    from yolox.models import YOLOX, YOLOPAFPN, YOLOXHead

    in_channels = [256, 512, 1024]  # in channels for head
    strides = [8, 16, 32]
    backbone = YOLOPAFPN(depth, width)
    head = YOLOXHead(num_classes, width, strides=strides, in_channels=in_channels)
    model = YOLOX(backbone, head)
    model.cuda()
    model.eval()

    # 根据mask修改卷积层
    change_conv = mask_dict.keys()  # 为了不包含head最后的Conv
    for key, m in model.named_modules():
        key = key.rsplit(".", 2)[0] + ".bn"  # 将conv名改为bn的名称
        if isinstance(m, nn.Conv2d) and key in change_conv:
            f_layer = from_layers[key]  # 上一层bn名称
            c1 = c2 = 0  # in_channel/out_channel
            if isinstance(f_layer, str):  # 上一层只有一层
                c1 = mask_dict[f_layer].sum()
            else:  # 上一层有两层
                for _layer in f_layer:
                    c1 += mask_dict[_layer].sum()
            c2 = mask_dict[key].sum()
            # 更改channel
            m.in_channels = c1
            m.out_channels = c2

    return model


def gather_bn_weights(module_list):
    size_list = [idx.weight.data.shape[0] for idx in module_list.values()]  # 各bn层的channel
    # 将需要被剪枝的所有BN层的channel加起来，大概右1万多个channel,生成一个shape为1万多的全0的mask，记录每个channel
    bn_weights = torch.zeros(sum(size_list))
    index = 0
    for i, idx in enumerate(module_list.values()):
        size = size_list[i]
        bn_weights[index:(index + size)] = idx.weight.data.abs().clone()
        index += size
    return bn_weights


def obtain_bn_mask(bn_module, thr):
    thr = thr.cuda()
    mask = bn_module.weight.data.abs().ge(thr).float()

    return mask


if __name__ == '__main__':
    model = get_model("D:\\python_all\\Workspace002\\best_ckpt.pth", 0.33, 0.5, 1)

    model_list = {}  # 被裁剪的层
    ignore_bn_list = []  # 不被剪枝的层

    # 记录不被剪枝的层，以及添加要剪枝的层到model_list中
    for key, m in model.named_modules():
        if isinstance(m, Bottleneck) and m.use_add:
            ignore_bn_list.append(key.rsplit(".", 2)[0] + ".conv1.bn")
            ignore_bn_list.append(key + '.conv1.bn')
            ignore_bn_list.append(key + '.conv2.bn')
    for key, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d) and key not in ignore_bn_list:
            model_list[key] = m
    bn_weights = gather_bn_weights(model_list)  # bn_weights的shape为所有bn层的channel之和，每个数为bn层的gamma的绝对值
    sorted_bn = torch.sort(bn_weights)[0]  # 对上面的权重进行升序排序

    highest_val = []  # 每一层中最高的权重值
    for m in model_list.values():
        highest_val.append(m.weight.data.abs().max().item())
    highest_val = min(highest_val)  # 每层中最高的gamma中挑选出一个最小的。避免剪枝掉所有层
    percent = (sorted_bn == highest_val).nonzero()[0, 0].item() / len(bn_weights)  # 剪枝的百分比
    print(f'Suggested Gamma threshold should be less than {highest_val:.4f}.')
    print(f'The corresponding prune ratio is {percent:.3f}, but you can set higher.')
    # assert opt.percent < percent, f"Prune ratio should less than {percent}, otherwise it may cause error!!!"

    thr_idx = int(len(sorted_bn) * percent)  # sorted_bn中前thr_idx个gamma要被剪枝掉
    thr = sorted_bn[thr_idx]  # 剪枝的gamma阈值
    print(f'Gamma value that less than {thr:.4f} are set to zero!')
    print("=" * 104)
    print(f"|\t{'layer name':<40}{'|':<10}{'origin channels':<20}{'|':<10}{'remaining channels':<20}|")

    mask_dict = {}  # 记录每一层的Bn的是否要被剪枝的mask  key:bn_name value: 是否被剪枝的mask
    remain_num = 0  # 记录被保存下来的gamma数量
    for key, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            bn_module = m
            mask = obtain_bn_mask(bn_module, thr)  # 根据阈值得到单层bn的mask,小于thr的mask为0,大于等于的为1
            if key in ignore_bn_list:
                mask = torch.ones(m.weight.data.size()).cuda()
            mask_dict[key] = mask
            remain_num += int(mask.sum())
            bn_module.weight.data.mul_(mask)  # 将mask乘上原先的权重，将一些gamma置为0
            bn_module.bias.data.mul_(mask)
            # 打印剪枝前的gamma数量和剪枝后的数量
            print(f"|\t{key:<40}{'|':<10}{bn_module.weight.data.size()[0]:<20}{'|':<10}{int(mask.sum()):<20}|")
    print("=" * 94)

    model_dict = model.state_dict()  # 原模型的权重字典
    pruned_model = get_prune_model(0.33, 0.5, 1, mask_dict)    # 加载剪枝模型
    # 开始对权重进行剪枝了。
    pruned_model_dict = pruned_model.state_dict()
    assert pruned_model_dict.keys() == model_dict.keys()  # 两个模型权重名称肯定是一样的。"
    # 对原始模型的权重model_dict->进行剪枝
    for ((key_origin, m_origin), (key_pruned, m_pruned)) in zip(model.named_modules(), pruned_model.named_modules()):
        assert key_origin == key_pruned
        bn_name = key_origin.rsplit(".", 2)[0] + ".bn"  # conv对应的当前层bn层名字
        pre_bn_name = from_layers[bn_name]  # 上一层的bn层名字
        # 卷积层剪枝
        if isinstance(m_origin, nn.Conv2d) and bn_name in mask_dict.keys():  # 卷积层剪枝，不包括head最后的Conv
            if isinstance(pre_bn_name, str):  # 上一层只有一层
                # 【out_idx】：当前层保留的channel idx  【in_idx】上一层保留的channel idx
                # 根据当前层保留的channel 可以知道滤除哪些滤波器; 根据in_idx可以知道单个滤波器可以剪掉哪些channel
                out_idx = np.squeeze(np.argwhere(np.asarray(mask_dict[bn_name].cpu().numpy())))
                in_idx = np.squeeze(np.argwhere(np.asarray(mask_dict[pre_bn_name].cpu().numpy())))
                w = m_origin.weight.data[:, in_idx, :, :].clone()  # 取出单个滤波器中保留的channel
                w = w[out_idx, :, :, :].clone()  # 取出保留的滤波器
                assert len(w.shape) == 4
                m_pruned.weight.data = w.clone()  # 将剪枝出来的权重放到剪枝模型上
            if isinstance(pre_bn_name, list):  # 上一层有2个层 比如 C3结构中的conv3

                out_idx = np.squeeze(np.argwhere(np.asarray(mask_dict[bn_name].cpu().numpy())))

            else: # 没有上一层 如FOCUS中的conv
                out_idx = np.squeeze(np.argwhere(np.asarray(mask_dict[bn_name].cpu().numpy())))
                w = m_pruned.weight.data[out_idx, :, :, :].clone()
                assert len(w.shape) == 4
                m_pruned.weight.data = w.clone()

        # if isinstance(m_origin, nn.Conv2d) and bn_name not in mask_dict.keys():  # 卷积层剪枝，head最后的Conv

        # if isinstance(m_origin, nn.BatchNorm2d):  # BN层剪枝


    # 保存剪枝后的模型。
    print()
