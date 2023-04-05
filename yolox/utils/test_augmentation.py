import torch
import math
import functools as F

"""
自定义版本的测试时数据增强，函数接口
"""


def _forward_augment(self, x):
    img_size = x.shape[-2:]  # height, width
    s = [1, 0.83, 0.67]  # scales
    f = [None, 3, None]  # flips (2-ud上下flip, 3-lr左右flip)
    y = []  # outputs

    # 这里相当于对输入x进行3次不同参数的测试数据增强推理, 每次的推理结构都保存在列表y中
    for si, fi in zip(s, f):
        # scale_img缩放图片尺寸
        # 通过普通的双线性插值实现，根据ratio来控制图片的缩放比例，最后通过pad 0补齐到原图的尺寸
        xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
        yi = self._forward_once(xi)[0]  # forward：torch.Size([1, 25200, 25])
        # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save

        # _descale_pred将推理结果恢复到相对原图图片尺寸, 只对坐标xywh：yi[..., :4]进行恢复
        # 如果f=2,进行上下翻转; 如果f=3,进行左右翻转
        yi = self._descale_pred(yi, fi, si, img_size)
        y.append(yi)  # [b, 25200, 25] / [b, 18207, 25] / [b, 12348, 25]

    y = self._clip_augmented(y)  # clip augmented tails
    return torch.cat(y, 1), None  # augmented inference, train


# 通过普通的双线性插值实现，根据ratio来控制图片的缩放比例，最后通过pad 0补齐到原图的尺寸
def scale_img(img, ratio=1.0, same_shape=False, gs=32):  # img(16,3,256,416)
    # scales img(bs,3,y,x) by ratio constrained to gs-multiple
    if ratio == 1.0:
        return img
    else:
        h, w = img.shape[2:]
        s = (int(h * ratio), int(w * ratio))  # new size
        img = F.interpolate(img, size=s, mode='bilinear', align_corners=False)  # resize
        if not same_shape:  # pad/crop img
            h, w = [math.ceil(x * ratio / gs) * gs for x in (h, w)]
        return F.pad(img, [0, w - s[1], 0, h - s[0]], value=0.447)  # value = imagenet mean
