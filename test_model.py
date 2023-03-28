import time
import torch

from yolox.models.yolo_head import  YOLOXHead
from yolox.models.yolo_pafpn import YOLOPAFPN_Ghost
from yolox.models.yolox import YOLOX
from torchstat import stat

from thop import clever_format, profile
from torchsummary import summary

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
backbone = YOLOPAFPN_Ghost(depth=0.33, width=0.5)
head = YOLOXHead(1, width=0.5, strides=[4, 8, 16], in_channels=[256, 256, 512])
model = YOLOX(backbone, head)
model = model.eval()
model = model.to(device)


input_shape = [768, 416]
# summary(model, (9, 3, input_shape[0], input_shape[1]))

dummy_input = torch.randn(1, 3, input_shape[0], input_shape[1]).to(device)
flops, params = profile(model, (dummy_input,), verbose=False)
flops = flops * 2
flops, params = clever_format([flops, params], "%.3f")
print('Total GFLOPS: %s' % (flops))
print('Total params: %s' % (params))
#
print(model)

'''
    s模型初始参数量和浮点数：
    Total GFLOPS: 26.635G
    Total params: 8.938M
    
    # 改变模型输入大小[768,416]：
    Total GFLOPS: 20.775G
    Total params: 8.938M
    
    # add BoT P2 GAM CBAM  removed P5
    Total GFLOPS: 53.464G
    Total params: 7.263M
    
    # Ghost Backbone
    Total GFLOPS: 49.259G
    Total params: 5.683M
'''
