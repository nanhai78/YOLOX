import time
import torch

from yolox.models.yolo_head import  YOLOXHead
from yolox.models.yolo_pafpn import YOLOPAFPN_P2
from yolox.models.yolox import YOLOX
from torchstat import stat

from thop import clever_format, profile
from torchsummary import summary

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
backbone = YOLOPAFPN_P2(depth=0.33, width=0.5)
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
    # 改变模型输入大小[800,480]
    Total GFLOPS: 24.970G
    Total params: 8.938M
    # 改变模型输入大小[768,448]
    Total GFLOPS: 22.374G
    Total params: 8.938M
    
    # 增加Spp模块
    # Total params: 14.511M
    # Total GFLOPS: 29.888G
    
    # CSPMvitLayer
    # Total params: 13.473M
    # Total GFLOPS: 27.398G
    # 修改1
    Total params: 13.525M
    Total GFLOPS: 27.525G
    
    将backbone最后一层的CSPlayer 替换成CSPTRlayer:
    Total GFLOPS: 26.739G
    Total params: 9.068M
    
    增加了p2head：
    Total GFLOPS: 60.288G
    Total params: 9.680M
    
    在base模型上,将backbone和pan的输出head的CSP模块都换成CSPTRlayer：
    Total GFLOPS: 27.048G
    Total params: 9.239M
    
    增加MobileVit_block: 在pafpn最后的3个c3模块后
    Total GFLOPS: 38.692G
    Total params: 11.272M
    
    在backbone后面对每个分支增加一个mvt_cross2
    Total GFLOPS: 44.294
    Total params: 18.604
'''
