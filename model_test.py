import torch
import torch.nn as nn

from yolox.models.network_blocks import Bottleneck
from yolox.models.yolo_head import YOLOXHead
from yolox.models.yolo_pafpn import YOLOPAFPN_Rep
from yolox.models.yolox import YOLOX

from thop import clever_format, profile

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
backbone = YOLOPAFPN_Rep(0.33, 0.5)
head = YOLOXHead(1, width=0.5, strides=[8, 16, 32], in_channels=[256, 512, 1024])
model = YOLOX(backbone, head)
model = model.eval()
model = model.to(device)

# ignore_bn_list = []
# bn_list = []
# for k, m in model.named_modules():
#     if isinstance(m, Bottleneck):
#         if m.use_add:
#             ignore_bn_list.append(k.rsplit(".", 2)[0] + ".conv1.bn")
#             ignore_bn_list.append(k + '.conv1.bn')
#             ignore_bn_list.append(k + '.conv2.bn')
#     if isinstance(m, nn.BatchNorm2d) and (k not in ignore_bn_list):
#         bn_list.append(k)
#
# print(ignore_bn_list)
# print("====================")
# print(bn_list)
# print(model)
input_shape = [768, 416]
# summary(model, (9, 3, input_shape[0], input_shape[1]))

dummy_input = torch.randn(1, 3, input_shape[0], input_shape[1]).to(device)
flops, params = profile(model, (dummy_input,), verbose=False)
flops = flops * 2
flops, params = clever_format([flops, params], "%.3f")
print('Total GFLOPS: %s' % (flops))
print('Total params: %s' % (params))
y = model(dummy_input)
print(y.shape)
print(model)

'''
    s模型初始参数量和浮点数：
    Total GFLOPS: 26.635G
    Total params: 8.938M
    
    # 改变模型输入大小[768,416]：
    Total GFLOPS: 20.775G
    Total params: 8.938M
    
    # backbone->CSPDarknet  neck->P2(p2 cbam rP5)
    Total GFLOPS: 49.605G
    Total params: 7.553M
    
    # backbone->CSPDarknet_Ghost  neck->P2(p2 cbam)
    Total GFLOPS: 45.157G
    Total params: 5.580M
    
    # backbone->CSPDarknet_Ghost  neck->P2(p2 cbam, ghost) 
    Total GFLOPS: 41.082G
    Total params: 4.932M
    
    # backbone->shuffle net  neck->P2(p2 cbam)
    Total GFLOPS: 44.546G
    Total params: 4.337M
    
    # backbone->cspdarknet   neck->no dark5
    Total GFLOPS: 19.351G
    Total params: 6.507M
'''
