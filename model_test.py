import torch

from yolox.models.yolo_head import YOLOXHead, YOLOXHead_Light
from yolox.models.yolo_pafpn import YOLOPAFPN_P2, YOLO_Shuffle
from yolox.models.yolox import YOLOX

from thop import clever_format, profile

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
backbone = YOLO_Shuffle()
head = YOLOXHead_Light(1, width=0.5, strides=[8, 16], in_channels=[128, 256])
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
# print(model)

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
