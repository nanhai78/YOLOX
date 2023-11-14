import torch

from yolox.models.yolo_head import YOLOXHead
from yolox.models.light_neck import YOLOPAFPN_Gs, YOLOPAFPN_Pico
from yolox.models.yolox import YOLOX

from thop import clever_format, profile

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
backbone = YOLOPAFPN_Pico(0.33, 0.5)
head = YOLOXHead(1, width=0.5, strides=[8, 16, 32], in_channels=[256, 512, 1024])
model = YOLOX(backbone, head)
model = model.eval()
model = model.to(device)
input_shape = [768, 416]

dummy_input = torch.randn(1, 3, input_shape[0], input_shape[1]).to(device)
flops, params = profile(model, (dummy_input,), verbose=False)
flops = flops * 2
flops, params = clever_format([flops, params], "%.3f")
print('Total GFLOPS: %s' % (flops))
print('Total params: %s' % (params))
# y = model(dummy_input)
# print(y.shape)
# print(model)