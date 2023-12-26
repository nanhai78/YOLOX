import torch
from yolox.models.yolo_head import YOLOXHead
from exps.example.custom.ES_DBB import YOLOPAFPN2
from exps.example.custom.RepConv import YOLOPAFPN1
from exps.example.custom.SlimNeck import YOLOPAFPN3
from exps.example.custom.Prune import YOLOPAFPN4
from yolox.models.yolo_pafpn import YOLOPAFPN
from exps.example.custom.GSConv import YOLOPAFPN5
from yolox.models.yolox import YOLOX
from thop import clever_format, profile
from yolox.utils.model_utils import fuse_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

width = 0.5
depth = 0.33

backbone = YOLOPAFPN3(depth, width)
head = YOLOXHead(1, width, in_channels=[256, 512, 1024])
model = YOLOX(backbone, head)
model = model.eval()
model = model.to(device)
# model = fuse_model(model)

input_shape = [768, 416]
dummy_input = torch.randn(1, 3, input_shape[0], input_shape[1]).to(device)
flops, params = profile(model, (dummy_input,), verbose=False)
flops = flops * 2
flops, params = clever_format([flops, params], "%.3f")
print('Total GFLOPS: %s' % (flops))
print('Total params: %s' % (params))
# print(model)