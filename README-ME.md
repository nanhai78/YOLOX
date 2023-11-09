# Updates

- [x] add P2
- [x] add GAM CBAM
- [x] add Bottleneck Transformer
- [x] add GhostNet Backbone 
- [ ] add TTA 
- [ ] add WBF
- [ ] Light Transformer

# 代码修改处
1. yolox_base.py

```
训练参数的修改;
get_model处的模型修改;
```

2. yolox_voc_s.py

```
训练参数的修改；
get_dataset处修改训练集路径；
get_eval_dataset处修改验证集路径
```

3. voc_classes.py

```
修改类别数和名称;
```

4. voc_eval.py

```python
parse_rec处修改xml文件的加载路径
由于xml文件中多了一些无关的标记，所以要在xml解析时加上：
    if obj.find("name").text not in VOC_CLASSES:
    	continue
```

5. val修改

```python
1.增加iou0.1到0.5的评估
2.不使用use_07_metric
3.增加f1 recall precision的输出
```



# val的流程

1. 获得评估器，位于yolox_voc_s.py下面，这个评估器是VOCEvaluator对象，这个类位于voc_evaluator.py下面
2. 然后调用它的成员函数evaluate；里面再次调用它的成员函数evaluate_prediction
3. 然后调用voc.py文件下VOCDetection类成员函数evaluate_detections
4. 里面又掉了自己的成员函数_do_python_eval
5. 里面还调了voc_eval.py文件下的voc_eval函数

# 指令

- 训练

```shell
python -m yolox.tools.train -f exps/example/yolox_voc/yolox_voc_s.py -d 1 -b 32 --fp16 -o
python -m yolox.tools.train -f exps/example/custom/yolox_tiny.py -d 1 -b 40 --fp16
```

- 验证

```shell
python -m yolox.tools.eval -f exps/example/yolox_voc/yolox_voc_s.py -d 1 -b 32 --conf 0.001 --fp16 --fuse --nms 0.65 -c 
```

- demo

```shell
python tools/demo.py image -f exps/example/yolox_voc/yolox_voc_s.py -c weight/light_models/best_ckpt_ghost.pth --path assets/demo --conf 0.25 --nms 0.65 --device cpu --speed --fp16
```

- tensorboard

```shell
在终端服务器 ->   tensorboard --logdir="事件地址" --port=6006
在本地终端映射 -> ssh -L 16006:127.0.0.1:6006 gli@192.168.0.108 -p 22
打开本地浏览器 -> 127.0.0.1:16006
```

- 查看文件数量



# 消融实验

```
原模型yolox-s
iou=0.1   f1=94.86 	rec=94.34 	prec=95.38
```

```
backbone：CSPDarknet  
Neck: 增加了P2;去掉了P5;
在输出位置添加了cbam模块 
iou=0.1   f1=94.85 	rec=93.78 	prec=95.96
```

```
backbone：CSPDarknet_Ghost  
Neck: P2;去掉了P5;在输出位置添加了cbam模块, neck没有替换成ghost
iou=0.1   f1=94.82 	rec=94.72 	prec=94.93
```

```
backbone：CSPDarknet_Ghost  
Neck: P2;去掉了P5;在输出位置添加了cbam模块 其中的C3模块也变成了C3_Ghost, BaseConv变成了Ghost_Conv
```

```
backbone：CSPDarknet Neck：去掉了P5
```

大库上：

| models    | ap(0.5:0.95) | ap0.5 | ap(0.1:0.5) | f1(0.5) | r(0.5) | p(0.5) | flops   | para   | speed(cpu)(ms) |
| --------- | ------------ | ----- | ----------- | ------- | ------ | ------ | ------- | ------ | -------------- |
| x-s       | 63.09        | 95.77 | 97.37       | 92.89   | 92.29  | 93.49  | 20.775G | 8.938M | 63             |
| **x-p2-cbam**  | 65.16        | 96.37 | 97.62       | 93.26   | 92.20  | 94.33  | 49.605G | 7.553M | 113            |
| x-p2 |  |  |  |  |  |  | 49.592G | 7.453M |  |
| b-ghost   | 64.98        | 96.42 | 97.73       | 93.10   | 92.81  | 93.39  | 45.157G | 5.580M | 149            |
| x-ghost   | 64.28        | 96.51 | 97.84       | 93.09   | 92.55  | 93.64  | 41.082G | 4.932M | 109            |
| **x-rP5** | 63.70        | 96.30 | 97.64       | 93.04   | 92.48  | 93.61  | 19.351G | 6.507M |                |
| x-rP5_nf  | 65.09        | 96.66 | 97.83       | 93.16   | 92.26  | 94.08  | 19.351G | 6.507M |                |
| x-s-rP5   | 61.35        | 95.83 | 97.43       | 92.56   | 91.24  | 93.92  | 5.449G  | 1.413M |                |
| x-p2-light |  |  |  |  |  |  | 43.505G | 6.607M | |
| x-rep-p2 |  |  |  |  |  |  | 43.851G | 6.783M | |
| **x-rep-rp5** | 64.19        | 96.43 | 97.72       | 93.31   | 92.60  | 94.04  | 18.42G  | 6.12M  |                |

| Model     | Flops  | Params | AP0.5 | AP    | Speed(cpu/gpu) |
| --------- | ------ | ------ | ----- | ----- | -------------- |
| x-small   | 20.78G | 8.94M  | 96.17 | 63.97 | /7.88+.65      |
| x-rP5     | 19.35G | 6.51M  | 96.30 | 63.70 | /6.59+.65      |
| x-P2      | 49.59G | 7.45M  | 96.71 | 64.99 | /8.98+.63      |
| x-P2-Cbam | 49.61G | 7.55M  | 96.37 | 65.16 | /9.46+.64      |
| x-rP5-Rep | 18.06G | 5.95M  | 96.57 | 64.19 | /6.46+.65      |

注. AP[0.5-0.95]简记为AP, input_size=[768, 400], nms=0.65, 基准网络是Yolox的small版本



# 8000组

| Model   | Flops  | Params | AP0.5 | AP    | Speed(cpu/gpu) |
| ------- | ------ | ------ | ----- | ----- | -------------- |
| x-small | 20.78G | 8.94M  | 96.17 | 63.97 | /7.88+.65      |





