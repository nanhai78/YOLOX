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
```

- 验证

```shell
python -m yolox.tools.eval -f exps/example/yolox_voc/yolox_voc_s.py -d 1 -b 32 --conf 0.001 --fp16 --fuse --nms 0.65
```

- demo

```shell
python tools/demo.py image -f exps/example/yolox_voc/yolox_voc_s.py -c weight/light_models/best_ckpt_ghost.pth --path assets/demo --conf 0.25 --nms 0.65 --device cpu
```

- tensorboard

```shell
在终端服务器 ->   tensorboard --logdir="事件地址" --port=6006
在本地终端映射 -> ssh -L 16006:127.0.0.1:6006 gli@192.168.0.108 -p 22
打开本地浏览器 -> 127.0.0.1:16006
```

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
backbone：CSPDarknet Neck：只去掉p5；在输出位置添加cbam模块
```

```
backbone：CSPDarknet_Ghost  
Neck: P2;去掉了P5;在输出位置添加了cbam模块 其中的C3模块也变成了C3_Ghost, BaseConv变成了Ghost_Conv


```

| models      | ap(0.5:0.95) | ap0.5 | ap(0.1:0.5) | f1(0.5) | r(0.5) | p(0.5) | flops   | para   | speed(cpu)(ms) |
| ----------- | ------------ | ----- | ----------- | ------- | ------ | ------ | ------- | ------ | -------------- |
| yolox-s     | 63.09        | 95.77 | 97.37       | 92.89   | 92.29  | 93.49  | 20.775G | 8.938M |                |
| yolox-p2    | 65.16        | 96.37 | 97.62       | 93.26   | 92.20  | 94.33  | 49.605G | 7.553M | 151            |
| yolox-ghost | 64.98        | 96.42 | 97.73       | 93.10   | 92.81  | 93.39  | 45.157G | 5.580M |                |

