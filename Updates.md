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

