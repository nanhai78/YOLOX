# Updates

- [x] add P2
- [x] add 注意力机制
- [x] add Bottleneck Transformer
- [x] add TTA

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

