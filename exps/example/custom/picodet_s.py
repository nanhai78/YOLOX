#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os

import torch.nn as nn

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.5
        self.input_size = (768, 416)
        self.mosaic_scale = (0.5, 1.5)
        # self.random_size = (10, 20)
        self.test_size = (768, 416)
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.enable_mixup = False
        self.flip_prob = 0.5


        # Define yourself dataset path
        self.data_dir = "datasets/coco"
        self.train_ann = "instances_train2017.json"
        self.val_ann = "instances_val2017.json"


        self.num_classes = 1

    def get_model(self, deploy=False):
        from yolox.models import YOLOX, YOLO_SlimNeck, YOLOXHead

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        if getattr(self, "model", None) is None:
            in_channels = [256, 512, 1024]  # in channels for head
            backbone = YOLO_SlimNeck(self.depth, self.width)
            head = YOLOXHead(self.num_classes, self.width, in_channels=in_channels, act=self.act)
            self.model = YOLOX(backbone, head)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        self.model.train()
        return self.model
