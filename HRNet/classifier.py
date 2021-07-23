# -*- coding: utf-8 -*-
# @Author: 红白黑
# @Date:   2021-07-23 12:39:39
# @Last Modified by:   红白黑
# @Last Modified time: 2021-07-23 12:44:25
import numpy as np

import paddle
from paddle import nn
from paddle.fluid.layers.nn import pad
from paddle.nn import initializer
from paddle.nn import functional as F

from model import HRNet

class HRNetPoolOutput(nn.Layer):
    '''校验完成：完成输出的通道变换，并经过自适应均值池化得到1x1的图像

        params:
            inchannels:  输出层的输入通道
            outchannels: 输出层的变换后的输出通道
    '''
    def __init__(self, inchannels, outchannels):
        super(HRNetPoolOutput, self).__init__()

        self.conv = nn.Conv2D(inchannels, outchannels, kernel_size=1, bias_attr=None)
        self.avgpool = nn.AdaptiveAvgPool2D(output_size=1)   # 自适应池化，得到指定大小的池化结果
        self.relu = nn.ReLU()
    

    def forward(self, inputs):
        assert isinstance(inputs, list), \
            "Please make sure input data is list({0}) in HRNetOutput.".format(type(inputs))

        outs = []
        
        # 针对不同的输入，进行相应的操作
        for i in range(len(inputs)):
            out = inputs[i]
            out = self.conv(out)
            out = self.avgpool(out)
            out = self.relu(out)
            outs.append(out)
        
        return outs

class HRNetClassifier(nn.Layer):
    '''校验完成：产生预测分类的结果，支持多分辨率预测输出

        params:
            inchannels:  输入大小
            num_classes: 分类数 > 0
    '''
    def __init__(self, inchannels, num_classes):
        super(HRNetClassifier, self).__init__()

        self.flatten = nn.Flatten()
        self.out_fc  = nn.Linear(inchannels, num_classes)

    def forward(self, inputs):
        assert isinstance(inputs, list), \
            "Please make sure input data is list({0}) in HRNetOutput.".format(type(inputs))
            
        outs = []
        # 针对不同的输入，进行相应的操作
        for i in range(len(inputs)):
            out = inputs[i]
            out = self.flatten(out)
            out = self.out_fc(out)
            outs.append(out)
        
        return outs


class HRNetClassification(nn.Layer):

    def __init__(self, num_classes, width=32, mode='fuse'):
        super(HRNetClassification, self).__init__()
        
        self.mode  = mode
        self.width = width

        # 骨干网络
        self.hrnet      = HRNet(width=width, mode=mode)
        # 特征池化输出层
        self.output     = HRNetPoolOutput(self.hrnet.hrnet_fusion.outchannels[0], 2048)
        # 分类层
        self.classifier = HRNetClassifier(2048, num_classes)

    def forward(self, inputs):
        
        x = self.hrnet(inputs)
        x = self.output(x)
        x = self.classifier(x)   # return list

        if self.mode == 'multi':
            return [x[-1]]
        else:
            return [x[0]]

if __name__ == "__main__":
    # 模型实例
    # keep, fuse, multi
    model = HRNetClassification(num_classes=2, width=32, mode='multi')
    
    # 构建模拟输入
    data = np.random.randint(0, 256, (1, 3, 256, 256)).astype(np.float32)
    data = paddle.to_tensor(data)

    # 输入预测 -- 返回list
    # 如果 list 长度为1，则只包含高分辨率的输出
    y_preds = model(data)

    # 遍历输出
    for i in range(len(y_preds)):
        print(y_preds[i].shape)
