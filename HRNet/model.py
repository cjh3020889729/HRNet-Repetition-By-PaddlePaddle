# -*- coding: utf-8 -*-
# @Author: 红白黑
# @Date:   2021-07-19 11:49:59
# @Last Modified by:   红白黑
# @Last Modified time: 2021-07-23 12:46:08
import numpy as np

import paddle
from paddle import nn
from paddle.fluid.layers.nn import pad
from paddle.nn import initializer
from paddle.nn import functional as F

# BatchNorm的动量
BN_MOMENTUM = 0.2

class HRNetConv3x3(nn.Layer):
    '''校验完成
        3x3卷积，进行一般的特征提取操作
    '''
    def __init__(self, in_channels, out_channels, stride=1, padding=0):
        super(HRNetConv3x3, self).__init__()

        # 关掉bias
        self.conv = nn.Conv2D(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=3, stride=stride, padding=padding, bias_attr=None)
        self.bn   = nn.BatchNorm2D(out_channels, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        
        x = self.conv(inputs)
        x = self.bn(x)
        x = self.relu(x)

        return x


class HRNetStem(nn.Layer):
    '''校验完成 -- scale: 1/4.
        Stem特征提取输入图像的浅层信息

        eg:
           model = HRNetStem(3, 64)
           y = model(tensor)  # (1, 3, 256, 256)
           output:
                [1, 64, 64, 64]
    '''
    def __init__(self, in_channels, out_channels):
        super(HRNetStem, self).__init__()

        self.conv1 = HRNetConv3x3(in_channels=in_channels, out_channels=out_channels, stride=2, padding=1)
        self.conv2 = HRNetConv3x3(in_channels=out_channels, out_channels=out_channels, stride=2, padding=1)


    def forward(self, inputs):

        x = self.conv1(inputs)
        x = self.conv2(x)
        
        return x


class PlaceHolder(nn.Layer):
    '''校验完成
        占位符，没有什么特别的操作

        eg:
           model = PlaceHolder()
           y = model(tensor)
           output:
                y == tensor
    '''
    def __init__(self):
        super(PlaceHolder, self).__init__()

    def forward(self, inputs):
        return inputs



class HRNetTrans(nn.Layer):
    '''校验完成：支持前stage到后stage的自动传播转换
        转换层：将上一个stage的多分支输出转换为下一层的多分支输入

        params:
            old_branch_channels: 上一stage的多分支对应的通道数
            new_branch_channels: 下一stage的多分支对应的通道数 -- 还可知分支数
        
        eg:
           model = HRNetTrans([32], [32, 64])
           y1, y2 = model([tensor])
           output:
                [1, 32, 66, 66]
                [1, 64, 33, 33]
    '''
    def __init__(self, old_branch_channels, new_branch_channels):
        super(HRNetTrans, self).__init__()
        # 检查转换是否合理--通过检查分支数是否连续变化来确定
        self.check_branch_num(old_branch_channels, new_branch_channels)

        # 上一个stage的分支数
        self.old_branch_num = len(old_branch_channels)
        # 当前/下一个stage的分支数
        self.new_branch_num = len(new_branch_channels)

        # stage的各分支通道数信息
        self.old_branch_channels = old_branch_channels
        self.new_branch_channels = new_branch_channels

        # trans_layers 是一个可以索引的Layer集合
        # 包含两层索引
        # 外层索引对应原来的分支indexs
        # 内层对应原始分支一一映射到新stage的分支的indexs
        self.trans_layers = self.create_new_branch_trans_layers()

    def forward(self, inputs):
        assert isinstance(inputs, list),\
            "Please input list({0}) data in the trans layer.".format(type(inputs))
        outs = []

        # 遍历上一个stage每个分支的信息，逐次生成下一个stage上相应的每个分支的贡献
        for i in range(0, self.old_branch_num):  # 得到上一stage的输出
            x = inputs[i]  # 上1stage分支输出
            # 派生下一stage的各个分支值
            out = []
            for j in range(0, self.new_branch_num):
                y = self.trans_layers[i][j](x)  # 派生转换
                # print(i, '-', j, ' : ', self.new_branch_num, '--shape: ', y.shape)
                out.append(y)  # 添加得到的派生结果
            
            # 融合上一个stage每个分支所贡献在下一个stage上每个分支的贡献
            # 第一次得到输出结果时，直接copy
            if len(outs) == 0:
                outs = out
            else: # 否则逐一相加聚合 -- 聚合方式为add
                for i in range(self.new_branch_num):
                    outs[i] += out[i]  # 对应贡献相加
                    
        # 返回输出是list，元素为tensor
        return outs

            
    def check_branch_num(self, old_branch_channels, new_branch_channels):
        '''
            检查stage间的分支数是否合理
        '''     
        assert len(new_branch_channels) - len(old_branch_channels) == 1, \
                "Please make sure the number of closed stage's branch is only less than 1."


    def create_new_branch_trans_layers(self):
        '''
            创建由上一stage分支产生下一stage分支的网络层
        '''
        totrans_layers = []  # 所有分支的转换层

        # 遍历上一个stage的每一个分支
        for i in range(self.old_branch_num):
            branch_trans = []
            # 生成新分支的元素
            for j in range(self.new_branch_num):
                if i == j:   # 不需要任何操作就可以匹配
                    layer = PlaceHolder()
                elif i > j:  # 上采样匹配的操作层
                    layer = []
                    inchannels = self.old_branch_channels[i]
                    for k in range(i - j):
                        layer.append(
                            # 变换通道 -- 1x1
                            nn.Conv2D(inchannels, self.new_branch_channels[j],
                                      kernel_size=1, bias_attr=None)
                            )
                        layer.append(
                            # 归一化
                            nn.BatchNorm2D(self.new_branch_channels[j], momentum=BN_MOMENTUM)
                            )
                        layer.append(
                            # 激活函数
                            nn.ReLU()
                            )
                        layer.append(
                            # 上采样 scale: 2.
                            nn.Upsample(scale_factor=2.)
                            )
                        inchannels = self.new_branch_channels[j]
                    layer = nn.Sequential(*layer)
                elif i < j:   # 下采样匹配的操作层
                    layer = []
                    inchannels = self.old_branch_channels[i]
                    for k in range(j - i):
                        layer.append(
                            # 变换通道 -- 1x1
                            nn.Conv2D(inchannels, self.new_branch_channels[j],
                                      kernel_size=1, bias_attr=None)
                            )
                        layer.append(
                            # 下采样 scale: 1./2.
                            nn.Conv2D(self.new_branch_channels[j], self.new_branch_channels[j],
                                      kernel_size=3, stride=2, padding=1, bias_attr=None)
                            )
                        layer.append(
                            # 归一化
                            nn.BatchNorm2D(self.new_branch_channels[j], momentum=BN_MOMENTUM)
                            )
                        layer.append(
                            # 激活函数
                            nn.ReLU()
                            )
                        inchannels = self.new_branch_channels[j]
                    layer = nn.Sequential(*layer)
                branch_trans.append(layer)
            # 构建一个原始分支到新分支的输出
            branch_trans = nn.LayerList(branch_trans)
            totrans_layers.append(branch_trans)
        
        return nn.LayerList(totrans_layers)
            

class NormalBlock(nn.Layer):
    '''校验完成：
        只进行前向传播的普通卷积block

        params:
            in_channels:  输入通道数
            out_channels: 输出通道数
    '''
    def __init__(self, in_channels, out_channels):
        super(NormalBlock, self).__init__()

        self.conv1 = HRNetConv3x3(in_channels=in_channels, out_channels=out_channels, stride=1, padding=1)
        self.conv2 = HRNetConv3x3(in_channels=out_channels, out_channels=out_channels, stride=1, padding=1)
    

    def forward(self, inputs):

        x = self.conv1(inputs)
        x = self.conv2(x)

        return x


class ResidualBlock(nn.Layer):
    '''校验完成：
        大小/通道不变的残差卷积block

        params:
            in_channels:  输入通道数
            out_channels: 输出通道数
    '''
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()

        self.conv1 = HRNetConv3x3(in_channels=in_channels, out_channels=out_channels, padding=1)
        
        self.conv2 = nn.Conv2D(in_channels=out_channels, out_channels=out_channels, kernel_size=3,
                               padding=1, bias_attr=None)
        self.bn2   = nn.BatchNorm2D(out_channels, momentum=BN_MOMENTUM)
        self.relu  = nn.ReLU()


    def forward(self, inputs):
        residual = inputs
        
        x = self.conv1(inputs)

        x = self.conv2(x)
        x = self.bn2(x)

        x += residual
        x = self.relu(x)

        return x


class HRNetInput(nn.Layer):
    '''校验完成：
        构建HRNet的输入部分

        params:
            in_channels:       图片输入特征
            out_channels:      输入图片浅层特征提取的通道数
            stage1_inchannels: stage1块的输入通道 -- 对应HRNet的WIDTH
    '''
    def __init__(self, in_channels, out_channels, stage1_inchannels):
        super(HRNetInput, self).__init__()

        self.stem = HRNetStem(in_channels, out_channels)  # OUTPUT WIDTH 64
        # 转换为stage1对应的通道数
        self.in_change_conv = nn.Conv2D(in_channels=out_channels, out_channels=stage1_inchannels, kernel_size=1, 
                                        stride=1, bias_attr=None)
        self.in_change_bn   = nn.BatchNorm2D(stage1_inchannels, momentum=BN_MOMENTUM)
        self.relu           = nn.ReLU()
    

    def forward(self, inputs):
        
        x = self.stem(inputs)
        x = self.in_change_conv(x)
        x = self.in_change_bn(x)
        x = self.relu(x)

        return x



class HRNetStage(nn.Layer):
    '''校验完成：stage块网络层的生成
        构建Stage块
        这里固定每一个分支的block数为4

        params:
            stage_channels: 当前stage的分支中通道数列表
            block:          stage中每一个block的类型
    '''
    def __init__(self, stage_channels, block):
        super(HRNetStage, self).__init__()

        assert isinstance(stage_channels, list), \
            "Please make sure the stage_channels type is list({0}) in HRNetStage".format(type(stage_channels))

        self.stage_channels = stage_channels
        self.stage_branch_num = len(stage_channels)
        self.block          = block
        self.block_num      = 4

        # 构建整个stage中的所有分支block
        self.stage_layers = self.create_stage_layers()

    
    def forward(self, inputs):
        assert isinstance(inputs, list), \
            "Please make sure input data is list({0}) in the HRNetStage forward".format(type(inputs))

        outs = []

        # 遍历分支，每一个分支进行对应的连续block操作
        for i in range(self.stage_branch_num):
            x = inputs[i]
            y = self.stage_layers[i](x)  # 经过对应的block路径
            outs.append(y)
        
        return outs


    def create_stage_layers(self):
        '''
            创建stage网络层的所有分支路径
        '''
        tostage_layers = []

        # 遍历每一个分支
        for i in range(self.stage_branch_num):
            branch_layer = []
            # 每个分支形成四个连续的block
            for j in range(self.block_num):
                branch_layer.append(self.block(self.stage_channels[i], self.stage_channels[i]))
            branch_layer = nn.Sequential(*branch_layer)
            # 放入整体的stage中
            tostage_layers.append(branch_layer)
        
        return nn.LayerList(tostage_layers)

# keep: 仅输出
# fuse: 融合输出高分辨率
# multi: 融合多尺度输出
Fusion_Mode = ['keep', 'fuse', 'multi']

class HRNetFusion(nn.Layer):
    '''校验完成：输出的head，执行不同模式的特征融合

        params:
            stage4_channels: list -- 最后一个stage的分支通道数列表
            mode: 融合模式
    '''
    def __init__(self, stage4_channels, mode='keep'):
        super(HRNetFusion, self).__init__()

        assert mode in Fusion_Mode, \
            "Please make sure mode({0}) is in ['keep', 'fuse', 'multi'].".format(mode)

        self.stage4_channels = stage4_channels
        self.mode = mode

        # 构建融合层 -- 自动根据模式生成对应的融合层
        self.fuse_layers = self.create_fusion_layers()

    def forward(self, inputs):
        assert isinstance(inputs, list), \
            "Please make sure input data is list({0}) in HRNetOutput.".format(type(inputs))
            
        # 取出不同分支的输入
        x1, x2, x3, x4 = inputs
        outs = []
        
        if self.mode == Fusion_Mode[0]:    # keep
            out = self.fuse_layers(x1)
            outs.append(out)
        elif self.mode == Fusion_Mode[1]:  # fuse
            out  = self.fuse_layers[0](x1)
            out += self.fuse_layers[1](x2)
            out += self.fuse_layers[2](x3)
            out += self.fuse_layers[3](x4)
            outs.append(out)
        elif self.mode == Fusion_Mode[2]:  # multi
            out1  = self.fuse_layers[0][0](x1)
            out1 += self.fuse_layers[0][1](x2)
            out1 += self.fuse_layers[0][2](x3)
            out1 += self.fuse_layers[0][3](x4)
            outs.append(out1)
            out2  = self.fuse_layers[1](out1)
            outs.append(out2)
            out3  = self.fuse_layers[2](out2)
            outs.append(out3)
            out4  = self.fuse_layers[3](out3)
            outs.append(out4)
            
        return outs

    def create_fusion_layers(self):
        '''
            根据模式，生成合适的聚合层
        '''
        layer = None
        
        if self.mode == Fusion_Mode[0]:    # keep
            layer = self.create_keep_fusion_layers()
        elif self.mode == Fusion_Mode[1]:  # fuse
            layer = self.create_fuse_fusion_layers()
        elif self.mode == Fusion_Mode[2]:  # multi
            layer = self.create_multi_fusion_layers()
        
        return layer
    
    def create_keep_fusion_layers(self):
        '''
            进输出最高分辨率的特征表示
        '''
        self.outchannels = [self.stage4_channels[0]]*4 # list
    
    def create_fuse_fusion_layers(self):
        '''
            融合各个分辨率，通道保持最大(不同于原论文进行通道扩增)，接着上采样
        '''
        layers = []
        outchannel = self.stage4_channels[3]   # keep max channel -- 统一目标通道数

        # 遍历分支
        for i in range(0, len(self.stage4_channels)):
            inchannel = self.stage4_channels[i]  # 得到每一个分支的通道数
            layer = []

            # 不是第一个分支，就需要进行通道数转换，匹配通道数
            if i != len(self.stage4_channels) - 1:
                layer.append(nn.Conv2D(inchannel, outchannel, kernel_size=1, bias_attr=None))
                layer.append(nn.BatchNorm2D(outchannel, momentum=BN_MOMENTUM))
                layer.append(nn.ReLU())

            # 根据当前的index，也对应着离第一个分支的距离(index差)
            # 每一个距离，对应一个上采样，进行大小匹配
            for j in range(i):
                layer.append(nn.Upsample(scale_factor=2.))
            
            layer = nn.Sequential(*layer)
            layers.append(layer)
        
        self.outchannels = [outchannel]*4  # list
        return nn.LayerList(layers)

    def create_multi_fusion_layers(self):
        '''
            多尺度融合，并保持多分辨率输出
        '''
        multi_fuse_layers = []

        max_resolution_fuse_layers = []
        outchannel = self.stage4_channels[3]   # keep max channel

        for i in range(0, len(self.stage4_channels)):
            inchannel = self.stage4_channels[i]
            layer = []

            if i != len(self.stage4_channels) - 1:
                layer.append(nn.Conv2D(inchannel, outchannel, kernel_size=1, bias_attr=None))
                layer.append(nn.BatchNorm2D(outchannel, momentum=BN_MOMENTUM))
                layer.append(nn.ReLU())

            for j in range(i):
                layer.append(nn.Upsample(scale_factor=2.))
            
            layer = nn.Sequential(*layer)
            max_resolution_fuse_layers.append(layer)
        max_resolution_fuse_layers = nn.LayerList(max_resolution_fuse_layers)
        multi_fuse_layers.append(max_resolution_fuse_layers) # branch1

        # 其它分辨率就进行downsample即可
        # 由branch1 产生 branch2
        multi_fuse_layers.append(
            nn.Sequential(
                nn.Conv2D(outchannel, outchannel, kernel_size=3, stride=2, padding=1, bias_attr=None),
                nn.BatchNorm2D(outchannel, momentum=BN_MOMENTUM),
                nn.ReLU()
            )
        )
        # 由branch2 产生 branch3
        multi_fuse_layers.append(
            nn.Sequential(
                nn.Conv2D(outchannel, outchannel, kernel_size=3, stride=2, padding=1, bias_attr=None),
                nn.BatchNorm2D(outchannel, momentum=BN_MOMENTUM),
                nn.ReLU()
            )
        )
        # 由branch3 产生 branch4
        multi_fuse_layers.append(
            nn.Sequential(
                nn.Conv2D(outchannel, outchannel, kernel_size=3, stride=2, padding=1, bias_attr=None),
                nn.BatchNorm2D(outchannel, momentum=BN_MOMENTUM),
                nn.ReLU()
            )
        )

        self.outchannels = [outchannel]*4 # 当前统一的输出通道数 -- list
        return nn.LayerList(multi_fuse_layers)


class HRNet(nn.Layer):
    '''校验完成：支持预测分类与输出2048-1x1特征图
        HRNet复现版本（WIDTH=32）中，默认以[32, 64, 128, 256]的通道拓展

        V1   OUTPUT: 直接输出最大分辨率[64x64]的特征 -- 通道不变(32)  -- mode:keep
        V2   OUTPUT: 融合输出最大分辨率[64x64]的特征 -- 通道扩展(256) -- mode:fuse
        V2P  OUTPUT: 融合输出多种分辨率[64x64, 32x32, 16x16, 8x8]的特征 -- 通道扩展(256) -- mode:multi

        > 输出没有经过softmax

        params:
            width:       网络宽度
            mode:        head融合输出的模式
                         keep： 直接输出最大分辨率的特征图
                         fuse:  融合多分辨率后再输出最大分辨率的特征图
                         multi: 融合多分辨率，并输出多分辨率的多个特征图
    '''
    def __init__(self, width=32, mode='fuse'):
        super(HRNet, self).__init__()
        self.width       = width           # 网络的宽度 -- width

        # 这里学习方便，就直接填入对应的通道数
        if self.width == 16:
            self.n_stage_channels = [[16], [16, 32], [16, 32 ,64], [16, 32 ,64, 128]]
        elif self.width == 32:
            self.n_stage_channels = [[32], [32, 64], [32, 64 ,128], [32, 64 ,128, 256]]
        elif self.width == 64:
            self.n_stage_channels = [[64], [64, 128], [64, 128 ,256], [64, 128 ,256, 512]]
        elif self.width == 128:
            self.n_stage_channels = [[128], [128, 256], [128, 256 ,512], [128, 256 ,512, 1024]]

        # 构建HRNet的输入 -- 3 to 64 to 32
        self.hrnet_input = HRNetInput(in_channels=3, out_channels=64, stage1_inchannels=self.width)  # OUTPUT WIDTH 64
        
        # stage1的特征提取部分 -- 支持block：NormalBlock, ResidualBlock
        self.stage1 = HRNetStage(stage_channels=self.n_stage_channels[0], block=ResidualBlock)
        self.trans_stage1to2 = HRNetTrans(self.n_stage_channels[0], self.n_stage_channels[1])  # 分支融合与拓展层

        # stage2的特征提取部分
        self.stage2 = HRNetStage(stage_channels=self.n_stage_channels[1], block=ResidualBlock)
        self.trans_stage2to3 = HRNetTrans(self.n_stage_channels[1], self.n_stage_channels[2])
        
        # stage3的特征提取部分
        self.stage3 = HRNetStage(stage_channels=self.n_stage_channels[2], block=ResidualBlock)
        self.trans_stage3to4 = HRNetTrans(self.n_stage_channels[2], self.n_stage_channels[3])

        # stage4的特征提取部分
        self.stage4 = HRNetStage(stage_channels=self.n_stage_channels[3], block=ResidualBlock)
        
        # 针对stage4的输出进行特征融合输出(head)
        self.hrnet_fusion = HRNetFusion(self.n_stage_channels[3], mode=mode)


    def forward(self, inputs):
        x = self.hrnet_input(inputs)
        x = [x]  # list化

        # stage1操作
        x = self.stage1(x)  # return x 为 list
        x = self.trans_stage1to2(x)  # 分支融合拓展 1-2

        # stage2操作
        x = self.stage2(x)
        x = self.trans_stage2to3(x)  # 分支融合拓展 2-3

        # stage3操作
        x = self.stage3(x)
        x = self.trans_stage3to4(x)  # 分支融合拓展 3-4

        # stage4操作
        x = self.stage4(x)

        # representation head  -- 输出融合层
        x = self.hrnet_fusion(x) # return x 为 list

        return x


if __name__ == "__main__":
    # 模型实例
    # keep, fuse, multi
    model = HRNet(width=32, mode='multi')
    
    # 构建模拟输入
    data = np.random.randint(0, 256, (1, 3, 256, 256)).astype(np.float32)
    data = paddle.to_tensor(data)

    # 输入预测 -- 返回list
    # 如果 list 长度为1，则只包含高分辨率的输出
    y_preds = model(data)

    # 遍历输出
    for i in range(len(y_preds)):
        print(y_preds[i].shape)
