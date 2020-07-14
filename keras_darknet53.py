# -*- coding: utf-8 -*-
# @Time : 2020/7/14 上午10:47
# @Author : zhufa
# @Software: PyCharm
# @discription: darnet53实现

import tensorflow as tf

def darknet53(input_shape, classes = 45):
    X_input = Input(input_shape)
    '''Darknent body having 52 Convolution2D layers'''
    x = DarknetConv2D_BN_Leaky(X_input, 32, (3, 3))
    x = resblock_body(x, 64, 1)
    x = resblock_body(x, 128, 2)
    x = resblock_body(x, 256, 8)
    x = resblock_body(x, 512, 8)
    x = resblock_body(x, 1024, 4)
    x = GlobalAveragePooling2D()(x) #全局平均池化，一个样本转换成特征图数量的向量
    x = Dense(classes, activation='softmax', name='fc'+str(classes),kernel_initializer=glorot_uniform(seed=0))(x)
    model = Model(input=X_input, output=x, name='darknet53')
    return model