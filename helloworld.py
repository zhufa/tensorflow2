# -*- coding: utf-8 -*-
# @Time : 2020/7/14 下午2:27
# @Author : zhufa
# @Software: PyCharm
# @discription: tensorflow2.0新手入门

import tensorflow as tf
import numpy as np

'''
官方代码使用mnist.load_data()联网下载mnist.npz数据集，但国内可能下载不了
解决办法：自己从网上下载数据集，复制mnist.load_data()里的加载数据集的代码，自己写方法实现

官方代码：
mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()
'''


# 方法内的代码复制自mnist.load_data()
def loadMinstData(path):
    with np.load(path, allow_pickle=True) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']

        return (x_train, y_train), (x_test, y_test)


path = 'mnist.npz'
(x_train, y_train),(x_test, y_test) = loadMinstData(path)
x_train, x_test = x_train / 255.0, x_test / 255.0

# 模型结构搭建
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

# 配置模型如损失函数和优化器等参数，准备训练
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 训练模型，batch_size默认32，那么每个batch训练60000/32=1875张图片
model.fit(x_train, y_train, batch_size=32, epochs=5)
# 测试模型的损失值和指标值（此处指精度，因为metrics=['accuracy']）
model.evaluate(x_test, y_test)
