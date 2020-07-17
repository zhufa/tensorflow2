# -*- coding: utf-8 -*-
# @Time : 2020/7/16 下午3:49
# @Author : zhufa
# @Software: PyCharm
# @discription: 使用训练好的模型进行预测：1.加载保存的模型，2.模型预测
import tensorflow as tf
import numpy as np


# 方法内的代码复制自mnist.load_data()
def loadMinstData(path):
    with np.load(path, allow_pickle=True) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']

        return (x_train, y_train), (x_test, y_test)


path = 'dataset/mnist.npz'
(x_train, y_train),(x_test, y_test) = loadMinstData(path)
x_train, x_test = x_train / 255.0, x_test / 255.0

# 加载模型，可以选择模型1还是2
model = tf.keras.models.load_model('./model1/')
# 预览模型结构
model.summary()
# 用测试集测试模型准确度
model.evaluate(x_test, y_test)

# 预测结果1: 整个测试集输入模型去测试
predictions = model.predict(x_test)
# tf2.0中，张量的主要属性有三个，分别为形状(shape)、类型(dtype)和值(numpy())，可以通过张量实例的shape、dtype属性和numpy()方法来获取
result = tf.argmax(predictions, 1).numpy()
# 因result中有10000个预测结果不好全部打印，下面只输出第一张测试图的预测结果
print('10000 test dataset predict result is: ' + str(result))


# 预测结果2: 单个数据输入模型去测试
x= x_test[0]  # 取单个数据
# 整个数据集去测试，10000张即输入为10000×28×28，第一个维度对应模型中的batchsize维度
# 当你是传入一张图片(单个数据)去测试时，输入为28×28，会缺少一个维度，下面进行升维处理，方法有如下几种，结果为1×28×28
x0 = x.reshape(-1,28,28)
x1 = x[None,:,:]
x2 = np.expand_dims(x,axis=0)

predictions = model.predict(x2)
# tf2.0中，张量的主要属性有三个，分别为形状(shape)、类型(dtype)和值(numpy())，可以通过张量实例的shape、dtype属性和numpy()方法来获取
result = tf.argmax(predictions, 1).numpy()
# 因result中有10000个预测结果不好全部打印，下面只输出第一张测试图的预测结果
print('the single img predict result is: ' + str(result))
