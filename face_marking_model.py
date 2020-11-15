# !usr/bin/env python
# -*- coding: utf-8 -*-
# @Date: 2020/6/15
# @Author: Koorye

import tensorflow as tf
import numpy as np
import pandas as pd


class model:
    def __init__(self):
        self.x_train, self.x_test, self.y_train, self.y_test = 0, 0, 0, 0

        self.model = tf.keras.Sequential(
            # 卷积层，filters指定卷积核数量，kernal_size指定卷积核尺寸
            # padding指定是否使用全零填充
            tf.keras.layers.Conv2D(filters=6, kernal_size=(5, 5), padding='same'),
            # 批标准化
            tf.keras.layers.BatchNormalization(),
            # relu激活函数
            tf.keras.layers.Activation('relu'),
            # 池化，pool_size指定池尺寸，strides指定池间隔，默认与尺寸一致
            tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same'),
            # 随机失活，指定每次失活20%的神经元
            tf.keras.layers.Dropout(0.2),
            # 数据平铺成一维数组
            tf.keras.layers.Flatten(),
            # 全连接，128个神经元，relu激活函数
            tf.keras.layers.Dense(128, activation='relu'),
            # 随机失活
            tf.keras.layers.Dropout(0.2),
            # 全连接，10个神经元，softmax激活函数，用于解决多分类问题
            tf.keras.layers.Dense(10, activation='softmax')
        )

    def fit(self):
        # 模型编译，optimize指定adam优化器，loss指定损失函数为均方差
        model.compile(optimize='adam', loss='mse')

        # 模型训练，batch_size指定每次喂入的数据量，epochs指定训练次数
        # validation_data指定验证集，validation_freq指定多少次训练进行一次验证
        model.fit(x_train, y_train, batch_size=32, epochs=5, validation_data=(x_test, y_test), validation_freq=1)
