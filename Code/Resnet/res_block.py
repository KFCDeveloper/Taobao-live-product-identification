# -*- coding: utf-8 -*-
import tensorflow as tf


class ResBlock(object):
    def __init__(self, input_layer, output_channel, first_block=False, strides=(1, 1)):
        self.input_layer = input_layer
        self.output_channel = output_channel
        self.strides = strides
        self.first_block = first_block

        input_channel = input_layer.get_shape().as_list()[-1]
        if input_channel * 2 == output_channel:
            self.increase_dim = True
            self.stride = 2
        elif input_channel == output_channel:
            self.increase_dim = False
            self.stride = 1
        else:
            raise ValueError('Output and input channel does not match in residual blocks!!!')

    def __call__(self, x):
        shortcut = x
        input_channel = x.get_shape().as_list()[-1]

        if self.first_block:  # 如果为第一个块 不归一化
            x = tf.keras.layers.Conv2D(self.output_channel, (3, 3),
                                       kernel_regularizer=tf.keras.regularizers.l2(0.00025), strides=1, use_bias=False,
                                       padding='same')(x)
        else:  # 要归一化
            # 不需要像之前的batch_normalization要传入variance 和 mean ，这里包装的很好
            x = tf.keras.layers.BatchNormalization(x)
            x = tf.keras.layers.ReLU()(x)
            x = tf.keras.layers.Conv2D(self.output_channel, (3, 3),
                                       kernel_regularizer=tf.keras.regularizers.l2(0.00025), strides=self.stride,
                                       use_bias=False,
                                       padding='same')(x)
        x = tf.keras.layers.BatchNormalization(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Conv2D(self.output_channel, (3, 3), kernel_regularizer=tf.keras.regularizers.l2(0.00025),
                                   strides=self.stride, use_bias=False, padding='same')(x)
        if self.increase_dim is True:
            shortcut = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), )(shortcut)
            padded_input = tf.pad(shortcut, [[0, 0], [0, 0], [0, 0], [input_channel // 2, input_channel // 2]])
        else:
            padded_input = shortcut
        x = tf.keras.layers.add([x, padded_input])
        return x
