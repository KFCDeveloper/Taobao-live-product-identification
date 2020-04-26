# coding=utf-8

import tensorflow as tf
from tensorflow.keras.models import Model

# scale for l2 regularization
weight_decay = 0.00025
# scale for fully connected layer's l2 regularization
fc_weight_decay = 0.00025

BN_EPSILON = 0.001
NUM_LABELS = 6


def create_variables(name, shape, initializer=tf.initializers.GlorotUniform(), is_fc_layer=False):
    """
        :param name: A string. The name of the new variable
        :param shape: A list of dimensions
        :param initializer: User Xavier as default.
        :param is_fc_layer: Want to create fc layer variable? May use different weight_decay for fc
        layers.(fc: fully connected)
        :return: The created variable
    """
    if is_fc_layer is True:
        regularizer = tf.keras.regularizers.l2(fc_weight_decay)
    else:
        regularizer = tf.keras.regularizers.l2(weight_decay)
    new_variables = tf.Variable(name=name, shape=shape, initializer=initializer, regularizer=regularizer)
    return new_variables


def output_layer(input_layer, num_labels):
    input_dim = input_layer.get_shape().as_list()[-1]
    # VarianceScaling 方差由scale定，保持 不变
    fc_w = create_variables(name='fc_weights', shape=[input_dim, num_labels], is_fc_layer=True,
                            initializer=tf.keras.initializers.VarianceScaling(scale=1.0))

    fc_b = create_variables(name='fc_bias', shape=[num_labels], initializer=tf.zeros_initializer)

    fc_w2 = create_variables(name='fc_weights2', shape=[input_dim, 4], is_fc_layer=True,
                             initializer=tf.keras.initializers.VarianceScaling(scale=1.0))
    fc_b2 = create_variables(name='fc_bias2', shape=[4], initializer=tf.zeros_initializer)
    fc_h = tf.matmul(input_layer, fc_w) + fc_b
    fc_h2 = tf.matmul(input_layer, fc_w2) + fc_b2
    return fc_h, fc_h2


# 这个函数先卷积 再 归一化
def conv_bn_relu_layer(input_layer, filter_shape, stride, second_conv_residual=False,
                       relu=True):
    out_channel = filter_shape[-1]  # filter 最后一个为输出channel
    if second_conv_residual is False:
        filter = create_variables(name='conv', shape=filter_shape)
    else:
        filter = create_variables(name='conv2', shape=filter_shape)

    conv_layer = tf.nn.conv2d(input_layer, filter, strides=[1, stride, stride, 1], padding='SAME')

    mean, variance = tf.nn.moments(conv_layer, axes=[0, 1, 2])
    if second_conv_residual is False:
        beta = tf.Variable("beta", out_channel, tf.float32, initializer=tf.constant_initializer(0.0))
        gamma = tf.Variable('gamma', out_channel, tf.float32, initializer=tf.constant_initializer(1.0))
    else:
        beta = tf.Variable('beta_second_conv', out_channel, tf.float32, initializer=tf.constant_initializer(0.0))
        gamma = tf.Variable('gamma_second_conv', out_channel, tf.float32, initializer=tf.constant_initializer(1.0))
    # 输入样本；均值；方差；样本偏移；缩放；是为了避免分母为0，添加的一个极小值         ；； y=scale∗(x−mean)/var+offset
    bn_layer = tf.nn.batch_normalization(conv_layer, mean, variance, beta, gamma, BN_EPSILON)

    if relu:
        output = tf.nn.relu(bn_layer)
    else:
        output = bn_layer
    return output


# 这个函数先归一化 再 卷积
def bn_relu_conv_layer(input_layer, filter_shape, stride, second_conv_residual=False):
    in_channel = input_layer.get_shape().as_list()[-1]  # 从 filter_shape 应该也是可以获取到的
    mean, variance = tf.nn.moments(input_layer, axes=[0, 1, 2])
    if second_conv_residual is False:
        beta = tf.Variable('beta', in_channel, tf.float32, initializer=tf.constant_initializer(0.0))
        gamma = tf.Variable('gamma', in_channel, tf.float32, initializer=tf.constant_initializer(1.0))
    else:
        beta = tf.Variable('beta_second_conv', in_channel, tf.float32, initializer=tf.constant_initializer(0.0))
        gamma = tf.Variable('gamma_second_conv', in_channel, tf.float32, initializer=tf.constant_initializer(1.0))
    bn_layer = tf.nn.batch_normalization(input_layer, mean, variance, beta, gamma, BN_EPSILON)
    relu_layer = tf.nn.relu(bn_layer)
    if second_conv_residual is False:
        filter = create_variables(name='conv', shape=filter_shape)
    else:
        filter = create_variables(name='conv2', shape=filter_shape)
    conv_layer = tf.nn.conv2d(relu_layer, filter, strides=[1, stride, stride, 1], padding='SAME')
    return conv_layer


def residual_block_new(input_layer, output_channel, first_block=False):
    input_channel = input_layer.get_shape().as_list()[-1]
    if input_channel * 2 == output_channel:
        increase_dim = True
        stride = 2
    elif input_channel == output_channel:
        increase_dim = False
        stride = 1
    else:
        raise ValueError('Output and input channel does not match in residual blocks!!!')

    if first_block:
        #  不归一化
        filter_temp = create_variables(name='conv', shape=[3, 3, input_channel, output_channel])
        conv1 = tf.nn.conv2d(input_layer, filter=filter_temp, strides=[1, 1, 1, 1], padding='SAME')
    else:
        # bn_relu_conv_layer 进行了正则化
        conv1 = bn_relu_conv_layer(input_layer, [3, 3, input_channel, output_channel], stride)
    conv2 = bn_relu_conv_layer(conv1, [3, 3, output_channel, output_channel], 1, second_conv_residual=True)

    if increase_dim is True:
        pooled_input = tf.nn.avg_pool(input_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  # 池化
        # 对四维的张量进行填充，在channel的前后进行填充
        padded_input = tf.pad(pooled_input, [[0, 0], [0, 0], [0, 0], [input_channel // 2, input_channel // 2]])
    else:
        padded_input = input_layer

    output = conv2 + padded_input
    return output


def inference(input_tensor_batch, n, reuse, keep_prob_placeholder):
    layers = []
    conv0 = conv_bn_relu_layer(input_tensor_batch, [3, 3, 3, 16], 1)

    for i in range(n):
        if i == 0:
            conv1 = residual_block_new(layers[-1], 16, first_block=True)
        else:
            conv1 = residual_block_new(layers[-1], 16)
