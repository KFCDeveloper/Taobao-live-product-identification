# coding=utf-8
import tensorflow as tf
from .res_block import ResBlock

NUM_LABELS = 12


def ResNet_6N_2(shape, n):
    inputs = tf.keras.Input(shape, name="img")
    x = inputs
    x = tf.keras.layers.Conv2D(16, (3, 3), kernel_regularizer=tf.keras.regularizers.l2(0.00025), strides=1,
                               use_bias=False, padding='same')(x)
    x = tf.keras.layers.BatchNormalization(x)
    x = tf.keras.layers.ReLU()(x)
    for i in range(n):
        if i == 0:
            x = ResBlock(input_layer=x, output_channel=16, first_block=True)(x)
        else:
            x = ResBlock(input_layer=x, output_channel=16)(x)
    for _ in range(n):
        x = ResBlock(input_layer=x, output_channel=32)(x)
    for _ in range(n):
        x = ResBlock(input_layer=x, output_channel=64)(x)

    x = tf.keras.layers.BatchNormalization(x)
    x = tf.keras.layers.ReLU()(x)
    # axis = i，则沿着第i个下标变化的方向进行操作
    global_pool = tf.reduce_mean(x, [1, 2])
    assert global_pool.get_shape().as_list()[-1:] == [64]
    cls_output, bbx_output = output_layer(global_pool, NUM_LABELS)
    model = tf.keras.Model(inputs=inputs, outputs=[cls_output, bbx_output, global_pool], name="ResNet_6N_2")
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss="categorical_crossentropy", metrics=["acc"])
    return model


def output_layer(input_layer, num_labels):
    input_dim = input_layer.get_shape().as_list()[-1]
    # VarianceScaling 方差由scale定，保持 不变
    fc_w = tf.Variable(name='fc_weights', shape=[input_dim, num_labels], is_fc_layer=True,
                       initializer=tf.keras.initializers.VarianceScaling(scale=1.0))

    fc_b = tf.Variable(name='fc_bias', shape=[num_labels], initializer=tf.zeros_initializer)

    fc_w2 = tf.Variable(name='fc_weights2', shape=[input_dim, 4], is_fc_layer=True,
                        initializer=tf.keras.initializers.VarianceScaling(scale=1.0))
    fc_b2 = tf.Variable(name='fc_bias2', shape=[4], initializer=tf.zeros_initializer)
    fc_h = tf.matmul(input_layer, fc_w) + fc_b
    fc_h2 = tf.matmul(input_layer, fc_w2) + fc_b2
    return fc_h, fc_h2
