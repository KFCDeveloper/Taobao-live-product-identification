# coding=utf-8
import tensorflow as tf
from Resnet.res_block import ResBlock

NUM_LABELS = 23  # 淦，这里之前是6，我忘记改了，导致后面sparse_softmax_cross_entropy_with_logits返回Nan


def ResNet_6N_2(shape, n):
    inputs = tf.keras.Input(shape, name="img")  # shape是三维的，不包括batch_size
    x = inputs
    x = tf.keras.layers.Conv2D(16, (3, 3), kernel_regularizer=tf.keras.regularizers.l2(0.00025), strides=1,
                               use_bias=False, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x, training=True)  # ()后面(x)前面的()别掉了
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

    x = tf.keras.layers.BatchNormalization()(x, training=True)
    x = tf.keras.layers.ReLU()(x)
    # axis = i，则沿着第i个下标变化的方向进行操作
    global_pool = tf.reduce_mean(x, [1, 2])
    assert global_pool.get_shape().as_list()[-1:] == [64]
    # 初始化输出层
    output_layer = OutputLayer([global_pool, NUM_LABELS])
    cls_output, bbx_output = output_layer(global_pool)
    model = tf.keras.Model(inputs=inputs, outputs=[cls_output, bbx_output, global_pool], name="ResNet_6N_2")
    # 由于我在主要的training已经手动 apply了梯度下降，所以这里应该不用配置模型了
    # model.compile(optimizer=tf.keras.optimizers.Adam(), loss="categorical_crossentropy", metrics=["acc"])
    return model


class OutputLayer(tf.keras.layers.Layer):
    def __init__(self, input_shape):
        super(OutputLayer, self).__init__()
        input_layer, num_labels = input_shape
        input_dim = input_layer.get_shape().as_list()[-1]
        # VarianceScaling 方差由scale定，保持 不变
        # TODO: 正则化这里可以填 0.001之类的
        self.fc_w = self.add_weight(name="fc_w", shape=[input_dim, num_labels],
                                    initializer=tf.keras.initializers.VarianceScaling(scale=1.0),
                                    regularizer=tf.keras.regularizers.l2(0.00025), trainable=True)
        self.fc_b = self.add_weight(name="fc_b", shape=[num_labels],
                                    initializer=tf.keras.initializers.VarianceScaling(scale=1.0),
                                    regularizer=tf.keras.regularizers.l2(0.00025), trainable=True)
        self.fc_w2 = self.add_weight(name="fc_w2", shape=[input_dim, 4],
                                     initializer=tf.keras.initializers.VarianceScaling(scale=1.0),
                                     regularizer=tf.keras.regularizers.l2(0.00025), trainable=True)
        self.fc_b2 = self.add_weight(name="fc_b2", shape=[4],
                                     initializer=tf.keras.initializers.VarianceScaling(scale=1.0),
                                     regularizer=tf.keras.regularizers.l2(0.00025), trainable=True)

    def call(self, input_layer, **kwargs):
        fc_h = tf.matmul(input_layer, self.fc_w) + self.fc_b
        fc_h2 = tf.matmul(input_layer, self.fc_w2) + self.fc_b2
        return fc_h, fc_h2


if __name__ == '__main__':
    # 输出ResNet的结构
    model = ResNet_6N_2(shape=[64, 64, 3], n=2)
    tf.keras.utils.plot_model(model, to_file='model3.png', show_shapes=True, show_layer_names=True)
