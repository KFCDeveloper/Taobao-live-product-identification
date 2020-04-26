# coding=utf-8
# 本文件乃之前还未掌握 TensorFlow2.0的手动训练之前的错误的做法
# 放在这里主要是我编写新的 手动训练 的时候需要用到我之前的成果
import tensorflow as tf
import tensorflow.keras as tk

from .Resnet.model import ResNet_6N_2
from .util.util_function import *
from .util.const import const

REPORT_FREQ = 50
TRAIN_BATCH_SIZE = 32
VALI_BATCH_SIZE = 25
TEST_BATCH_SIZE = 25
FULL_VALIDATION = False
Error_EMA = 0.98

STEP_TO_TRAIN = 45000
DECAY_STEP0 = 25000
DECAY_STEP1 = 35000


def loss(model, input_image, input_label, input_bbox):
    logits, bbox, _ = model(input_image)
    # 先 SoftMax 然后计算Cross-Entropy   传入的logits为神经网络输出层的输出
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=input_label,
                                                                   name='cross_entropy_per_example')
    mse_loss = tf.reduce_mean(tf.losses.mean_squared_error(input_bbox, bbox), name='mean_square_loss')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    return cross_entropy_mean + mse_loss


def top_k_error(predictions, input_label, k):
    batch_size = predictions.get_shape().as_list()[0]
    in_top1 = tf.nn.in_top_k(predictions, input_label, k=k)
    num_correct = tf.reduce_sum(in_top1)
    return (batch_size - num_correct) / float(batch_size)


def train_operation(base_model, global_step, total_loss, top1_error, lr):
    tf.summary.scalar('learning_rate', lr)
    tf.summary.scalar('train_loss', total_loss)
    tf.summary.scalar('train_top1_error', top1_error)

    ema = tf.train.ExponentialMovingAverage(0.95, global_step)
    # 这个 apply 返回的是一个操作，每apply一下，就会更新一次参数
    train_ema_op = ema.apply([total_loss, top1_error])
    tf.summary.scalar('train_top1_error_avg', ema.average(top1_error))
    tf.summary.scalar('train_loss_avg', ema.average(total_loss))

    # TODO:后面可以改成 Adam来试一下，
    opt = tf.keras.optimizers.RMSprop(learning_rate=lr, momentum=0.9)
    train_op = opt.minimize(total_loss, var_list=lambda: base_model.trainable_weights)
    # 我也不是很确认 # 这个是lambda无参数形式，这样可以让var_list变成可以调用的函数，就返回后面那坨 # var_list() 这就是可调用
    return train_op, train_ema_op


def validation_op(validation_step, top1_error, loss):
    ema = tf.train.ExponentialMovingAverage(0.0, validation_step)
    ema2 = tf.train.ExponentialMovingAverage(0.95, validation_step)
    # group返回的也是operation       # validation_step.assign_add(1)会让validation_step加1，可能是会循环的
    val_op = tf.group(validation_step.assign_add(1), ema.apply([top1_error, loss]), ema2.apply([top1_error, loss]))
    top1_error_val = ema.average(top1_error)
    top1_error_avg = ema2.average(top1_error)
    loss_val = ema.average(loss)
    loss_val_avg = ema2.average(loss)
    tf.summary.scalar('val_top1_error', top1_error_val)
    tf.summary.scalar('val_top1_error_avg', top1_error_avg)
    tf.summary.scalar('val_loss', loss_val)
    tf.summary.scalar('val_loss_avg', loss_val_avg)
    return val_op


def full_validation(validation_df, sess, vali_loss, vali_top1_error, batch_data, batch_label, batch_bbox):
    num_batches = len(validation_df) // VALI_BATCH_SIZE  # //表示整数除法
    error_list = []
    loss_list = []

    for i in range(num_batches):
        offset = i * VALI_BATCH_SIZE
        vali_batch_df = validation_df.iloc[offset:offset + VALI_BATCH_SIZE, :]
        validation_image_batch, validation_labels_batch, validation_bbox_batch = load_data_numpy(vali_batch_df)

        # vali_error, vali_loss_value = sess.run([vali_top1_error, vali_loss],
        #                                        {input_image: batch_data,
        #                                         input_label: batch_label,
        #                                         input_bbox: batch_bbox,
        #                                         input_vali_image: validation_image_batch,
        #                                         input_vali_label: validation_labels_batch,
        #                                         input_vali_bbox: validation_bbox_batch,
        #                                         input_lr: FLAGS.learning_rate,
        #                                         input_dropout_prob: 0.5})
        # error_list.append(vali_error)
        # loss_list.append(vali_loss_value)

    return np.mean(error_list), np.mean(loss_list)


def train(self):
    # TODO: 这里的输入是最为简单的输入
    train_df = prepare_df(const.train_path,
                          usecols=['image_path', 'category', 'x1_modified', 'y1_modified', 'x2_modified',
                                   'y2_modified'])
    vali_df = prepare_df(const.vali_path,
                         usecols=['image_path', 'category', 'x1_modified', 'y1_modified', 'x2_modified',
                                  'y2_modified'])
    num_train = len(train_df)

    # 定义输入
    input_image = tf.keras.Input([TRAIN_BATCH_SIZE, IMG_ROWS, IMG_COLS, 3])
    input_label = tf.keras.Input([TRAIN_BATCH_SIZE])
    input_bbox = tf.keras.Input([TRAIN_BATCH_SIZE, 4])
    input_vali_image = tf.keras.Input([VALI_BATCH_SIZE, IMG_ROWS, IMG_COLS, 3])
    input_vali_label = tf.keras.Input([VALI_BATCH_SIZE])
    input_vali_bbox = tf.keras.Input([VALI_BATCH_SIZE, 4])
    input_lr = tf.keras.Input([])
    input_dropout_prob = tf.keras.Input([])

    # 取得 ResNet 输出
    base_model = ResNet_6N_2(shape=[TRAIN_BATCH_SIZE, IMG_ROWS, IMG_COLS, 3], n=const.num_residual_blocks)
    base_model_vali = ResNet_6N_2(shape=[TRAIN_BATCH_SIZE, IMG_ROWS, IMG_COLS, 3], n=const.num_residual_blocks)

    logits, bbox, _ = base_model(input_image)  # logits 是输出的label
    vali_logits, vali_bbox, _ = base_model_vali(input_vali_image)

    # 定义 loss
    reg_losses = base_model.losses
    model_loss = loss(base_model, input_image, input_label, input_bbox)
    full_loss = tf.add_n([model_loss] + reg_losses)
    vali_loss = loss(base_model_vali, input_vali_image, input_vali_label, input_vali_bbox)

    # 定义 error
    predictions = tf.nn.softmax(logits)
    top1_error = top_k_error(predictions, input_label, 1)
    vali_predictions = tf.nn.softmax(vali_logits)
    vali_top1_error = top_k_error(vali_predictions, input_vali_label, 1)

    # 定义 参数平均滑动更新操作  和  优化器
    global_step = tf.Variable(tf.constant(0), trainable=False)
    train_op, train_ema_op = train_operation(base_model, global_step, full_loss, top1_error, input_lr)

    validation_step = tf.Variable(tf.constant(0), trainable=False)
    val_op = validation_op(validation_step, vali_top1_error, vali_loss)

    # TODO: 这里可能也要全改了，Keras 通过Callback接口来追踪训练过程的每一步结果
    for step in range(STEP_TO_TRAIN):
        # TODO: 这里还需修改，generator_batch
        # 从 [0,num_train - TRAIN_BATCH_SIZE) 之间随便取一个
        offset = np.random.choice(num_train - TRAIN_BATCH_SIZE, 1)[0]

        train_batch_df = train_df.iloc[offset:offset + TRAIN_BATCH_SIZE, :]
        batch_data, batch_label, batch_bbox = load_data_numpy(train_batch_df)  # 要注意区别 epoch 和 batch 和 step

        if step == 0:
            if FULL_VALIDATION is True:
                top1_error_value, vali_loss_value = full_validation(vali_df,
                                                                    vali_loss=vali_loss,
                                                                    vali_top1_error=vali_top1_error,
                                                                    batch_data=batch_data,
                                                                    batch_label=batch_label,
                                                                    batch_bbox=batch_bbox)
