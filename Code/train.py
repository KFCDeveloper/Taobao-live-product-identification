# coding=utf-8
import os
import time
from datetime import datetime

import tensorflow as tf
import tensorflow.keras as tk

from Resnet.model import ResNet_6N_2
from util.main_model_util import *
from util.const import const

tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)

REPORT_FREQ = 50
TRAIN_BATCH_SIZE = 32
VALI_BATCH_SIZE = 25
TEST_BATCH_SIZE = 25
FULL_VALIDATION = False
Error_EMA = 0.98

STEP_TO_TRAIN = 45000
DECAY_STEP0 = 25000
DECAY_STEP1 = 35000
DIR_LOG_SUMMARY = "../Temp-File/Summary/"
TRAIN_DIR = "../Temp-File/Save-Weight/"
ERROR_SAVE_DIR = "../Temp-File/Save-Error/"
# 定义权重保存位置
train_min_checkpoint_path = TRAIN_DIR + 'min_train_model_checkpoint'
vali_min_checkpoint_path = TRAIN_DIR + 'min_vali_model_checkpoint'
train_checkpoint_path = TRAIN_DIR + 'train_model_checkpoint'
vali_checkpoint_path = TRAIN_DIR + 'vali_model_checkpoint'

# 指定日志目录
log_dir = DIR_LOG_SUMMARY + datetime.now().strftime("%Y%m%d-%H%M%S")
summary_writer = tf.summary.create_file_writer(log_dir)  # 创建日志文件句柄


def generate_validation_batch(df):
    """
    :param df: a pandas dataframe with validation image paths and the corresponding labels
    :return: two random numpy arrays: validation_batch and validation_label
    """
    offset = np.random.choice(len(df) - VALI_BATCH_SIZE, 1)[0]
    validation_df = df.iloc[offset:offset + VALI_BATCH_SIZE, :]

    validation_batch, validation_label, validation_bbox_label = load_data_numpy(validation_df)
    return validation_batch, validation_label, validation_bbox_label


class ResNetImprove(tk.Model):
    def __init__(self, is_validation, lr):
        super(ResNetImprove, self).__init__()
        self.train_loss = None
        self.top1_error = None
        self.is_validation = is_validation
        self.lr = lr

        # 定义 引入的ResNet
        self.basic_resnet_model = ResNet_6N_2(shape=[IMG_ROWS, IMG_COLS, 3], n=const.num_residual_blocks)

    # **kwargs 是传键值对的  TODO: 为什么要加这个
    def call(self, input_tensor, **kwargs):
        # TODO:注意 train_model 和 validation_model 是共用learn_rate 和 dropout_prob 的
        input_image, input_label, input_bbox, input_lr, input_dropout_prob = input_tensor
        logits, bbox, global_pool = self.basic_resnet_model(input_image)  # logits 是输出的label
        self.lr = input_lr if input_lr is not None else self.lr
        return logits, bbox, global_pool

    def top_k_error(self, predictions, input_label, k):
        batch_size = predictions.get_shape().as_list()[0]
        in_top1 = tf.math.in_top_k(input_label, predictions, k=k)  # 因为 tf.math.in_top_k 的api是反的
        temp = in_top1.numpy().astype(int)
        num_correct = tf.reduce_sum(tf.convert_to_tensor(temp))
        # 加上了name主要是我想要它变成tensor
        self.top1_error = tf.math.divide(float(batch_size - num_correct), float(batch_size), "top1_error")
        return self.top1_error

    def loss(self, logits, bbox, input_label, input_bbox):
        # 先 SoftMax 然后计算Cross-Entropy   传入的logits为神经网络输出层的输出
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=input_label,
                                                                       name='cross_entropy_per_example')
        mse_loss = tf.reduce_mean(tf.losses.mean_squared_error(input_bbox, bbox), name='mean_square_loss')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        return cross_entropy_mean + mse_loss

    def get_loss(self, input_tensor):
        # 解包入参,注意要传入的是列表
        input_image, input_label, input_bbox, epoch = input_tensor

        # 取得 ResNet 输出
        logits, bbox, _ = self.basic_resnet_model(input_tensor)  # logits 是输出的label

        # 定义 loss
        reg_losses = self.basic_resnet_model.losses
        model_loss = self.loss(logits, bbox, input_label, input_bbox)
        full_loss = tf.add_n([model_loss] + reg_losses)
        return full_loss

    def get_loss_validation(self, input_tensor):
        # 解包入参,注意要传入的是列表
        input_image, input_label, input_bbox, epoch = input_tensor

        # 取得 ResNet 输出
        logits, bbox, _ = self.basic_resnet_model(input_tensor)  # logits 是输出的label
        model_loss = self.loss(logits, bbox, input_label, input_bbox)
        # 保存记录 loss 方便输出
        self.train_loss = model_loss
        return model_loss

    def get_grad(self, input_tensor):
        with tf.GradientTape() as tape:
            tape.watch(self.trainable_variables)
            L = self.get_loss(input_tensor)
            # 保存一下loss，用于输出
            self.train_loss = L
            # tape.gradient就是求梯度，前面是f(x,y,z,,,,),后面是x
            g = tape.gradient(L, self.trainable_variables)  # 所有的权重都会被更新,之前是variables,然后改成trainable_variables
        return g

    def network_learn(self, input_tensor):
        # 解包入参,注意要传入的是列表
        input_image, input_label, input_bbox, epoch = input_tensor

        # 应用梯度更新权重  并且会初始化
        g = self.get_grad(input_tensor)
        tk.optimizers.Adam(learning_rate=self.lr).apply_gradients(zip(g, self.trainable_variables))

        # 更新输出
        logits, bbox, global_pool = self.basic_resnet_model(input_image)
        # 开始计算 top_k_error
        predictions = tf.nn.softmax(logits)
        self.top1_error = self.top_k_error(predictions, input_label, 1)

        # 画图
        with summary_writer.as_default():
            tf.summary.scalar('learning_rate', self.lr, epoch)
            tf.summary.scalar('train_loss', self.train_loss, epoch)
            tf.summary.scalar('train_top1_error', self.top1_error, epoch)
        # 先用 EMA 更新参数
        global_step = tf.Variable(tf.constant(0), trainable=False)
        ema = tf.train.ExponentialMovingAverage(0.95, global_step)
        # 需要将传入ema的qpply的参数变成 Variable 不然top1_error和train_loss无法改变
        self.top1_error = tf.Variable(self.top1_error)
        self.train_loss = tf.Variable(self.train_loss)
        train_ema_op = ema.apply([self.train_loss, self.top1_error])  # TODO: 我康康不返回 operation 会不会更新参数
        with summary_writer.as_default():
            tf.summary.scalar('train_top1_error_avg', ema.average(self.top1_error), epoch)
            tf.summary.scalar('train_loss_avg', ema.average(self.train_loss), epoch)
        return self.train_loss, self.top1_error

    def network_learn_validation(self, input_tensor):
        # 解包入参,注意要传入的是列表
        input_image, input_label, input_bbox, epoch = input_tensor
        # 初始化 或者 更新 loss
        self.get_loss_validation(input_tensor)
        # 更新输出
        logits, bbox, global_pool = self.basic_resnet_model(input_image)
        # 开始计算 top_k_error
        predictions = tf.nn.softmax(logits)
        self.top1_error = self.top_k_error(predictions, input_label, 1)
        # 下面就 EMA开始更新参数
        validation_step = tf.Variable(tf.constant(0), trainable=False)

        ema1 = tf.train.ExponentialMovingAverage(0.0, validation_step)
        ema2 = tf.train.ExponentialMovingAverage(0.95, validation_step)

        # 需要将传入ema的qpply的参数变成 Variable 不然top1_error和train_loss无法改变
        self.top1_error = tf.Variable(self.top1_error)
        self.train_loss = tf.Variable(self.train_loss)
        # group返回的也是operation       # validation_step.assign_add(1)会让validation_step加1，可能是会循环的
        val_op = tf.group(validation_step.assign_add(1), ema1.apply([self.top1_error, self.train_loss]),
                          ema2.apply([self.top1_error, self.train_loss]))
        # ema1.average(val) 表示去取得val更新后的值，不是指的再求一次平均值
        # TODO:我要看看这里的 top1_error_val 和 top1_error_avg 是否是一样的
        top1_error_val = ema1.average(self.top1_error)
        top1_error_avg = ema2.average(self.top1_error)
        loss_val = ema1.average(self.train_loss)
        loss_val_avg = ema2.average(self.train_loss)
        with summary_writer.as_default():
            tf.summary.scalar('val_top1_error', top1_error_val, epoch)
            tf.summary.scalar('val_top1_error_avg', top1_error_avg, epoch)
            tf.summary.scalar('val_loss', loss_val, epoch)
            tf.summary.scalar('val_loss_avg', loss_val_avg, epoch)
        # 需要val_op的时候自然是要返回的
        # TODO: 我康康不返回 operation 会不会更新参数
        return self.train_loss, self.top1_error

    def full_validation(self, validation_df):
        num_batches = len(validation_df) // VALI_BATCH_SIZE
        error_list = []
        loss_list = []
        for i in range(num_batches):
            offset = i * VALI_BATCH_SIZE
            vali_batch_df = validation_df.iloc[offset:offset + VALI_BATCH_SIZE, :]
            validation_image_batch, validation_labels_batch, validation_bbox_batch = load_data_numpy(vali_batch_df)
            temp_validation_loss, temp_validation_top_k_error = self.network_learn_validation(
                [validation_image_batch, validation_labels_batch, validation_bbox_batch, i])
            loss_list.append(temp_validation_loss)
            error_list.append(temp_validation_top_k_error)
        return np.mean(error_list), np.mean(loss_list)


def training():
    # TODO: 这里的输入是最为简单的输入 后面需要修改
    # todo: 预处理文件里，对于图片的预处理略显草率，后面等Yolo调试好了，要将里面的box都识别出来，现在先把feature输出出来再说
    train_df = prepare_df(const.train_path,
                          usecols=['img_path', 'label', 'x_min', 'y_min', 'x_max', 'y_max'], if_shuffle=True)
    vali_df = prepare_df(const.vali_path,
                         usecols=['img_path', 'label', 'x_min', 'y_min', 'x_max', 'y_max'], if_shuffle=True)

    num_train = len(train_df)

    loss_list = []
    step_list = []
    train_error_list = []
    validation_error_list = []
    min_error = 0.5

    # 初始化模型
    train_model = ResNetImprove(False, lr=0.001)
    # TODO: 不知道这行加不加
    # train_model.build(input_shape=)
    validation_model = ResNetImprove(True, lr=0.001)

    # 开始手动训练
    for epoch in range(STEP_TO_TRAIN):
        offset = np.random.choice(num_train - TRAIN_BATCH_SIZE, 1)[0]

        train_batch_df = train_df.iloc[offset:offset + TRAIN_BATCH_SIZE, :]
        batch_data, batch_label, batch_bbox = load_data_numpy(train_batch_df)

        validation_image_batch, validation_labels_batch, validation_bbox_batch = generate_validation_batch(vali_df)

        start_time = time.time()
        if epoch == 0:
            if FULL_VALIDATION is True:
                vali_temp_top1_error, vali_temp_loss = validation_model.full_validation(vali_df)
                with summary_writer.as_default():
                    tf.summary.scalar("full_validation_error", vali_temp_top1_error, step=epoch)
                    tf.summary.scalar("full_validation_loss", vali_temp_loss, step=epoch)
                pass
            else:
                vali_temp_top1_error, vali_temp_loss = validation_model.network_learn_validation(
                    [validation_image_batch, validation_labels_batch, validation_bbox_batch, epoch])
            print('Validation top1 error = ' + str(vali_temp_top1_error.numpy()))
            print('Validation loss = ' + str(vali_temp_loss.numpy()))
            print('----------------------------')

        temp_loss, temp_top_k_error = train_model.network_learn([batch_data, batch_label, batch_bbox, epoch])
        duration = time.time() - start_time

        # 隔几个epoch  就打印一次报告
        if epoch % REPORT_FREQ == 0:
            num_examples_per_step = TRAIN_BATCH_SIZE
            examples_per_sec = num_examples_per_step / duration
            sec_per_batch = float(duration)
            format_str = '%s: step %d, loss = %.4f (%.1f examples/sec; %.3f ' 'sec/batch)'
            print(format_str % (datetime.now(), epoch, temp_loss.numpy(), examples_per_sec, sec_per_batch))
            print('Train top1 error = ', temp_top_k_error.numpy())
            if FULL_VALIDATION is True:
                vali_temp_top1_error, vali_temp_loss = validation_model.full_validation(vali_df)
                with summary_writer.as_default():
                    tf.summary.scalar("full_validation_error", vali_temp_top1_error, step=epoch)
                    tf.summary.scalar("full_validation_loss", vali_temp_loss, step=epoch)
            else:
                vali_temp_top1_error, vali_temp_loss = validation_model.network_learn_validation(
                    [validation_image_batch, validation_labels_batch, validation_bbox_batch, epoch])
            print('Validation top1 error = %.4f' % vali_temp_top1_error.numpy())
            print('Validation loss = ', vali_temp_loss.numpy())
            print('----------------------------')

            if vali_temp_top1_error < min_error:
                min_error = vali_temp_top1_error
                # 保存权重  我这种子类模型是不能用 model.save()的，只能保存权重
                train_model.save_weights(train_min_checkpoint_path)
                validation_model.save_weights(vali_min_checkpoint_path)
                print('Current lowest error = ', min_error)

            step_list.append(epoch)
            train_error_list.append(temp_top_k_error)
            validation_error_list.append(vali_temp_top1_error)

            if epoch == DECAY_STEP0 or epoch == DECAY_STEP1:  # 学习率会在这两个地方衰减
                # 事实上我并没有用到学习率，我认为Adam足够的好，并不需要我来调整学习率
                # 原本别人的代码是用的单纯的 动量优化器，表现肯定不佳，于是他进行手动调整参数，我用的高级的函数，我就不用调整了
                const.learning_rate = const.learning_rate * 0.1
            if epoch % 10000 == 0 or (epoch + 1) == STEP_TO_TRAIN:
                # 保存模型
                train_model.save_weights(train_checkpoint_path)
                validation_model.save_weights(vali_checkpoint_path)
                # TODO:saver.save(sess, checkpoint_path, global_step=epoch)
                # 将error保存下来
                error_df = pd.DataFrame(data={'step': step_list, 'train_error': train_error_list,
                                              'validation_error': validation_error_list})
                error_df.to_csv(ERROR_SAVE_DIR + "log_error.csv", index=False)
            if (epoch + 1) == STEP_TO_TRAIN:
                # 保存模型
                train_model.save_weights(train_checkpoint_path)
                validation_model.save_weights(vali_checkpoint_path)
                # 将error保存下来
                error_df = pd.DataFrame(data={'step': step_list, 'train_error': train_error_list,
                                              'validation_error': validation_error_list})
                error_df.to_csv(ERROR_SAVE_DIR + "log_error.csv", index=False)
    print('Training finished!!')


def test():
    # 构建测试模型
    test_model = ResNetImprove(False, lr=0.001)

    # 加载之前保存的权重
    test_model.load_weights(train_min_checkpoint_path)
    print('Model restored!')

    # TODO: 这里的输入需要修改
    test_df = prepare_df(const.test_path, usecols=['img_path', 'label', 'x_min', 'y_min', 'x_max', 'y_max'],
                         if_shuffle=False)
    test_df = test_df.iloc[-25:, :]

    prediction_np = np.array([]).reshape((-1, 6))
    fc_np = np.array([]).reshape((-1, 64))
    # Hack here: 25 as batch size. 50000 images in total
    for epoch in range(len(test_df) // TEST_BATCH_SIZE):
        df_batch = test_df.iloc[epoch * 25: (epoch + 1) * 25, :]
        test_batch, test_label, test_bbox = load_data_numpy(df_batch)
        test_loss, test_error_value = test_model.network_learn([test_batch, test_label, test_bbox, epoch])
        if epoch % 100 == 0:
            print('Testing %i batches...' % epoch)
            if epoch != 0:
                print('Test_error = ', test_error_value)
        # 注意这里没有进行训练的，只是用模型 predict了一下，相当于输出了一下
        logits, bbox, fc_batch_value = test_model([test_batch, test_label, test_bbox, None, None])
        prediction_batch_value = tf.nn.softmax(logits)
        prediction_np = np.concatenate((prediction_np, prediction_batch_value), axis=0)
        fc_np = np.concatenate((fc_np, fc_batch_value))
    print('Predictin array has shape ', fc_np.shape)
    # TODO:这保存下来的应该就是特征了吧
    np.save(const.fc_path, fc_np[-5:, :])


# physical_devices =

# training()
test()
