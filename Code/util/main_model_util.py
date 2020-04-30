# coding=utf-8
import cv2
import numpy as np
import pandas as pd
import json
from .const import const

shuffle = True
localization = const.is_localization
imageNet_mean_pixel = [103.939, 116.799, 123.68]
global_std = 68.76

IMG_ROWS = 64
IMG_COLS = 64


def get_image(path, x1, y1, x2, y2):
    """
    :param path: image path
    :param x1: the upper left and lower right coordinates to localize the apparels
    :param y1:
    :param x2:
    :param y2:
    :return: a numpy array with dimensions [img_row, img_col, img_depth]
    """
    img = cv2.imread(path[0])
    if localization is True:
        if img is None or img.shape[0] == 0 or img.shape[1] == 0:
            img = np.zeros((1, IMG_ROWS, IMG_COLS, 0))
        img = cv2.resize(img, (IMG_ROWS, IMG_COLS))
        assert img.shape == (IMG_ROWS, IMG_COLS, 3)
    else:
        img = cv2.resize(img, (IMG_ROWS, IMG_COLS))

    img = img.reshape(1, IMG_ROWS, IMG_COLS, 3)

    return img


def load_data_numpy(df):
    """
    :param df: a pandas dataframe with the image paths and localization coordinates
    :return: the numpy representation of the images and the corresponding labels
    """

    num_images = len(df)
    image_path_array = df['img_path'].to_frame().values

    label_series = df['label']
    label_array_one_hot = np.zeros(shape=(1, 23)).astype(np.int)  # 创建一个全0的，只有一行的numpy数组
    label_array = []  # 我发现独热编码没法输入到 tf.nn.in_top_k() 中，所以我又要将独热编码转化为 整数
    for i, v in label_series.items():  # 遍历Series 将每一行抽出来变成numpy数组
        row = np.array(json.loads(v)).astype(np.int)
        label_array_one_hot = np.vstack((label_array_one_hot, row))  # 逐行叠加  注意有两层括号
        int_value = np.argwhere(row == 1)[0][1]  # 因为列表中只有一个1，所以获取1所在的索引就行
        label_array.append(int_value)
    label_array = np.array(label_array)
    np.delete(label_array_one_hot, 0, axis=0)  # 删除第一行 得到了将str Series转化而成的二维numpy数组
    x1 = df['x_min'].to_frame().values.reshape(-1, 1)
    y1 = df['y_min'].to_frame().values.reshape(-1, 1)
    x2 = df['x_max'].to_frame().values.reshape(-1, 1)
    y2 = df['y_max'].to_frame().values.reshape(-1, 1)
    bbox_array = np.concatenate((x1, y1, x2, y2), axis=1)

    image_array = np.array([]).reshape((-1, IMG_ROWS, IMG_COLS, 3))
    adjusted_std = 1.0 / np.sqrt(IMG_COLS * IMG_ROWS * 3)

    for i in range(num_images):
        img = get_image(image_path_array[i], x1=x1[i, 0], y1=y1[i, 0], x2=x2[i, 0], y2=y2[i, 0])
        flip_indicator = np.random.randint(low=0, high=2)
        if flip_indicator == 0:
            img[0, ...] = cv2.flip(img[0, ...], 1)

        image_array = np.concatenate((image_array, img))

    image_array = (image_array - imageNet_mean_pixel) / global_std

    # Convert to BGR image for pre-train vgg16
    assert image_array.shape[1:] == (IMG_ROWS, IMG_COLS, 3)
    # image_array = image_array.transpose((0, 3, 1, 2))

    return image_array, label_array, bbox_array


def prepare_df(path, usecols, if_shuffle):
    """
    :param if_shuffle: whether shuffle the data
    :param path: the path of a csv file
    :param usecols: which columns to read
    :return: a pandas dataframe
    """
    df = pd.read_csv(path, usecols=usecols)
    if if_shuffle is True:
        order = np.random.permutation(len(df))
        df = df.iloc[order, :]
    return df
