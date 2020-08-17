"""
Author : Byunghyun Ban
Date : 2020.07.17.
This code uses DCGAN sample codes from Tensorflow.org
which has Apache 2.0 License.
"""
# pip install -q imageio
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from matplotlib import pyplot as plt
from tensorflow import keras
import tensorflow as tf
from PIL import Image
import numpy as np


# 데이터를 떠먹여 줄 클래스를 제작합니다.
class DataReader():
    def __init__(self):
        (self.origin_train_X, _), (_, _) = keras.datasets.cifar10.load_data()
        self.train_X = self.preprocess(self.origin_train_X)
        self.train_dataset = tf.data.Dataset.from_tensor_slices(self.train_X).shuffle(50000).batch(256)

    def preprocess(self, images):
        images = images / 127.5 - 1
        return images

    def show_processed_images(self):
        plt.figure(figsize=(10, 10))
        for i in range(25):
            plt.subplot(5, 5, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(self.train_X[i])
        plt.show()
