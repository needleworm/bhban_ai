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
    def __init__(self, dataset):
        if dataset == "mnist":
            (self.train_X, _), (_, _) = keras.datasets.mnist.load_data()
            self.train_X = self.preprocess(self.train_X)
            self.train_dataset = tf.data.Dataset.from_tensor_slices(self.train_X).shuffle(60000).batch(256)
        elif dataset == "fashion_mnist":
            (self.train_X, _), (_, _) = keras.datasets.fashion_mnist.load_data()
            self.train_X = self.preprocess(self.train_X)
            self.train_dataset = tf.data.Dataset.from_tensor_slices(self.train_X).shuffle(50000).batch(256)
        else:
            print("Only mnist and fashion_mnist supported")
            exit(1)


    def preprocess(self, images):
        images = images.reshape(images.shape[0], 28, 28, 1).astype('float32')
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
