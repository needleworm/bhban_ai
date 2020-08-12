"""
Author : Byunghyun Ban
Date : 2020.07.17.
"""
import numpy as np
from tensorflow import keras
from matplotlib import pyplot as plt


# 데이터를 떠먹여 줄 클래스를 제작합니다.
class DataReader():
    def __init__(self):
        self.label_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
        self.cifar = keras.datasets.cifar10
        (self.train_X, self.train_Y), (self.test_X, self.test_Y) = self.cifar.load_data()

        self.train_X = self.preprocess(self.train_X)
        self.test_X = self.preprocess(self.test_X)

    def preprocess(self, images):
        return images / 255.0

    def show_processed_images(self):
        plt.figure(figsize=(10, 10))
        for i in range(25):
            plt.subplot(5, 5, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(self.train_X[i], cmap=plt.cm.binary)
            plt.xlabel(self.label_names[int(self.train_Y[i])])
        plt.show()


def draw_graph(history):
    train_history = history.history["loss"]
    validation_history = history.history["val_loss"]
    fig2 = plt.figure(figsize=(8, 8))
    plt.plot(train_history, "red")
    plt.plot(validation_history, 'blue')
    fig2.savefig("train_history.png")
