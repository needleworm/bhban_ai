"""
Author : Byunghyun Ban
Date : 2020.07.17.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow import keras
import random
from matplotlib import pyplot as plt


# 데이터를 떠먹여 줄 클래스를 제작합니다.
class DataReader():
    def __init__(self):
        (train_X, self.train_Y), (test_X, self.test_Y) = keras.datasets.imdb.load_data(num_words=6000)
        self.train_X = keras.preprocessing.sequence.pad_sequences(train_X, value=0, padding='post', maxlen=256)
        self.test_X = keras.preprocessing.sequence.pad_sequences(test_X, value=0, padding='post', maxlen=256)


def draw_graph(history):
    train_history = history.history["loss"]
    validation_history = history.history["val_loss"]
    fig2 = plt.figure(figsize=(8, 8))
    plt.plot(train_history, "red")
    plt.plot(validation_history, 'blue')
    fig2.savefig("train_history.png")
