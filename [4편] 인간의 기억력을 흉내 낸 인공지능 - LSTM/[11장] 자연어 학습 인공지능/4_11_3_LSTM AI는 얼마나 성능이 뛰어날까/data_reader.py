"""
Author : Byunghyun Ban
Date : 2020.07.17.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow import keras
import random

import time

try:
    from matplotlib import pyplot as plt
except ModuleNotFoundError:
    import pip
    pip.main(['install', 'matplotlib'])
    try:
        from matplotlib import pyplot as plt
    except ModuleNotFoundError:
        time.sleep(2)
        from matplotlib import pyplot as plt

try:
    import numpy as np
except ModuleNotFoundError:
    import pip
    pip.main(['install', 'numpy'])
    try:
        import numpy as np
    except ModuleNotFoundError:
        time.sleep(2)
        import numpy as np


# 데이터를 떠먹여 줄 클래스를 제작합니다.
class DataReader():
    def __init__(self):
        self.train_X, self.train_Y, self.test_X, self.test_Y = self.read_data()

        # 데이터 읽기가 완료되었습니다.
        # 읽어온 데이터의 정보를 출력합니다.
        print("\n\nData Read Done!")
        print("Training X Size : " + str(self.train_X.shape))
        print("Training Y Size : " + str(self.train_Y.shape))
        print("Test X Size : " + str(self.test_X.shape))
        print("Test Y Size : " + str(self.test_Y.shape) + '\n\n')

    def read_data(self):
        filename = "data/" + os.listdir("data")[0]
        file = open(filename)
        data = []

        for line in file:
            splt = line.split("\t")
            if splt[0] == "ham":
                y = 0.0
            elif splt[0] == "spam":
                y = 1.0

            x = splt[1].strip()
            if (x, y) not in data:
                data.append((x, y))

        random.shuffle(data)

        X = []
        Y = []

        for el in data:
            X.append(el[0])
            Y.append(el[1])

        Y = np.asarray(Y)

        tokenizer = keras.preprocessing.text.Tokenizer()
        tokenizer.fit_on_texts(X)
        X = tokenizer.texts_to_sequences(X)
        X = keras.preprocessing.sequence.pad_sequences(X, value=0, padding='post', maxlen=189)

        train_X = X[:int(0.8*len(X))]
        train_Y = Y[:int(0.8*len(Y))]
        test_X = X[int(0.8*len(X)):]
        test_y = Y[int(0.8*len(Y)):]

        return train_X, train_Y, test_X, test_y


def draw_graph(history):
    train_history = history.history["loss"]
    validation_history = history.history["val_loss"]
    fig = plt.figure(figsize=(8, 8))
    plt.title("Loss History")
    plt.xlabel("EPOCH")
    plt.ylabel("LOSS Function")
    plt.plot(train_history, "red")
    plt.plot(validation_history, 'blue')
    fig.savefig("train_history.png")

    train_history = history.history["accuracy"]
    validation_history = history.history["val_accuracy"]
    fig = plt.figure(figsize=(8, 8))
    plt.title("Accuracy History")
    plt.xlabel("EPOCH")
    plt.ylabel("Accuracy")
    plt.plot(train_history, "red")
    plt.plot(validation_history, 'blue')
    fig.savefig("accuracy_history.png")
