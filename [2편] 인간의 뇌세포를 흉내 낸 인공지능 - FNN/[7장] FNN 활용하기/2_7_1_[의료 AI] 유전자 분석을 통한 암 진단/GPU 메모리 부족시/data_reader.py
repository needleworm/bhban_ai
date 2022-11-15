"""
Author : Byunghyun Ban
Date : 2020.07.17.
"""
import random
import os

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
        self.label = []

        self.train_X = []
        self.train_Y = []
        self.test_X = []
        self.test_Y = []

        self.read_data()

    def read_data(self):
        print("Reading Data...")
        files = os.listdir("data")
        data = []
        for i, file in enumerate(files):
            book = np.loadtxt("data/" + file, delimiter=",", dtype=np.float32).transpose()

            self.label.append(file[:-4])
            for el in book:
                data.append((el / np.max(el), i))

        random.shuffle(data)

        for i, el in enumerate(data):
            if i < 0.8*len(data):
                self.train_X.append(el[0])
                self.train_Y.append(el[1])
            else:
                self.test_X.append(el[0])
                self.test_Y.append(el[1])

        self.train_X = np.asarray(self.train_X)
        self.train_Y = np.asarray(self.train_Y)
        self.test_X = np.asarray(self.test_X)
        self.test_Y = np.asarray(self.test_Y)

        # 데이터 읽기가 완료되었습니다.
        # 읽어온 데이터의 정보를 출력합니다.
        print("\n\nData Read Done!")
        print("Training X Size : " + str(self.train_X.shape))
        print("Training Y Size : " + str(self.train_Y.shape))
        print("Test X Size : " + str(self.test_X.shape))
        print("Test Y Size : " + str(self.test_Y.shape) + '\n\n')


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
