"""
Author : Byunghyun Ban
Date : 2020.07.17.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
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
        file = open("data/" + os.listdir('data')[0])
        data = []
        file.readline()
        for line in file:
            splt = line.split(",")
            chest = process(splt[2])
            arm = process(splt[3])
            height = process(splt[4])
            waist = process(splt[5])
            sat = process(splt[6])
            head = process(splt[7])
            feet = process(splt[8])
            weight = process(splt[9])

            data.append((chest, arm, height, waist, sat, head, feet, weight))

        random.shuffle(data)
        data = np.asarray(data)

        normalized_data = data / np.max(data, axis=0)

        x, y = normalized_data.shape

        train_X = normalized_data[:int(x * 0.8), :-1]
        train_Y = normalized_data[:int(x * 0.8), -1]
        test_X = normalized_data[int(x * 0.8):, :-1]
        test_Y = normalized_data[int(x * 0.8):, -1]

        file.close()

        return train_X, train_Y, test_X, test_Y


def process(txt):
    if "(" in txt:
        txt = txt.split("(")[0]
    txt = txt.strip()
    return float(txt)


def draw_graph(prediction, label, history):
    X = prediction / np.max(prediction, axis=0)
    Y = label / np.max(label, axis=0)

    minval = min(np.min(X), np.min(Y))
    maxval = max(np.max(X), np.max(Y))

    fig = plt.figure(figsize=(8, 8))
    plt.title("Regression Result")
    plt.xlabel("Ground Truth")
    plt.ylabel("AI Predict")
    plt.scatter(X, Y)
    plt.plot([minval, maxval], [minval, maxval], "red")
    fig.savefig("result.png")

    train_history = history.history["loss"]
    validation_history = history.history["val_loss"]
    fig = plt.figure(figsize=(8, 8))
    plt.title("Loss History")
    plt.xlabel("EPOCH")
    plt.ylabel("LOSS Function")
    plt.plot(train_history, "red")
    plt.plot(validation_history, 'blue')
    fig.savefig("train_history.png")
