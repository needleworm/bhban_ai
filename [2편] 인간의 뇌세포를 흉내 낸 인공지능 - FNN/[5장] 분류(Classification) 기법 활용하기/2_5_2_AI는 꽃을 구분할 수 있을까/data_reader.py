"""
Author : Byunghyun Ban
Date : 2020.07.17.
"""
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
        self.label = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]

        self.train_X, self.train_Y, self.test_X, self.test_Y = self.read_data()

        # 데이터 읽기가 완료되었습니다.
        # 읽어온 데이터의 정보를 출력합니다.
        print("\n\nData Read Done!")
        print("Training X Size : " + str(self.train_X.shape))
        print("Training Y Size : " + str(self.train_Y.shape))
        print("Test X Size : " + str(self.test_X.shape))
        print("Test Y Size : " + str(self.test_Y.shape) + '\n\n')

    def read_data(self):
        print("Reading Data...")
        file = open("data/iris.csv")
        data = []
        for line in file:
            splt = line.split(",")
            if len(splt) != 5:
              break
            feature_1 = float(splt[0].strip())
            feature_2 = float(splt[1].strip())
            feature_3 = float(splt[2].strip())
            feature_4 = float(splt[3].strip())
            label = self.label.index(splt[4].strip())
            data.append(((feature_1, feature_2, feature_3, feature_4), label))

        random.shuffle(data)

        X = []
        Y = []

        for el in data:
            X.append(el[0])
            Y.append(el[1])

        X = np.asarray(X)
        Y = np.asarray(Y)

        X = X / np.max(X, axis=0)

        train_X = X[:int(len(X)*0.8)]
        train_Y = Y[:int(len(Y)*0.8)]
        test_X = X[int(len(X)*0.8):]
        test_Y = Y[int(len(Y)*0.8):]

        return train_X, train_Y, test_X, test_Y


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
