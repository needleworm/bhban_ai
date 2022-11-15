"""
Author : Byunghyun Ban
Date : 2020.07.17.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
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
    def __init__(self, window_size):
        self.headers = []
        self.train_X, self.train_Y, self.test_X, self.test_Y = self.read_data(window_size)

        # 데이터 읽기가 완료되었습니다.
        # 읽어온 데이터의 정보를 출력합니다.
        print("\n\nData Read Done!")
        print("Training X Size : " + str(self.train_X.shape))
        print("Training Y Size : " + str(self.train_Y.shape))
        print("Test X Size : " + str(self.test_X.shape))
        print("Test Y Size : " + str(self.test_Y.shape) + '\n\n')

    def read_data(self, window_size):
        filename = "data/" + os.listdir("data")[0]
        data = np.loadtxt(filename, delimiter=",", skiprows=1, usecols=(1, 2, 3, 4, 5, 6))
        data = data - np.min(data, axis=0) + 0.0001
        data = data / np.max(data, axis=0)
        train_data = data[:int(len(data)*0.95)]
        test_data = data[int(len(data)*0.95):]

        train_X, train_Y = self.windowing(train_data, window_size)
        test_X, test_Y = self.windowing(test_data, window_size)

        return train_X, train_Y[:, :, :-1], test_X, test_Y[:, :, :-1]

    def windowing(self, array, window_size):
        X = []
        Y = []

        for i in range(len(array)-window_size*2):
            X.append(array[i:i+window_size])
            Y.append(array[i+window_size:i + window_size*2])

        return np.asarray(X), np.asarray(Y)


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
