"""
Author : Byunghyun Ban
Date : 2020.07.17.
This code uses data visualization sample codes from Tensorflow.org
which has Apache 2.0 License.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import random
import matplotlib.pyplot as plt


# 데이터를 떠먹여 줄 클래스를 제작합니다.
class DataReader():
    def __init__(self):
        self.normalize_factors = []
        self.train_X, self.train_Y, self.test_X, self.test_Y = self.read_data()

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

            data.append((chest, arm, sat, head, feet, waist, height, weight))

        random.shuffle(data)
        data = np.asarray(data)
        self.normalize_factors = np.max(data, axis=0)

        normalized_data = data / self.normalize_factors

        x, y = normalized_data.shape

        train_X = normalized_data[:int(x * 0.8), :-2]
        train_Y = normalized_data[:int(x * 0.8), -2:]
        test_X = normalized_data[int(x * 0.8):, :-2]
        test_Y = normalized_data[int(x * 0.8):, -2:]

        file.close()

        return train_X, train_Y, test_X, test_Y


def process(txt):
    if "(" in txt:
        txt = txt.split("(")[0]
    txt = txt.strip()
    return float(txt)


def save_statistics(graph, test_X, test_Y):
    prediction = graph.predict(test_X)
    fig = plt.figure()
    plt.scatter(prediction[:, 0], test_Y[:, 0])
    plt.xlabel('Test_Y')
    plt.ylabel('Model Prediction')
    plt.axis('equal')
    plt.axis('square')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.plot([-100, 100], [-100, 100])
    plt.savefig("height Result.png")
    plt.close(fig)

    fig = plt.figure()
    plt.scatter(prediction[:, 1], test_Y[:, 1])
    plt.xlabel('Test_Y')
    plt.ylabel('Model Prediction')
    plt.axis('equal')
    plt.axis('square')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.plot([-100, 100], [-100, 100])
    plt.savefig("weight Result.png")
    plt.close(fig)

    fig = plt.figure()
    error = np.average((prediction - test_Y), axis=1)
    plt.hist(error, bins=25)
    plt.xlabel("Prediction - Label")
    plt.ylabel("Count")
    plt.savefig("histogram.png")
    plt.close(fig)
