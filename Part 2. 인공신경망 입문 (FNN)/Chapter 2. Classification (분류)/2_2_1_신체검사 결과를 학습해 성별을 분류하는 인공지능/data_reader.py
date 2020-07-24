"""
Author : Byunghyun Ban
Date : 2020.07.17.
"""
import numpy as np
import os
import random


class DataReader():
    def __init__(self, datadir):
        filename = []
        self.datadir = datadir
        files = os.listdir(self.datadir)

        for el in files:
            if ".csv" not in el:
                continue
            filename.append(el)

        if len(filename) != 1:
            print("Please Provide Only 1 CSV file in data/ directory.")
            exit(1)

        self.filename = filename[0]

        self.train_X = []
        self.train_Y = []
        self.test_X = []
        self.test_Y = []

        self.read_data()

    def read_data(self):
        file = open(self.datadir + "/" + self.filename)
        file.readline()

        X = []
        Y = []

        for line in file:
            splt = line.split(",")

            data, sex = self.process_data(splt)
            X.append(data)
            Y.append(sex)

        X = np.asarray(X)
        Y = np.asarray(Y)

        X[:, -1] /= np.max(X[:, -1])
        X[:, -2] /= np.max(X[:, -2])

        for i in range(len(X)):
            if random.random() < 0.8:
                self.train_X.append(X[i])
                self.train_Y.append(Y[i])
            else:
                self.test_X.append(X[i])
                self.test_Y.append(Y[i])

        self.train_X = np.asarray(self.train_X)
        self.train_Y = np.asarray(self.train_Y)
        self.test_X = np.asarray(self.test_X)
        self.test_Y = np.asarray(self.test_Y)

        print("Data Read Done!")
        print("Training X Size : " + str(self.train_X.shape))
        print("Training Y Size : " + str(self.train_Y.shape))
        print("Test X Size : " + str(self.test_X.shape))
        print("Test Y Size : " + str(self.test_Y.shape))

    def process_data(self, splt):
        school = splt[9]
        grade = int(splt[11])
        gender = splt[13]
        height = float(splt[15])
        weight = float(splt[16])

        data = []

        # 학교분류 (100), (010), (001)
        if school.endswith("초등학교"):
            data += [1, 0, 0]
        elif school.endswith("중학교"):
            data += [0, 1, 0]
        elif school.endswith("고등학교"):
            data += [0, 0, 1]

        # 학년분류 (100000), (010000), (001000) ....
        grd = [0, 0, 0, 0, 0, 0]
        grd[grade -1] = 1
        data += grd

        # 성별 (10), (01)
        if gender == "남":
            sex = [1, 0]
        else:
            sex = [0, 1]

        # 키와 몸무게는 바로 삽입
        data += [height, weight]

        return data, sex
