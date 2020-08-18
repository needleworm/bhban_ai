"""
Author : Byunghyun Ban
Date : 2020.07.17.
"""
import numpy as np
import os
import random
from matplotlib import pyplot as plt


# 데이터를 떠먹여 줄 클래스를 제작합니다.
class DataReader():
    def __init__(self):
        # 읽어올 파일 이름을 저장할 변수입니다.
        filename = []

        # 해당 폴더 안에 있는 파일들을 목록화 합니다.
        files = os.listdir("data")

        # 이번 예제에서는 ".csv"파일을 사용할 계획이므로, ".csv"파일만 골라냅니다.
        for el in files:
            if ".csv" not in el:
                continue
            filename.append(el)

        # 이번 예제에서는 단 하나의 csv 파일만 사용할 계획이므로
        # 폴더 안의 CSV파일 개수가 1개가 아닐 경우 경고를 출력하며 시스템을 종료합니다.
        if len(filename) != 1:
            print("Please Provide Only 1 CSV file in data/ directory.")
            exit(1)

        # 최종 확인된 파일 이름을 지정합니다.
        self.filename = filename[0]

        # 데이터를 저장할 변수들입니다.
        self.train_X = []
        self.train_Y = []
        self.test_X = []
        self.test_Y = []

        # 본격적으로 파일을 읽어옵니다.
        self.read_data()

    # 데이터를 읽어오기 위한 매서드입니다.
    def read_data(self):
        # 파일을 실행합니다.
        file = open(self.datadir + "/" + self.filename)

        # 헤더를 제거합니다.
        file.readline()

        # 데이터와 레이블을 저장하기 위한 변수입니다.
        data = []

        # 파일을 한 줄씩 읽어옵니다.
        for line in file:
            # 컴마를 기준으로 split()을 실행합니다.
            splt = line.split(",")

            # split 결과물을 정리해 X값과 Y값으로 추립니다.
            x, cls = self.process_data(splt)

            # 추려낸 데이터를 저장합니다.
            data.append((x, cls))

        # 데이터를 섞습니다
        random.shuffle(data)

        # 트레이닝 데이터와 테스트 데이터를 분리할 것입니다.
        for i in range(len(data)):
            if i < len(data) * 0.8:
                self.train_X.append(data[i][0])
                self.train_Y.append(data[i][1])
            else:
                self.test_X.append(data[i][0])
                self.test_Y.append(data[i][1])

        # 최종적으로 변수를 np.array 형태로 정리합니다.
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

    # split() 값을 정리하기 위한 매서드입니다.
    def process_data(self, splt):
        # 읽어온 splt 값에서 학교, 성별, 키, 몸무게만 추출합니다.
        school = splt[9]
        gender = splt[13]
        height = float(splt[15]) / 194.2
        weight = float(splt[16]) / 130.7

        # 완성된 데이터를 저장할 변수입니다.
        data = []

        # 키와 몸무게를 삽입합니다.
        data.append(height)
        data.append(weight)

        # 성별을 삽입합니다. 남자일 경우 1, 여자일 경우 0을 삽입합니다.
        if gender == "남":
            data.append(1)
        else:
            data.append(0)

        # 초등학교, 중학교, 고등학교 정보를 원 핫 벡터로 정리합니다.
        # cls는 레이블 역할을 수행합니다.
        if school.endswith("초등학교"):
            cls = 0
        elif school.endswith("중학교"):
            cls = 1
        elif school.endswith("고등학교"):
            cls = 2

        # 결과물을 리턴합니다.
        return data, cls


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
