"""
Author : Byunghyun Ban
Date : 2020.07.24.
"""
import tensorflow as tf
from tensorflow import keras
import data_reader
import time


# 시작 시간을 기록합니다.
start = time.time()
print("Process Start...\n\n")

# 몇 에포크 만큼 학습을 시킬 것인지 결정합니다.
EPOCHS = 10 # 예제 기본값은 10입니다.

# 데이터를 읽어옵니다.
dr = data_reader.DataReader()

# 인공신경망을 제작합니다.
graph = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 3)),  # 1층
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),  # 2층
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),  # 3층
    keras.layers.Flatten(),  # 4층
    keras.layers.Dense(128, activation='relu'),  # 5층
    keras.layers.Dense(10, activation='softmax')  # 6층
])

# 인공신경망을 컴파일합니다.
graph.compile(optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
)

print("Artificial Neural Network Compile Done")

# 인공신경망을 학습시킵니다.
print("\n\n************ TRAINING START ************ ")
graph.fit(dr.train_X, dr.train_Y, epochs=EPOCHS)

# 학습 정확도를 평가합니다.
test_loss, test_accuracy = graph.evaluate(dr.test_X, dr.test_Y)

# 테스트 결과를 출력합니다.
print("\n\n************ TEST RESULT ************ ")
print("Loss : ", test_loss)
print("Accuracy : ", str(test_accuracy * 100)[:5] + "%")

# 코드 종료 시간을 계산하여 총 몇초가 소요되었는지 출력합니다.
duration = time.time() - start
print("\n\nProcess Done!")
print("Total ", duration, " seconds spent.")
