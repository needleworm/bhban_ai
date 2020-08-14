"""
Author : Byunghyun Ban
Date : 2020.07.24.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow import keras
import data_reader
import time


# 시작 시간을 기록합니다.
start = time.time()
print("Process Start...\n\n")

# 몇 에포크 만큼 학습을 시킬 것인지 결정합니다.
EPOCHS = 20 # 예제 기본값은 20입니다.

# 데이터를 읽어옵니다.
dr = data_reader.DataReader()

# 인공신경망을 제작합니다.
# 총 3층짜리 신경망입니다.
graph = keras.Sequential([
    keras.layers.Flatten(input_shape=(32, 32)), # 1층
    keras.layers.Dense(128, activation='relu'), # 3층
    keras.layers.Dense(10, activation='softmax') # 4층
])

# 인공신경망을 컴파일합니다.
graph.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print("Artificial Neural Network Compile Done")

# 인공신경망을 학습시킵니다.
print("\n\n************ TRAINING START ************ ")
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
history = graph.fit(dr.train_X, dr.train_Y, epochs=EPOCHS, validation_data=(dr.test_X, dr.test_Y), callbacks=[early_stop])

# 학습 결과를 그래프로 출력합니다.
data_reader.draw_graph(history)

# 코드 종료 시간을 계산하여 총 몇초가 소요되었는지 출력합니다.
duration = time.time() - start
print("\n\nProcess Done!")
print("Total ", duration, " seconds spent.")
