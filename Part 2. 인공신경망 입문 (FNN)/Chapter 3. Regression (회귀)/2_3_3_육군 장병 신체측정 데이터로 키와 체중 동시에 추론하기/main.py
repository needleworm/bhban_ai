"""
Author : Byunghyun Ban
Date : 2020.07.24.
"""
import data_reader
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow import keras

# 시작 시간을 기록합니다.
start = time.time()
print("Process Start...\n\n")

# 몇 에포크 만큼 학습을 시킬 것인지 결정합니다.
EPOCHS = 50  # 예제 기본값은 100입니다.

# 데이터를 읽어옵니다.
dr = data_reader.DataReader()

# 인공신경망을 제작합니다.
graph = keras.Sequential([
    keras.layers.Dense(256, activation="relu", input_shape=[6]),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(256, activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(256, activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(256, activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(2, activation='sigmoid')
])

# 인공신경망을 컴파일합니다.
graph.compile(
    optimizer="adam",  # 아담 옵티마이저
    loss="mse",  # MAPE
    metrics=['mae']
)

# 인공신경망을 학습시킵니다.
print("\n\n************ TRAINING START ************ ")
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
graph.fit(dr.train_X, dr.train_Y, epochs=EPOCHS, validation_data=(dr.test_X, dr.test_Y), callbacks=[early_stop])

# 학습 정확도를 평가합니다.
test_loss, test_accuracy = graph.evaluate(dr.test_X, dr.test_Y)

# 학습 결과를 확인합니다.
data_reader.save_statistics(graph, dr.test_X, dr.test_Y)


# 코드 종료 시간을 계산하여 총 몇초가 소요되었는지 출력합니다.
duration = time.time() - start
print("\n\nProcess Done!")
print("Total ", duration, " seconds spent.")
