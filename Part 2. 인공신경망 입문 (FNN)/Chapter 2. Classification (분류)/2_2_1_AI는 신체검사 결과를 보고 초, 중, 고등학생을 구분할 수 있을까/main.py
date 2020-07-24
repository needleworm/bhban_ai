"""
Author : Byunghyun Ban
Date : 2020.07.24.
"""
import os
import tensorflow as tf
from tensorflow import keras
import data_reader as dr
import time

# 텐서플로의 경고메시지 출력 기능을 끕니다.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)

# 시작 시간을 기록합니다.
start = time.time()
print("Process Start...\n\n")

# 몇 에포크 만큼 학습을 시킬 것인지 결정합니다.
EPOCHS = 30 # 예제 기본값은 30입니다.

# 데이터를 읽어옵니다.
DataReader = dr.DataReader("data")

# 인공신경망을 제작합니다.
# 총 3층짜리 신경망입니다.
graph = keras.Sequential([
    keras.layers.Dense(3, activation="tanh"), # 1층
    keras.layers.Dense(21, activation='tanh'), # 2층
    keras.layers.Dense(3, activation='softmax') # 3층
])

# 인공신경망을 컴파일합니다.
graph.compile(
    optimizer="adam", # 아담 옵티마이저
    loss="categorical_crossentropy", # 크로스엔트로피
    metrics=["accuracy"]
)

print("Artificial Neural Network Compile Done")

# 인공신경망을 학습시킵니다.
print("\n\n************ TRAINING START ************ ")
graph.fit(DataReader.train_X, DataReader.train_Y, epochs=EPOCHS)

# 학습 정확도를 평가합니다.
test_loss, test_accuracy = graph.evaluate(DataReader.test_X, DataReader.test_Y)

# 테스트 결과를 출력합니다.
print("\n\n************ TEST RESULT ************ ")
print("Loss : ", test_loss)
print("Accuracy : ", str(test_accuracy * 100)[:5] + "%")

# 코드 종료 시간을 계산하여 총 몇초가 소요되었는지 출력합니다.
duration = time.time() - start
print("\n\nProcess Done!")
print("Total ", duration, " seconds spent.")
