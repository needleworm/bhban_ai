"""
Author : Byunghyun Ban
Date : 2020.07.24.
This code uses sample codes from Tensorflow.org,
which has Apache 2.0 License.
"""
import os
import tensorflow as tf
from tensorflow import keras
import data_reader
import time
import unet

# 시작 시간을 기록합니다.
start = time.time()
print("Process Start...\n\n")

# 몇 에포크 만큼 학습을 시킬 것인지 결정합니다.
EPOCHS = 50  # 예제 기본값은 100입니다.

# 데이터를 읽어옵니다.
dr = data_reader.DataReader()

# U-Net을 불러옵니다.
# U-Net의 규모나 구조가 이전 예제에 비해 복잡하여 별도 파일로 제작했습니다.
# U-Net의 제작 방법이 궁금하시다면 "unet.py" 파일을 확인해보시기 바랍니다.
graph = unet.graph(2)

# 인공신경망을 컴파일합니다.
graph.compile(
    optimizer="adam",  # 아담 옵티마이저
    loss="binary_crossentropy",  # 크로스엔트로피
    metrics=['accuracy']
)

print("Artificial Neural Network Compile Done")

# 인공신경망을 학습시킵니다.
print("\n\n************ TRAINING START ************ ")
graph.fit(dr.train_X, dr.train_Y, epochs=EPOCHS)

# 학습 정확도를 평가합니다.
test_loss, test_accuracy = graph.evaluate(dr.test_X, dr.test_Y)

# Segmentation 결과를 저장합니다.
data_reader.save_segmentation_results(dr.test_X, dr.test_Y, graph)

# 결과를 화면에 출력합니다

# 테스트 결과를 출력합니다.
print("\n\n************ TEST RESULT ************ ")
print("Loss : ", test_loss)
print("Accuracy : ", str(test_accuracy * 100)[:5] + "%")

# 코드 종료 시간을 계산하여 총 몇초가 소요되었는지 출력합니다.
duration = time.time() - start
print("\n\nProcess Done!")
print("Total ", duration, " seconds spent.")
