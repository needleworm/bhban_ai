"""
Author : Byunghyun Ban
Date : 2020.07.24.
This code uses sample codes from Tensorflow.org,
which has Apache 2.0 License.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow import keras
import data_reader
import time
import unet

# 시작 시간을 기록합니다.
start = time.time()
print("Process Start...\n\n")

# 몇 에포크 만큼 학습을 시킬 것인지 결정합니다.
EPOCHS = 50  # 예제 기본값은 50입니다.

# 데이터를 읽어옵니다.
dr = data_reader.DataReader()

# U-Net을 불러옵니다.
# U-Net의 규모나 구조가 이전 예제에 비해 복잡하여 별도 파일로 제작했습니다.
# U-Net의 제작 방법이 궁금하시다면 "unet.py" 파일을 확인해보시기 바랍니다.
graph = unet.graph(128, 128)

# 인공신경망을 컴파일합니다.
loss = keras.losses.BinaryCrossentropy(from_logits=True)
graph.compile(optimizer="adam", loss=loss, metrics=['accuracy'])

# 인공신경망을 학습시킵니다.
print("\n\n************ TRAINING START ************ ")
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
history = graph.fit(dr.train_X, dr.train_Y, epochs=EPOCHS, validation_data=(dr.test_X, dr.test_Y), callbacks=[early_stop])

# Segmentation 결과를 저장합니다.
data_reader.save_segmentation_results(dr.test_X, dr.test_Y, graph)

# 학습 결과를 그래프로 출력합니다.
data_reader.draw_graph(history)

# 코드 종료 시간을 계산하여 총 몇초가 소요되었는지 출력합니다.
duration = time.time() - start
print("\n\nProcess Done!")
print("Total ", duration, " seconds spent.")
