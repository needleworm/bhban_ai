"""
Author : Byunghyun Ban
Date : 2020.07.24.
"""
import tensorflow as tf
import time
import tensorflow_hub as hub
import data_reader

# 시작 시간을 기록합니다.
start = time.time()
print("Process Start...\n\n")

dr = data_reader.DataReader("content.jpg", "style.jpg")

# TF Hub를 통해 작업을 수행합니다.
# Hub로부터 style transfer 모듈을 불러옵니다.
hub_module = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/1')

# 모듈에 이미지를 삽입해 Style Transfer를 실시합니다.
stylized_image = hub_module(tf.constant(dr.content), tf.constant(dr.style))[0]

# 결과를 출력합니다.
result = data_reader.tensor_to_image(stylized_image)

# 결과를 저장합니다.
result.save("result.jpg")

# 코드 종료 시간을 계산하여 총 몇초가 소요되었는지 출력합니다.
duration = time.time() - start
print("\n\nProcess Done!")
print("Total ", duration, " seconds spent.")
