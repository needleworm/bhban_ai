"""
Author : Byunghyun Ban
Date : 2020.07.24.
"""
import data_reader
import tensorflow_hub as hub


# 데이터를 불러옵니다.
dr = data_reader.DataReader("content.jpg", "style.jpg")

# Hub로부터 style transfer 모듈을 불러옵니다.
hub_module = hub.load(
    'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/1'
)

# 모듈에 이미지를 삽입해 Style Transfer를 실시합니다.
stylized_image = hub_module(dr.content, dr.style)[0]

# 결과를 출력합니다.
result = data_reader.tensor_to_image(stylized_image)

# 결과를 저장합니다.
result.save("result.jpg")
