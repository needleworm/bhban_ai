"""
Author : Byunghyun Ban
Date : 2020.07.24.
"""
from tensorflow import keras
import data_reader
import unet


# 몇 에포크 만큼 학습을 시킬 것인지 결정합니다.
EPOCHS = 50  # 예제 기본값은 50입니다.

# 데이터를 읽어옵니다.
dr = data_reader.DataReader()

# U-Net을 불러옵니다.
model = unet.graph(128, 128)

# 인공신경망을 컴파일합니다.
loss = keras.losses.BinaryCrossentropy(from_logits=True)
model.compile(optimizer="adam", loss=loss, metrics=['accuracy'])

# 인공신경망을 학습시킵니다.
print("\n\n************ TRAINING START ************ ")
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
history = model.fit(dr.train_X, dr.train_Y, epochs=EPOCHS,
                    validation_data=(dr.test_X, dr.test_Y),
                    callbacks=[early_stop])

# Segmentation 결과를 저장합니다.
data_reader.save_segmentation_results(dr.test_X, dr.test_Y, model)

# 학습 결과를 그래프로 출력합니다.
data_reader.draw_graph(history)
