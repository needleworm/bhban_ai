"""
Author : Byunghyun Ban
Date : 2020.07.24.
"""
import data_reader
from tensorflow import keras

# 몇 에포크 만큼 학습을 시킬 것인지 결정합니다.
EPOCHS = 1  # 예제 기본값은 1입니다.

# 데이터를 읽어옵니다.
dr = data_reader.DataReader()

# 인공신경망을 제작합니다.
model = keras.Sequential([
    keras.layers.Embedding(8983, 128),
    keras.layers.Bidirectional(keras.layers.LSTM(32)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dropout(0.1),
    keras.layers.Dense(1, activation='sigmoid')
])

# 인공신경망을 컴파일합니다.
model.compile(optimizer="adam", metrics=['accuracy'],
              loss="binary_crossentropy")

# 인공신경망을 학습시킵니다.
print("\n\n************ TRAINING START ************ ")
history = model.fit(dr.train_X, dr.train_Y, epochs=EPOCHS,
                    validation_data=(dr.test_X, dr.test_Y))

# 학습 결과를 그래프로 출력합니다.
data_reader.draw_graph(history)
