"""
Author : Byunghyun Ban
Date : 2020.07.24.
"""
import data_reader
from tensorflow import keras

# 몇 에포크 만큼 학습을 시킬 것인지 결정합니다.
EPOCHS = 100  # 예제 기본값은 100입니다.

# 데이터를 읽어옵니다.
dr = data_reader.DataReader(14)

# 인공신경망을 제작합니다.
model = keras.Sequential([
    keras.layers.Bidirectional(keras.layers.LSTM(128, return_sequences=True)),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(5)
])

# 인공신경망을 컴파일합니다.
model.compile(optimizer="adam", loss="mae", metrics=["mse"])

# 인공신경망을 학습시킵니다.
print("\n\n************ TRAINING START ************ ")
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)
history = model.fit(dr.train_X, dr.train_Y, epochs=EPOCHS,
                    validation_data=(dr.test_X, dr.test_Y),
                    callbacks=[early_stop])

# 학습 결과를 그래프로 출력합니다.
data_reader.draw_graph(model(dr.test_X), dr.test_Y, history)
