"""
Author : Byunghyun Ban
Date : 2020.07.24.
This code uses DCGAN sample codes from Tensorflow.org
which has Apache 2.0 License.
"""
import data_reader
import time
import gan

# 시작 시간을 기록합니다.
start = time.time()
print("Process Start...\n\n")

# 몇 에포크 만큼 학습을 시킬 것인지 결정합니다.
EPOCHS = 100  # 예제 기본값은 100입니다.

# 데이터를 읽어옵니다.
dr = data_reader.DataReader("mnist")
#dr = data_reader.DataReader("fashion_mnist")

# GAN을 불러옵니다.
# Generator
generator = gan.make_generator()
# Discriminator
discriminator = gan.make_discriminator()

# 인공신경망을 학습시킵니다.
print("\n\n************ TRAINING START ************ ")
gan.train(generator, discriminator, dr.train_dataset, EPOCHS)

# GIF 애니메이션을 저장합니다.
gan.gif_generation()

# 코드 종료 시간을 계산하여 총 몇초가 소요되었는지 출력합니다.
duration = time.time() - start
print("\n\nProcess Done!")
print("Total ", duration, " seconds spent.")
