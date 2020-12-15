"""
Author : Byunghyun Ban
Date : 2020.07.24.
"""
import data_reader
import gan

# 몇 에포크 만큼 학습을 시킬 것인지 결정합니다.
EPOCHS = 200  # 예제 기본값은 200입니다.

# 데이터를 읽어옵니다.
dr = data_reader.DataReader()

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
