"""
Author : Byunghyun Ban
Date : 2020.07.24.
This code uses DCGAN sample codes from Tensorflow.org
which has Apache 2.0 License.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow import keras
import glob
import time

try:
    import imageio
except ModuleNotFoundError:
    import pip
    pip.main(['install', 'imageio'])
    try:
        import imageio
    except ModuleNotFoundError:
        time.sleep(2)
        import imageio
try:
    from matplotlib import pyplot as plt
except ModuleNotFoundError:
    import pip
    pip.main(['install', 'matplotlib'])
    try:
        from matplotlib import pyplot as plt
    except ModuleNotFoundError:
        time.sleep(2)
        from matplotlib import pyplot as plt

try:
    from IPython import display
except ModuleNotFoundError:
    import pip
    pip.main(['install', 'ipython'])
    try:
        from IPython import display
    except ModuleNotFoundError:
        time.sleep(2)
        from IPython import display


# 인공신경망을 제작합니다.
def make_generator():
    model = keras.Sequential([
        keras.layers.Dense(7*7*256, use_bias=False, input_shape=(100,)),
        keras.layers.BatchNormalization(),
        keras.layers.LeakyReLU(),

        keras.layers.Reshape((7, 7, 256)),

        keras.layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False),
        keras.layers.BatchNormalization(),
        keras.layers.LeakyReLU(),

        keras.layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False),
        keras.layers.BatchNormalization(),
        keras.layers.LeakyReLU(),

        keras.layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')
    ])

    return model


def make_discriminator():
    model = keras.Sequential([
        keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1]),
        keras.layers.LeakyReLU(),
        keras.layers.Dropout(0.3),

        keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'),
        keras.layers.LeakyReLU(),
        keras.layers.Dropout(0.3),

        keras.layers.Flatten(),
        keras.layers.Dense(1)
    ])

    return model


def loss_D(real_output, fake_output):
    real_loss = keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(real_output), real_output)
    fake_loss = keras.losses.BinaryCrossentropy(from_logits=True)(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def loss_G(fake_output):
    return keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(fake_output), fake_output)


generator_optimizer = keras.optimizers.Adam(1e-4)
discriminator_optimizer = keras.optimizers.Adam(1e-4)

noise_dim = 100
seed = tf.random.normal([36, noise_dim])


# `tf.function`이 어떻게 사용되는지 주목해 주세요.
# 이 데코레이터는 함수를 "컴파일"합니다.
@tf.function
def train_step(generator, discriminator, images):
    noise = tf.random.normal([256, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = loss_G(fake_output)
        disc_loss = loss_D(real_output, fake_output)

    gradient_G = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradient_D = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradient_G, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradient_D, discriminator.trainable_variables))

    return gen_loss, disc_loss


def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input, training=False)
    fig = plt.figure(figsize=(6, 6))

    for i in range(predictions.shape[0]):
        plt.subplot(6, 6, i+1)
        plt.imshow(((predictions[i, :, :, 0]) + 1)/2)
        plt.axis('off')

    plt.savefig('results/image_at_epoch_{:04d}.png'.format(epoch))
    plt.close(fig)


def train(generator, discriminator, dataset, epochs):
    if "results" not in os.listdir():
        os.mkdir("results")

    for epoch in range(epochs):
        start = time.time()
        for image_batch in dataset:
            gen_loss, disc_loss = train_step(generator, discriminator, image_batch)
        duration = time.time() - start
        display.clear_output(wait=True)
        generate_and_save_images(generator, epoch + 1, seed)
        print("Epoch " + str(epoch + 1) + "   Generator Loss : " + str(float(gen_loss))[:7]
                        + "   Discriminator Loss : " + str(float(disc_loss))[:7]
                        + "   Time : " + str(duration)[:5] + " seconds")

    # 마지막 에포크가 끝난 후 생성합니다.
    display.clear_output(wait=True)
    generate_and_save_images(generator, epochs, seed)


def gif_generation():
    anim_file = 'results/dcgan.gif'

    with imageio.get_writer(anim_file, mode='I') as writer:
        filenames = glob.glob('results/image*.png')
        filenames = sorted(filenames)
        last = -1
        for i, filename in enumerate(filenames):
            frame = 2 * (i ** 0.5)
            if round(frame) > round(last):
                last = frame
            else:
                continue
            image = imageio.imread(filename)
            writer.append_data(image)
        image = imageio.imread(filename)
        writer.append_data(image)

    import IPython
    if IPython.version_info > (6, 2, 0, ''):
        display.Image(filename=anim_file)
