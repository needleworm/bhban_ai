"""
Author : Byunghyun Ban
Date : 2020.07.17.
This code uses DCGAN sample codes from Tensorflow.org
which has Apache 2.0 License.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow import keras
import tensorflow as tf
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
    import numpy as np
except ModuleNotFoundError:
    import pip
    pip.main(['install', 'numpy'])
    try:
        import numpy as np
    except ModuleNotFoundError:
        time.sleep(2)
        import numpy as np

try:
    from PIL import Image

except ModuleNotFoundError:
    import pip
    pip.main(['install', 'pillow'])
    try:
        from PIL import Image
    except ModuleNotFoundError:
        time.sleep(2)
        from PIL import Image



# 데이터를 떠먹여 줄 클래스를 제작합니다.
class DataReader():
    def __init__(self):
        (self.train_X, _), (_, _) = keras.datasets.fashion_mnist.load_data()
        self.train_X = self.preprocess(self.train_X)
        self.train_dataset = tf.data.Dataset.from_tensor_slices(self.train_X).shuffle(50000).batch(256)
        
    def preprocess(self, images):
        images = images.reshape(images.shape[0], 28, 28, 1).astype('float32')
        images = images / 127.5 - 1
        return images

    def show_processed_images(self):
        plt.figure(figsize=(10, 10))
        for i in range(25):
            plt.subplot(5, 5, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(self.train_X[i])
        plt.show()
