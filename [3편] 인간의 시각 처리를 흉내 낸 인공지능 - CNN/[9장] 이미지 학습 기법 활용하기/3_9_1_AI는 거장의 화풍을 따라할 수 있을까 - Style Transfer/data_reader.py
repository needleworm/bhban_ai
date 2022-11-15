"""
Author : Byunghyun Ban
Date : 2020.07.17.
"""
"""
This code uses sample codes from "tensorflow.org",
which has Apache 2.0 license.
"""
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
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

try:
    import tensorflow_hub as hub
except ModuleNotFoundError:
    import pip
    pip.main(['install', 'tensorflow_hub'])
    try:
        import tensorflow_hub as hub
    except ModuleNotFoundError:
        time.sleep(2)
        import tensorflow_hub as hub


# 데이터를 떠먹여 줄 클래스를 제작합니다.
class DataReader():
    def __init__(self, content, style):
        self.content = self.load_img(content)
        self.style = self.load_img(style)

    def load_img(self, path_to_img):
      max_dim = 512
      img = tf.io.read_file(path_to_img)
      img = tf.image.decode_image(img, channels=3)
      img = tf.image.convert_image_dtype(img, tf.float32)

      shape = tf.cast(tf.shape(img)[:-1], tf.float32)
      long_dim = max(shape)
      scale = max_dim / long_dim

      new_shape = tf.cast(shape * scale, tf.int32)

      img = tf.image.resize(img, new_shape)
      img = img[tf.newaxis, :]
      return tf.constant(img)


def tensor_to_image(tensor):
  tensor = tensor*255
  tensor = np.array(tensor, dtype=np.uint8)
  if np.ndim(tensor)>3:
    assert tensor.shape[0] == 1
    tensor = tensor[0]
  return Image.fromarray(tensor)
