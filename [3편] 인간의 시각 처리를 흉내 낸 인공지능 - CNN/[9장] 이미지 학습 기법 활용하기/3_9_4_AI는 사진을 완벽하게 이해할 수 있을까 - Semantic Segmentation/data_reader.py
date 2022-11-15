"""
Author : Byunghyun Ban
Date : 2020.07.17.
"""
import random
import os
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


# 데이터를 떠먹여 줄 클래스를 제작합니다.
class DataReader():
    def __init__(self):
        self.label = ["Background", "Pet"]

        self.train_X = []
        self.train_Y = []
        self.test_X = []
        self.test_Y = []

        self.read_data()

    def read_data(self):
        print("Reading Data...")
        images = os.listdir("data/images")
        annotations = os.listdir("data/annotations")

        images.sort()
        annotations.sort()

        data = []

        for i in range(len(images)):
            img = Image.open("data/images/" + images[i])
            ant = Image.open("data/annotations/" + annotations[i])

            if img.mode != "RGB":
                img = img.convert("RGB")

            X = np.asarray(img) / 255.0

            Y_temp = np.asarray(ant)[:, :, 0]
            Y = np.zeros_like(Y_temp)
            Y[Y_temp > 127.5] = 1.0

            data.append((X, Y))
            img.close()
            ant.close()

        random.shuffle(data)

        for i, el in enumerate(data):
            if i < 0.8*len(data):
                self.train_X.append(el[0])
                self.train_Y.append(el[1])
            else:
                self.test_X.append(el[0])
                self.test_Y.append(el[1])

        self.train_X = np.asarray(self.train_X)
        self.train_Y = np.asarray(self.train_Y)
        self.test_X = np.asarray(self.test_X)
        self.test_Y = np.asarray(self.test_Y)

        # 데이터 읽기가 완료되었습니다.
        # 읽어온 데이터의 정보를 출력합니다.
        print("\n\nData Read Done!")
        print("Training X Size : " + str(self.train_X.shape))
        print("Training Y Size : " + str(self.train_Y.shape))
        print("Test X Size : " + str(self.test_X.shape))
        print("Test Y Size : " + str(self.test_Y.shape) + '\n\n')

    def show_processed_image(self):
        plt.figure(figsize=(15, 15))
        image = self.train_X[0]
        annotation = self.train_Y[0]
        plt.subplot(1, 2, 1)
        plt.title("image")
        plt.imshow(image)
        plt.subplot(1, 2, 2)
        plt.title("annotation")
        plt.imshow(annotation)


def save_segmentation_results(image, mask_y, graph):
    prediction = graph.predict(image)[:, :, :, 0]
    prediction[prediction < 0] = 0
    pred_mask = (np.copy(image)*255).astype(np.uint8)[:, :, :, 0]
    pred_mask[prediction > 0.5] = 255

    mask = (np.copy(image)*255).astype(np.uint8)[:, :, :, 0]
    mask[mask_y > 0.5] = 255

    image = (image * 255).astype(np.uint8)

    template = np.copy(image)[:, :, :, 0]

    mask = np.stack((template, mask, template), axis=3)
    pred_mask = np.stack((template, template, pred_mask), axis=3)

    if "results" not in os.listdir():
        os.mkdir("results")

    for i in range(len(image)):
        new_canvas = np.concatenate((image[i], mask[i], pred_mask[i]), axis=1)
        img = Image.fromarray(new_canvas)
        img.save("results/" + str(time.time()) + ".jpg")
        img.close()

    print("RESULT SAVED")


def draw_graph(history):
    train_history = history.history["loss"]
    validation_history = history.history["val_loss"]
    fig = plt.figure(figsize=(8, 8))
    plt.title("Loss History")
    plt.xlabel("EPOCH")
    plt.ylabel("LOSS Function")
    plt.plot(train_history, "red")
    plt.plot(validation_history, 'blue')
    fig.savefig("train_history.png")

    train_history = history.history["accuracy"]
    validation_history = history.history["val_accuracy"]
    fig = plt.figure(figsize=(8, 8))
    plt.title("Accuracy History")
    plt.xlabel("EPOCH")
    plt.ylabel("Accuracy")
    plt.plot(train_history, "red")
    plt.plot(validation_history, 'blue')
    fig.savefig("accuracy_history.png")
