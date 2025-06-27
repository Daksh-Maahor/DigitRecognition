import numpy as np
import pathlib
from PIL import Image
import random

def get_mnist():
    with np.load(f"{pathlib.Path(__file__).parent.absolute()}/data/mnist.npz") as f:
        images, labels = f["x_train"], f["y_train"]
    images = images.astype("uint8")

    for image in images:
        image = images_int[0]
        temp_image = []
        for row in image:
            new_row = []
            for i in row:
                new_row.append((i, i, i))
            temp_image.append(new_row)

        temp_image = np.array(temp_image)
        temp_image = Image.fromarray(temp_image)

        # rotate the image randomly : -60 to 60
        angle = random.random() * 120 - 60
        rotated = temp_image.rotate(angle)

        pixels = [i[0] for i in list(rotated.getdata())]
        image_new = np.array(pixels).reshape((28, 28))
        image_new.shape = (1, ) + image_new.shape

        images_2 = np.vstack((images_2, image_new))
    
    images = images_2

    print(images.shape)

    images = np.reshape(images, (images.shape[0], images.shape[1] * images.shape[2]))
    labels = np.eye(10)[labels]
    return images, labels

images, labels = get_mnist()
print(images)
