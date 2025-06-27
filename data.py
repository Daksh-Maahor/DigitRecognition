import numpy as np
import pathlib
import csv

def get_mnist():
    with np.load(f"{pathlib.Path(__file__).parent.absolute()}/data/mnist.npz") as f:
        images, labels = f["x_train"], f["y_train"]
    images = images.astype("uint8")
    images = np.reshape(images, (images.shape[0], images.shape[1] * images.shape[2]))
    labels = np.eye(10)[labels]
    return images, labels

if __name__ == "__main__":
    images, labels = get_mnist()
    l = list(zip(images, labels))
    with open('data/data.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for i, (img, label) in enumerate(l):
            print(f"Row : {i}")
            img = list(img)
            label = list(label)

            img.extend(label)
            writer.writerow(img)
