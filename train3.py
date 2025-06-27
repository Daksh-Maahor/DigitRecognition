from data import get_mnist
import numpy as np
import matplotlib.pyplot as plt
import pickle
from PIL import Image
from view_neural_data3 import get_neural_data

print("Running")
images, labels = get_mnist()
print("Data Acquired")

'''w_i_h1, b_i_h1, w_h1_h2, b_h1_h2, w_h2_h3, b_h2_h3, w_h3_o, b_h3_o = get_neural_data()'''
w_i_h1 = np.random.uniform(-0.5, 0.5, (200, 784))
b_i_h1 = np.zeros((200, 1))
w_h1_h2 = np.random.uniform(-0.5, 0.5, (200, 200))
b_h1_h2 = np.zeros((200, 1))
w_h2_h3 = np.random.uniform(-0.5, 0.5, (200, 200))
b_h2_h3 = np.zeros((200, 1))
w_h3_o = np.random.uniform(-0.5, 0.5, (10, 200))
b_h3_o = np.zeros((10, 1))

learn_rate = 0.01
correct = 0
epochs = 100
samples_per_batch = 5000

# learn
for epoch in range(epochs):
    print(f"Epoch {epoch}")
    indices = np.floor(np.random.random(samples_per_batch) * 60000)
    for idx in indices:
        idx = int(idx)
        img = images[idx]
        l = labels[idx]
        img.shape += (1,)
        temp = np.reshape(img, (28, 28))
        temp_image = []
        for row in temp:
            new_row = []
            for i in row:
                new_row.append((i, i, i))
            temp_image.append(new_row)

        temp_image = np.array(temp_image)
        temp_image = Image.fromarray(temp_image)

        # rotate the image randomly : -60 to 60
        angle = np.random.random() * 120 - 60
        rotated = temp_image.rotate(angle)

        img = [i[0] for i in list(rotated.getdata())]
        img = np.array(img)
        img.shape += (1, )
        img = img.astype('float32') / 255
        l.shape += (1,)

        # forward propagation : input -> hidden1
        h1_pre = b_i_h1 + w_i_h1 @ img
        h1 = 1 / (1 + np.exp(-h1_pre))

        # forward propagation : hidden1 -> hidden2
        h2_pre = b_h1_h2 + w_h1_h2 @ h1
        h2 = 1 / (1 + np.exp(-h2_pre))

        # forward propagation : hidden2 -> hidden3
        h3_pre = b_h2_h3 + w_h2_h3 @ h2
        h3 = 1 / (1 + np.exp(-h3_pre))

        # forward propagation : hidden3 -> output
        o_pre = b_h3_o + w_h3_o @ h3
        o = 1 / (1 + np.exp(-o_pre))

        # error calculation
        e = 1 / len(o) * np.sum((o-l)**2, axis=0)
        correct += int(np.argmax(o) == np.argmax(l))

        # back propagation : output -> hidden3
        delta_o = o - l
        w_h3_o += -learn_rate * delta_o @ np.transpose(h3)
        b_h3_o += -learn_rate * delta_o

        # back propagation : hidden3 -> hidden2
        delta_h3 = np.transpose(w_h3_o) @ delta_o * (h3 * (1-h3))
        w_h2_h3 += -learn_rate * delta_h3 @ np.transpose(h2)
        b_h2_h3 += -learn_rate * delta_h3

        # back propagation : hidden2 -> hidden1
        delta_h2 = np.transpose(w_h2_h3) @ delta_h3 * (h2 * (1-h2))
        w_h1_h2 += -learn_rate * delta_h2 @ np.transpose(h1)
        b_h1_h2 += -learn_rate * delta_h2

        # back propagation : hidden1 -> input
        delta_h1 = np.transpose(w_h1_h2) @ delta_h2 * (h1 * (1-h1))
        w_i_h1 += -learn_rate * delta_h1 @ np.transpose(img)
        b_i_h1 += -learn_rate * delta_h1
    
    print(f"Accuracy : {np.round(correct/samples_per_batch * 100, 2)}%")
    correct = 0

with open("data/neural_data3.dat", "wb") as f:
    pickle.dump(w_i_h1, f)
    pickle.dump(b_i_h1, f)
    pickle.dump(w_h1_h2, f)
    pickle.dump(b_h1_h2, f)
    pickle.dump(w_h2_h3, f)
    pickle.dump(b_h2_h3, f)
    pickle.dump(w_h3_o, f)
    pickle.dump(b_h3_o, f)
    input()
    