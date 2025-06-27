from data import get_mnist
import numpy as np
import matplotlib.pyplot as plt
import pickle
from view_neural_data import get_neural_data
from PIL import Image

print("Running")
images, labels = get_mnist()
print("Data Acquired")

w_i_h, b_i_h, w_h_o, b_h_o = get_neural_data()

learn_rate = 0.01
correct = 0
epochs = 1
samples_per_batch = 60000

# learn
for epoch in range(epochs):
    print(f"Epoch {epoch}")
    indices = np.floor(np.random.uniform(0, 59999, samples_per_batch))
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

        # forward propagation : input -> hidden
        h_pre = b_i_h + w_i_h @ img
        h = 1 / (1 + np.exp(-h_pre))

        # forward propagation : hidden -> output
        o_pre = b_h_o + w_h_o @ h
        o = 1 / (1 + np.exp(-o_pre))

        # error calculation
        e = 1 / len(o) * np.sum((o-l)**2, axis=0)
        correct += int(np.argmax(o) == np.argmax(l))

        # back propagation : output -> hidden
        delta_o = o - l
        w_h_o += -learn_rate * delta_o @ np.transpose(h)
        b_h_o += -learn_rate * delta_o

        # back propagation : hidden -> input
        delta_h = np.transpose(w_h_o) @ delta_o * (h * (1-h))
        w_i_h += -learn_rate * delta_h @ np.transpose(img)
        b_i_h += -learn_rate * delta_h
    
    print(f"Accuracy : {np.round(correct/samples_per_batch * 100, 2)}%")
    correct = 0

"""with open("data/neural_data.dat", "wb") as f:
    pickle.dump(w_i_h, f)
    pickle.dump(b_i_h, f)
    pickle.dump(w_h_o, f)
    pickle.dump(b_h_o, f)
    """