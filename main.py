from data import get_mnist
import numpy as np
import matplotlib.pyplot as plt
import pickle
from view_neural_data import get_neural_data

print("Running")
images, labels = get_mnist()
print("Data Acquired")

w_i_h, b_i_h, w_h_o, b_h_o = get_neural_data()

learn_rate = 0.01
correct = 0
epochs = 100

# learn
for epoch in range(epochs):
    print(f"Epoch {epoch}")
    for img, l in zip(images, labels):
        img.shape += (1,)
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
    
    print(f"Accuracy : {np.round(correct/images.shape[0] * 100, 2)}%")
    correct = 0

with open("data/neural_data.dat", "wb") as f:
    pickle.dump(w_i_h, f)
    pickle.dump(b_i_h, f)
    pickle.dump(w_h_o, f)
    pickle.dump(b_h_o, f)

while True:
    index = int(input("Enter a number (0 - 59999): "))
    img = images[index]
    plt.imshow(img.reshape(28, 28), cmap="Greys")

    img.shape += (1,)
    # Forward propagation input -> hidden
    h_pre = b_i_h + w_i_h @ img.reshape(784, 1)
    h = 1 / (1 + np.exp(-h_pre))
    # Forward propagation hidden -> output
    o_pre = b_h_o + w_h_o @ h
    o = 1 / (1 + np.exp(-o_pre))

    plt.title(f"Subscribe if its a {o.argmax()} :)")
    plt.show()
