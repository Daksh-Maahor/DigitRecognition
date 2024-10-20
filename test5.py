from data import get_mnist
import numpy as np
import matplotlib.pyplot as plt
import keyboard as kb
import pickle
from view_neural_data5 import get_neural_data

print("Running")
images, labels = get_mnist()
print("Data Acquired")

w_i_h1, b_i_h1, w_h1_h2, b_h1_h2, w_h2_h3, b_h2_h3, w_h3_h4, b_h3_h4, w_h4_h5, b_h4_h5, w_h5_o, b_h5_o = get_neural_data()

while not kb.is_pressed('shift'):
    index = np.random.randint(0, 60000)
    img = images[index]
    plt.imshow(img.reshape(28, 28), cmap="Greys")

    img.shape += (1,)
    # Forward propagation input -> hidden1
    h1_pre = b_i_h1 + w_i_h1 @ img.reshape(784, 1)
    h1 = 1 / (1 + np.exp(-h1_pre))
    # Forward propagation hidden1 -> hidden2
    h2_pre = b_h1_h2 + w_h1_h2 @ h1
    h2 = 1 / (1 + np.exp(-h2_pre))
    # Forward propagation hidden2 -> hidden3
    h3_pre = b_h2_h3 + w_h2_h3 @ h2
    h3 = 1 / (1 + np.exp(-h3_pre))
    # Forward propagation hidden3 -> hidden4
    h4_pre = b_h3_h4 + w_h3_h4 @ h3
    h4 = 1 / (1 + np.exp(-h4_pre))
    # Forward propagation hidden4 -> hidden5
    h5_pre = b_h4_h5 + w_h4_h5 @ h4
    h5 = 1 / (1 + np.exp(-h5_pre))
    # Forward propagation hidden5 -> output
    o_pre = b_h5_o + w_h5_o @ h5
    o = 1 / (1 + np.exp(-o_pre))

    plt.title(f"Subscribe if its a {o.argmax()} :)")
    plt.show()
