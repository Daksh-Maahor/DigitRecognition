from data import get_mnist
import numpy as np
import matplotlib.pyplot as plt
from view_neural_data import get_neural_data
import keyboard as kb

print("Running")
images, labels = get_mnist()
print("Data Acquired")

w_i_h, b_i_h, w_h_o, b_h_o = get_neural_data()

while not kb.is_pressed('shift'):
    index = np.random.randint(0, 60000)
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
