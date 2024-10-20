from data import get_mnist
import numpy as np
import matplotlib.pyplot as plt
import keyboard as kb
import pickle
from view_neural_data3 import get_neural_data
import keyboard as kb

print("Running")
images, labels = get_mnist()
print("Data Acquired")

w_i_h1, b_i_h1, w_h1_h2, b_h1_h2, w_h2_h3, b_h2_h3, w_h3_o, b_h3_o = get_neural_data()

wrong_predictions = []

for img, l in zip(images, labels):
    '''plt.imshow(img.reshape(28, 28), cmap="Greys")'''

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
    # Forward propagation hidden3 -> output
    o_pre = b_h3_o + w_h3_o @ h3
    o = 1 / (1 + np.exp(-o_pre))

    if (o.argmax() != l.argmax()):
        wrong_predictions.append((img, o))

    '''plt.title(f"Subscribe if its a {o.argmax()} :)")
    plt.show()'''

i = 0
print(f"No of wrong predictions : {len(wrong_predictions)}")
while i < len(wrong_predictions) and not kb.is_pressed('shift'):
    img, o = wrong_predictions[i]
    plt.imshow(img.reshape(28, 28), cmap="Greys")

    plt.title(f"Predicted {o.argmax()}")
    plt.show()
    i += 1
