# LOAD DATA

from data_loader import load_images, load_labels, load_custom_data
from nn_math import one_hot_encode
import numpy as np


x_train = load_images("../data/train-images.idx3-ubyte")
y_train = load_labels("../data/train-labels.idx1-ubyte")


x_train_custom, y_train_custom = load_custom_data("../custom_data/")
print("x_train_custom:", x_train_custom.shape)
print("y_train_custom:", y_train_custom.shape)


x_test = load_images("../data/t10k-images.idx3-ubyte")
y_test = load_labels("../data/t10k-labels.idx1-ubyte")


x_combined = np.concatenate([x_train_custom, x_train], axis=0)
y_combined = np.concatenate([y_train_custom, y_train], axis=0)


indices = np.random.permutation(x_combined.shape[0])
x_combined = x_combined[indices]
y_combined = y_combined[indices]


print("x_combined:", x_combined.shape)
print("y_combined:", y_combined.shape)


# --------------------------------------------------------------------------------------------------------------------------------------------------------------------

# CREATE NEURAL NETWORK

from network import NeuralNetwork

nn = NeuralNetwork()
nn.train(x_combined, y_combined, epochs=50, batch_size=64)

# TEST
predictions = nn.predict(x_test)
accuracy = (predictions == y_test).mean()
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# # SHOW WRONGS
# import matplotlib.pyplot as plt
# import numpy as  np

# wrong = (predictions != y_test)
# wrong_indices = np.where(wrong)[0][:5]

# for i in wrong_indices:
#     plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
#     plt.title(f"Predicted: {predictions[i]}, Actual: {y_test[i]}")
#     plt.axis('off')
#     plt.show()

name = input("Enter model name: ")
nn.save(f"../models/{name}.npz")