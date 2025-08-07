from keras.datasets import mnist

# TODO: Load the data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# TODO: Normalize the pixel values of each image
x_test, x_train = x_test / 255.0, x_train / 255.0

from keras import models, layers, Input

model = models.Sequential([
    Input(shape=(28,28)),
    # TODO: Flatten the input from (28, 28) to 784
    layers.Flatten(),
    # TODO: Add a dense layer with 128 neurons and ReLU activation
    layers.Dense(128, activation="relu"),
    # TODO: Add a dropout layer with 20% dropout
    layers.Dropout(0.2),
    # TODO: Add the output layer with 10 neurons and softmax activation
    layers.Dense(10, activation="softmax")
])

model.compile(
    # This controls how the model updates weights to reduce the loss.
    optimizer='adam',
    # This measures how far off the model’s predictions are from the actual labels.
    loss='sparse_categorical_crossentropy',
    # This tells the model what to report after each epoch.
    metrics=['accuracy']
)

model.fit(x_train, y_train, epochs=5)

test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")

import matplotlib.pyplot as plt
import numpy as np

predictions = model.predict(x_test)

# Show first 5 test images, predicted labels, and actual labels
for i in range(5):
    plt.imshow(x_test[i], cmap='gray')
    plt.title(f"Predicted: {np.argmax(predictions[i])}, Actual: {y_test[i]}")
    plt.axis('off')
    plt.show()
