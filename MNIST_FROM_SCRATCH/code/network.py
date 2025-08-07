import numpy as np
from nn_math import relu, relu_derivative, softmax, cross_entropy, cross_entropy_derivative

# NEURAL NETWORK PLAN
#
# Input Layer -> 784 values (28×28 pixels)
#
# Hidden Layer 1 -> 128 neurons + ReLU
#
# Hidden Layer 2 -> 64 neurons + ReLU
#
# Output -> 10 neurons + Softmax (digits 0–9)

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------


class NeuralNetwork:
    def __init__(self, input_size=784, hidden1_size=128, hidden2_size=64, output_size=10, learning_rate=0.01):
        # Xavier Initialization for weights
        self.W1 = np.random.randn(input_size, hidden1_size) * np.sqrt(2. / input_size)
        self.b1 = np.zeros((1, hidden1_size))

        self.W2 = np.random.randn(hidden1_size, hidden2_size) * np.sqrt(2. / hidden1_size)
        self.b2 = np.zeros((1, hidden2_size))

        self.W3 = np.random.randn(hidden2_size, output_size) * np.sqrt(2. / hidden2_size)
        self.b3 = np.zeros((1, output_size))

        self.learning_rate = learning_rate


    def forward(self, x):
        self.z1 = x @ self.W1 + self.b1
        self.a1 = relu(self.z1)
        self.a1 *= (np.random.rand(*self.a1.shape) > 0.1)
        self.a1 /= (1.0 - 0.2)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = relu(self.z2)
        self.a2 *= (np.random.rand(*self.a2.shape) > 0.05)
        self.a2 /= (1.0 - 0.1)
        self.z3 = self.a2 @ self.W3 + self.b3
        self.a3 = softmax(self.z3)
        return self.a3


    def backward(self, x, y, output):
        dz3 = cross_entropy_derivative(output, y)
        dW3 = self.a2.T @ dz3
        db3 = np.sum(dz3, axis=0, keepdims=True)

        da2 = dz3 @ self.W3.T
        dz2 = da2 * relu_derivative(self.z2)
        dW2 = self.a1.T @ dz2
        db2 = np.sum(dz2, axis=0, keepdims=True)

        da1 = dz2 @ self.W2.T
        dz1 = da1 * relu_derivative(self.z1)
        dW1 = x.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)

        # Gradient descent step
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W3 -= self.learning_rate * dW3
        self.b3 -= self.learning_rate * db3


    def train(self, x, y, epochs=5, batch_size=64):
        for epoch in range(epochs):
            permutation = np.random.permutation(x.shape[0])
            x_shuffled = x[permutation]
            y_shuffled = y[permutation]

            for i in range(0, x.shape[0], batch_size):
                x_batch = x_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]

                output = self.forward(x_batch)
                loss = cross_entropy(output, y_batch)
                self.backward(x_batch, y_batch, output)

            print(f"Epoch {epoch+1}/{epochs} - Loss: {loss:.4f}") # pyright: ignore[reportPossiblyUnboundVariable]


    def save(self, path):
        np.savez(path,
            W1=self.W1, b1=self.b1,
            W2=self.W2, b2=self.b2,
            W3=self.W3, b3=self.b3)


    def load(self, path):
        data = np.load(path)
        self.W1 = data['W1']
        self.b1 = data['b1']
        self.W2 = data['W2']
        self.b2 = data['b2']
        self.W3 = data['W3']
        self.b3 = data['b3']


    def predict(self, x):
        output = self.forward(x)
        return np.argmax(output, axis=1)
