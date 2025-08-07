import numpy as np


def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return (x > 0).astype(float)


def softmax(x):
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))  # prevent overflow
    return exps / np.sum(exps, axis=1, keepdims=True)


def cross_entropy(predictions, labels):
    # labels should be integers (0–9)
    n = predictions.shape[0]
    p = predictions[range(n), labels]
    return -np.mean(np.log(p + 1e-9))  # prevent log(0)


def cross_entropy_derivative(predictions, labels):
    n = predictions.shape[0]
    grad = predictions.copy()
    grad[range(n), labels] -= 1
    grad = grad / n
    return grad


def one_hot_encode(y, num_classes=10):
    return np.eye(num_classes)[y]
