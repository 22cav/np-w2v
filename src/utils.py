import numpy as np


def sigmoid(x):
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / np.sum(e, axis=0)


def cross_entropy_loss(y_pred, y_true):
    return -np.sum(y_true * np.log(y_pred + 1e-15))


def one_hot_encode(index, vocab_size):
    one_hot = np.zeros(vocab_size)
    one_hot[index] = 1
    return one_hot


def one_hot_decode(one_hot_vector):
    return np.argmax(one_hot_vector)