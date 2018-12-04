import numpy as np


def softmax(x):
    """
    Computes softmax on a vector of shape (n,) or (n, 1)
    """
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def one_hot_encode(n, depth):
    """
    Return one-hot encoded vector of shape (depth, 1)
    """
    a = np.zeros([depth, 1])
    a[n, 0] = 1
    return a
