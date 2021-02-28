import numpy as np


def calculate_cosineDist_gradient(W, X):
    norm_X = np.linalg.norm(X)
    norm_W = np.linalg.norm(W)

    a = np.dot(W, X) / (norm_X * norm_W)
    b = W / norm_W

    c = X / (norm_W * norm_X)
    return a * b - c