import numpy as np


def euclidean_dist(a, b):
    dist = 0
    for i in range(len(a)):
        dist += np.power(a[i] - b[i], 2)
    return np.sqrt(dist)


def manhattan_dist(a, b):
    dist = 0
    for i in range(len(a)):
        dist += abs(a[i] - b[i])
    return dist
