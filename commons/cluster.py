import numpy as np
from .distance import euclidean_dist

def get_cluster(x, center, n_cluster, distance=euclidean_dist):
    distances = list(distance(center[i],x) for i in range(n_cluster))
    return distances.index(min(distances))

def get_cluster_member(data, labels, cluster):
    indices = [i for i, x in enumerate(labels) if x == cluster]
    return np.array(data)[indices].tolist()