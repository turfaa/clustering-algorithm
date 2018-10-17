import numpy as np
from distance import euclidean_dist


def average_linkage_dist(nodes, cluster_a, cluster_b, distance_function=euclidean_dist):
    dist = 0

    for a in cluster_a:
        for b in cluster_b:
            dist += distance_function(nodes[a], nodes[b])

    return dist / (len(cluster_a) * len(cluster_b))


def single_linkage_dist(nodes, cluster_a, cluster_b, distance_function=euclidean_dist):
    dist = np.inf

    for a in cluster_a:
        for b in cluster_b:
            dist = min(dist, distance_function(nodes[a], nodes[b]))

    return dist


def complete_linkage_dist(nodes, cluster_a, cluster_b, distance_function=euclidean_dist):
    dist = -np.inf

    for a in cluster_a:
        for b in cluster_b:
            dist = max(dist, distance_function(nodes[a], nodes[b]))

    return dist


def average_group_linkage_dist(nodes, cluster_a, cluster_b, distance_function=euclidean_dist):
    cluster_a = np.array([nodes[idx] for idx in cluster_a])
    cluster_b = np.array([nodes[idx] for idx in cluster_b])

    center_cluster_a = np.add.reduce(cluster_a) / len(cluster_a)
    center_cluster_b = np.add.reduce(cluster_b) / len(cluster_b)

    return distance_function(center_cluster_a, center_cluster_b)
