import numpy as np
from commons.distance import manhattan_dist, euclidean_dist
from commons.linkage import single_linkage_dist, complete_linkage_dist, average_linkage_dist, average_group_linkage_dist
from commons.converter import convert_dict_key
from commons.data_structure import UnionFindDisjointSet
from commons.cluster import get_cluster, get_cluster_member


class KMedoidClustering:
    def __init__(self, n_clusters, max_iter=1000):
        if n_clusters < 1:
            raise Exception('n_cluster should be 1 or more')

        self.n_clusters = n_clusters

        if max_iter < 1:
            raise Exception('max_iter should be 1 or more')

        self.max_iter = max_iter

    def _new_medoid(self, cluster):
        return np.array(cluster)[np.sort(np.random.choice(len(cluster), 1, replace=False))].tolist()

    def _init_medoid(self, X):
        return np.array(X)[np.sort(np.random.choice(len(X), self.n_clusters, replace=False))].tolist()

    def _k_medoid_cluster_cost(self, cluster, medoid):
        cost = 0

        for i in range(len(cluster)):
            cost += manhattan_dist(cluster[i], medoid)

        return cost

    def fit_predict(self, X):
        medoids = self._init_medoid(X)
        cost = 0
        next_cost = 0
        labels = []
        iteration = 0
        convergense = False

        labels = list(get_cluster(
            X[i], medoids, self.n_clusters, distance=manhattan_dist) for i in range(len(X)))
        cost = next_cost = sum(list(self._k_medoid_cluster_cost(
            get_cluster_member(X, labels, i), medoids[i]) for i in range(self.n_clusters)))

        while not convergense:
            if(next_cost < cost):
                cost = next_cost

            medoids = list(self._new_medoid(get_cluster_member(
                X, labels, i)) for i in range(self.n_clusters))
            labels = list(get_cluster(
                X[i], medoids, self.n_clusters, distance=manhattan_dist) for i in range(len(X)))
            next_cost = sum(list(self._k_medoid_cluster_cost(get_cluster_member(
                X, labels, i), medoids[i]) for i in range(self.n_clusters)))

            iteration += 1
            convergense = (next_cost == cost or iteration >= self.max_iter)

        return labels
