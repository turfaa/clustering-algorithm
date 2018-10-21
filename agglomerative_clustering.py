import numpy as np
from commons.distance import manhattan_dist, euclidean_dist
from commons.linkage import single_linkage_dist, complete_linkage_dist, average_linkage_dist, average_group_linkage_dist
from commons.converter import convert_dict_key
from commons.data_structure import UnionFindDisjointSet


class AgglomerativeClustering:
    def __init__(self, n_clusters, distance='euclidean', linkage='single'):
        if n_clusters < 1:
            raise Exception('n_cluster should be 1 or more')

        self.n_clusters = n_clusters

        if callable(distance):
            self.distance_function = distance
        elif distance == 'manhattan':
            self.distance_function = manhattan_dist
        elif distance == 'euclidean':
            self.distance_function = euclidean_dist
        else:
            raise NotImplementedError()

        self.linkage = linkage
        if callable(linkage):
            self.linkage_dist_function = linkage
        elif linkage == 'single':
            self.linkage_dist_function = single_linkage_dist
        elif linkage == 'complete':
            self.linkage_dist_function = complete_linkage_dist
        elif linkage == 'average':
            self.linkage_dist_function = average_linkage_dist
        elif linkage == 'average_group':
            self.linkage_dist_function = average_group_linkage_dist
        else:
            raise NotImplementedError()

    def fit_predict(self, X):
        if len(X) < self.n_clusters:
            raise Exception('Number of data > number of cluster')

        if self.linkage == 'single':
            return self._cluster_with_ufd(X)

        self._dist_dict = {}

        clusters = [(idx,) for idx in range(len(X))]

        while len(clusters) > self.n_clusters:
            a = 0
            b = 0
            cluster_dist = np.inf

            for i in range(len(clusters)):
                for j in range(i+1, len(clusters)):
                    current_cluster_dist = self._get_linkage_distance(
                        X, clusters[i], clusters[j])

                    if current_cluster_dist < cluster_dist:
                        a, b, cluster_dist = i, j, current_cluster_dist

            clusters[a] = clusters[a] + clusters[b]
            del clusters[b]

        result = np.zeros(len(X), dtype=np.int8)

        for i in range(len(clusters)):
            for idx in clusters[i]:
                result[idx] = i

        return result

    def _get_linkage_distance(self, X, cluster_a, cluster_b):
        dist = self._dist_dict.get((cluster_a, cluster_b), None)

        if dist is None:
            dist = self._dist_dict.get((cluster_b, cluster_a))

        if dist is None:
            self._dist_dict[(cluster_a, cluster_b)] = self.linkage_dist_function(
                X, cluster_a, cluster_b, self.distance_function)
            dist = self._dist_dict[(cluster_a, cluster_b)]

        return dist

    def _cluster_with_ufd(self, X, reverse=False):
        dist = []

        for i in range(len(X)):
            for j in range(i + 1, len(X)):
                dist.append((self.distance_function(X[i], X[j]), i, j))

        dist.sort(reverse=reverse)

        ufd = UnionFindDisjointSet(n_node=len(X))

        current_n_clusters = len(X)
        i = 0

        while current_n_clusters > self.n_clusters:
            while (ufd.is_same_set(dist[i][1], dist[i][2])):
                i += 1

            ufd.join(dist[i][1], dist[i][2])
            current_n_clusters -= 1

        cluster = []
        for i in range(len(X)):
            cluster.append(ufd.find(i))

        return np.array(convert_dict_key(cluster))
