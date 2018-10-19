import numpy as np
from commons.distance import manhattan_dist, euclidean_dist
from commons.linkage import single_linkage_dist, complete_linkage_dist, average_linkage_dist, average_group_linkage_dist
from commons.converter import convert_dict_key
from commons.data_structure import UnionFindDisjointSet
from commons.cluster import get_cluster, get_cluster_member


class KMeansClustering:
    def __init__(self, n_clusters, max_iter=1000):
        if n_clusters < 1:
            raise Exception('n_cluster should be 1 or more')

        self.n_clusters = n_clusters

        if max_iter < 1:
          raise Exception('max_iter should be 1 or more')

        self.max_iter = max_iter

    def _init_means(self, X):
        return np.array(X)[np.sort(np.random.choice(len(X), self.n_clusters, replace=False))].tolist()
    
    def _new_means(self, cluster):
        return (np.add.reduce(cluster) / len(cluster)).tolist()
      
    def fit_predict(self, X):
        if len(X) < self.n_clusters:
            raise Exception('Number of data > number of cluster')

        means = self._init_means(X)
        next_means = []
        labels = []
        iteration = 0
        next_means = means
        convergense = False
        
        while not convergense:
            means = next_means
                
            labels = list(get_cluster(X[i], means, self.n_clusters) for i in range(len(X)))
            next_means = list(self._new_means(get_cluster_member(X, labels, i)) for i in range(self.n_clusters))
            
            iteration += 1
            convergense = (means == next_means or iteration >= self.max_iter)
        
        return labels
    