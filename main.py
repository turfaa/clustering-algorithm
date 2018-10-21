#!/usr/bin/env python
# coding: utf-8

# ## Load Library

# In[1]:


import numpy as np
from sklearn import datasets


# ## Load Dataset Iris

# In[2]:


iris = datasets.load_iris()
X_iris = iris.data
Y_iris = iris.target


# ## Purity

# In[3]:


def get_purity(cluster, target):
    cnt = [[0 for _ in range(2)] for _ in range(1000000)]
    for i in range(len(cluster)):
        if cluster[i] >= 0:
            cnt[cluster[i]][target[i]] += 1
    sm = 0
    for i in range(1000000):
        sm += max(cnt[i][0], cnt[i][1])
    return sm / len(cluster)


# ## DBSCAN

# In[6]:


class DBSCAN:
    
    min_pts = 2
    epsilon = 2
    
    N2_POINT = -5
    CALL_POINT = -1
    
    def __init__(self, epsilon=2, min_pts=2):
        self.epsilon = epsilon
        self.min_pts = min_pts
        
    def __euclidean_dist(self, a, b):
        dist = 0
        for i in range(len(a)):
            dist += (a[i] - b[i])**2
        return np.sqrt(dist)
    
    def fit_predict(self, data):
        neighbors = []
        for i in range(len(data)):
            neighbors.append([])
            for j in range(len(data)):
                if self.__euclidean_dist(data[i], data[j]) <= self.epsilon:
                    neighbors[i].append(j)
        
        core = 0
        label = np.ones(len(data)) * DBSCAN.CALL_POINT
        for i, point in enumerate(data):
            if label[i] != DBSCAN.CALL_POINT:
                continue
            
            if len(neighbors[i]) < self.min_pts:
                label[i] = DBSCAN.N2_POINT
                continue
            
            label[i] = core
            cd_point = neighbors[i][:]
            for neighbors_i in cd_point:
                if label[neighbors_i] == DBSCAN.N2_POINT:
                    label[neighbors_i] = core
                elif label[neighbors_i] == DBSCAN.CALL_POINT:
                    label[neighbors_i] = core
                    if len(neighbors[neighbors_i]) >= self.min_pts:
                        for x in neighbors[neighbors_i]:
                            if x not in cd_point:
                                cd_point.append(x)

            core += 1

        return label


# In[9]:


dbscan = DBSCAN()
pred = dbscan.fit_predict(X_iris)
purity = get_purity(pred, Y_iris)

print('Purity: ' + str(purity))

