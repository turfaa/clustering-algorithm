#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np


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
        label = np.ones(size) * DBSCAN.CALL_POINT
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


# In[ ]:





# In[ ]:




