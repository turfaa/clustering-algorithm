#!/usr/bin/env python
# coding: utf-8

# In[1]:


def get_purity(cluster, target):
    cnt = [[0 for _ in range(2)] for _ in range(1000000)]
    for i in range(len(cluster)):
        if cluster[i] >= 0:
            cnt[cluster[i]][target[i]] += 1
    sm = 0
    for i in range(1000000):
        sm += max(cnt[i][0], cnt[i][1])
    return sm / len(cluster)

