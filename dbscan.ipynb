{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "\n",
    "\n",
    "class DBSCAN:\n",
    "    \n",
    "    min_pts = 2\n",
    "    epsilon = 2\n",
    "    \n",
    "    N2_POINT = -5\n",
    "    CALL_POINT = -1\n",
    "    \n",
    "    def __init__(self, epsilon=2, min_pts=2):\n",
    "        self.epsilon = epsilon\n",
    "        self.min_pts = min_pts\n",
    "        \n",
    "    def __euclidean_dist(self, a, b):\n",
    "        dist = 0\n",
    "        for i in range(len(a)):\n",
    "            dist += (a[i] - b[i])**2\n",
    "        return np.sqrt(dist)\n",
    "    \n",
    "    def fit_predict(self, data):\n",
    "        neighbors = []\n",
    "        for i in range(len(data)):\n",
    "            neighbors.append([])\n",
    "            for j in range(len(data)):\n",
    "                if self.__euclidean_dist(data[i], data[j]) <= self.epsilon:\n",
    "                    neighbors[i].append(j)\n",
    "        \n",
    "        core = 0\n",
    "        label = np.ones(size) * DBSCAN.CALL_POINT\n",
    "        for i, point in enumerate(data):\n",
    "            if label[i] != DBSCAN.CALL_POINT:\n",
    "                continue\n",
    "            \n",
    "            if len(neighbors[i]) < self.min_pts:\n",
    "                label[i] = DBSCAN.N2_POINT\n",
    "                continue\n",
    "            \n",
    "            label[i] = core\n",
    "            cd_point = neighbors[i][:]\n",
    "            for neighbors_i in cd_point:\n",
    "                if label[neighbors_i] == DBSCAN.N2_POINT:\n",
    "                    label[neighbors_i] = core\n",
    "                elif label[neighbors_i] == DBSCAN.CALL_POINT:\n",
    "                    label[neighbors_i] = core\n",
    "                    if len(neighbors[neighbors_i]) >= self.min_pts:\n",
    "                        for x in neighbors[neighbors_i]:\n",
    "                            if x not in cd_point:\n",
    "                                cd_point.append(x)\n",
    "\n",
    "            core += 1\n",
    "\n",
    "        return label"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
