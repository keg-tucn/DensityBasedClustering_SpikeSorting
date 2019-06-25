import numpy as np
import sys
import queue

from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans

sys.setrecursionlimit(100000)

import matplotlib.pyplot as plt

import Meeting10.BFS.benchmarking as f

X = np.genfromtxt("s2_labeled.csv", delimiter=",")
X, y = X[:, [0,1]], X[:, 2]
n=25

###### READING UNBALANCE ######
print('##### UNBALANCE DATASET')

X = np.genfromtxt("unbalance.csv", delimiter=",")
X, y = X[:, [0,1]], X[:, 2]

kmeans = KMeans(n_clusters=8).fit(X)
labels = kmeans.labels_
label_color = [f.LABEL_COLOR_MAP[l] for l in labels]
fig = plt.figure()
plt.scatter(X[:, 0], X[:, 1], marker='o', c=label_color, s=25, edgecolor='k')
plt.show()

print('### KMEANS CORRECTNESS')
f.benchmark(labels, y)

###### UNBALANCE DBSCAN ######
#eps = 5000
#min_samples=np.log(len(X))*10
eps = 5000
min_samples=np.log(len(X))
db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
labels = db.labels_
label_color = [f.LABEL_COLOR_MAP[l] for l in labels]
unique, counts = np.unique(labels, return_counts=True)
print('#FINAL:'+ str(dict(zip(unique, counts))))
fig = plt.figure()
plt.scatter(X[:, 0], X[:, 1], marker='o', c=label_color, s=25, edgecolor='k')
plt.show()

print('### DBSCAN CORRECTNESS')
f.benchmark(labels, y)

###### UNBALANCE MYALG #####
labels = f.squareBFS(X, n)
f.benchmark(labels, y)