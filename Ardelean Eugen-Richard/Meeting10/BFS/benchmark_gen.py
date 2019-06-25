import numpy as np
import queue
import sys

from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans

sys.setrecursionlimit(100000)

import matplotlib.pyplot as plt

import Meeting10.BFS.benchmarking as f
import Meeting10.BFS.functions as fs

###### READING S1 ######
X, y = fs.getGenData()


X, y = fs.getGenData()
n = 25

kmeans = KMeans(n_clusters=4).fit(X)
labels = kmeans.labels_


label_color = [f.LABEL_COLOR_MAP[l] for l in labels]
fig = plt.figure()
plt.scatter(X[:, 0], X[:, 1], marker='o', c=label_color, s=25, edgecolor='k')
plt.show()

print('### KMEANS CORRECTNESS')
f.benchmark(labels, y)


NN = NearestNeighbors(n_neighbors=int(np.log(len(X)))).fit(X)
distances, indices = NN.kneighbors(X)

fig = plt.figure()
plt.plot(np.sort(distances[:, distances.shape[1]-1]), color='red')

###### S1 DBSCAN ######
eps = 0.5
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

###### S1 MYALG #####
labels = f.squareBFS(X, n)
print('### MYALG CORRECTNESS')
f.benchmark(labels, y)