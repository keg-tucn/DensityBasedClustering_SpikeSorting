import numpy as np
import sys
from sklearn import metrics

from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans

sys.setrecursionlimit(100000)

import matplotlib.pyplot as plt

import Meeting11.alg as alg
import Meeting11.benchmarking as f

X = np.genfromtxt("s2_labeled.csv", delimiter=",")
X, y = X[:, [0,1]], X[:, 2]


kmeans = KMeans(n_clusters=15).fit(X)
labels = kmeans.labels_
label_color = [f.LABEL_COLOR_MAP[l] for l in labels]
fig = plt.figure()
plt.scatter(X[:, 0], X[:, 1], marker='o', c=label_color, s=25, edgecolor='k')
plt.show()

print('### KMEANS CORRECTNESS')
#f.benchmark(labels, y)
print("ARI:" + str(metrics.adjusted_rand_score(labels, y)))
print("AMI:" + str(metrics.adjusted_mutual_info_score(labels, y)))


NN = NearestNeighbors(n_neighbors=int(np.log(len(X)))).fit(X)
distances, indices = NN.kneighbors(X)

fig = plt.figure()
plt.plot(np.sort(distances[:, distances.shape[1]-1]), color='red')

###### S1 DBSCAN ######
eps = 45000
min_samples=np.log(len(X))*10
db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
labels = db.labels_
label_color = [f.LABEL_COLOR_MAP[l] for l in labels]
unique, counts = np.unique(labels, return_counts=True)
#print('#FINAL:'+ str(dict(zip(unique, counts))))
fig = plt.figure()
plt.scatter(X[:, 0], X[:, 1], marker='o', c=label_color, s=25, edgecolor='k')
plt.show()

print('### DBSCAN CORRECTNESS')
#f.benchmark(labels, y)

print("ARI:" + str(metrics.adjusted_rand_score(labels, y)))
print("AMI:" + str(metrics.adjusted_mutual_info_score(labels, y)))

###### S1 MYALG #####
n=25
labels = alg.nDimAlg(X, n)
print('### MYALG CORRECTNESS')
#f.benchmark(labels, y)
print("ARI:" + str(metrics.adjusted_rand_score(labels, y)))
print("AMI:" + str(metrics.adjusted_mutual_info_score(labels, y)))

