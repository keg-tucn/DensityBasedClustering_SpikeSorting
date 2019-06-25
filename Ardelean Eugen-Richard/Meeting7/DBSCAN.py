from matplotlib import colors as mcolors

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN

from scipy.spatial import Voronoi, voronoi_plot_2d

np.random.seed(0)
avgPoints = 250

colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
colors.pop('w')
colors.pop('k')
print(colors)

C1 = [-5, -2] + .8 * np.random.randn(avgPoints*2, 2)
C4 = [-2, 3] + .3 * np.random.randn(avgPoints//5, 2)
C3 = [1, -2] + .2 * np.random.randn(avgPoints*5, 2)
C5 = [3, -2] + 1.6 * np.random.randn(avgPoints, 2)
C2 = [4, -1] + .1 * np.random.randn(avgPoints//2, 2)
C6 = [5, 6] + 2 * np.random.randn(avgPoints, 2)
X = np.vstack((C1, C2, C3, C4, C5, C6))



fig, ((plt1, plt2, plt3), (plt4,plt5, plt6)) = plt.subplots(2, 3, figsize=(15, 9))

vor = Voronoi(X)
voronoi_plot_2d(vor)

plt1.set_title('Generated data')
plt1.plot(C1[:, 0], C1[:, 1], 'b.', alpha=0.3)
plt1.plot(C2[:, 0], C2[:, 1], 'r.', alpha=0.3)
plt1.plot(C3[:, 0], C3[:, 1], 'g.', alpha=0.3)
plt1.plot(C4[:, 0], C4[:, 1], 'c.', alpha=0.3)
plt1.plot(C5[:, 0], C5[:, 1], 'm.', alpha=0.3)
plt1.plot(C6[:, 0], C6[:, 1], 'y.', alpha=0.3)




NN = NearestNeighbors(n_neighbors=np.log(X.size).astype(int)).fit(X)
distances, indices = NN.kneighbors(X)

plt2.set_title('eps elbow')
plt2.plot(np.sort(distances[:, distances.shape[1]-1]), color='red', label = 'Elbow')
plt2.legend()


# DBSCAN at 0.25
eps = 0.25
min_samples=np.log(len(X))
db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
labels = db.labels_

n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

print(eps)
print('Estimated number of clusters for eps: %d' % n_clusters_)
unique, counts = np.unique(labels, return_counts=True)
print(dict(zip(unique, counts)))

plt3.set_title('DBSCAN at '+str(eps))
plt3.scatter(X[:, 0], X[:, 1], marker='o', c=labels, s=25, edgecolor='k')

# DBSCAN at 0.5
eps = 0.5
min_samples=np.log(len(X))
db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
labels = db.labels_

n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

print(eps)
print('Estimated number of clusters for eps: %d' % n_clusters_)
unique, counts = np.unique(labels, return_counts=True)
print(dict(zip(unique, counts)))

plt4.set_title('DBSCAN at '+str(eps))
plt4.scatter(X[:, 0], X[:, 1], marker='o', c=labels, s=25, edgecolor='k')

# DBSCAN at 0.75
eps = 0.8
min_samples=np.log(len(X))
db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
labels = db.labels_

n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

print(eps)
print('Estimated number of clusters for eps: %d' % n_clusters_)
unique, counts = np.unique(labels, return_counts=True)
print(dict(zip(unique, counts)))

plt5.set_title('DBSCAN at '+str(eps))
plt5.scatter(X[:, 0], X[:, 1], marker='o', c=labels, s=25, edgecolor='k')

# DBSCAN at 1
eps = 1
min_samples=np.log(len(X))
db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
labels = db.labels_

n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

print(eps)
print('Estimated number of clusters for eps: %d' % n_clusters_)
unique, counts = np.unique(labels, return_counts=True)
print(dict(zip(unique, counts)))

plt6.set_title('DBSCAN at '+str(eps))
plt6.scatter(X[:, 0], X[:, 1], marker='o', c=labels, s=25, edgecolor='k')

plt.tight_layout()
plt.show()

