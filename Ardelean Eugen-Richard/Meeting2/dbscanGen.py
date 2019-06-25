import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

# make 3-class dataset for classification
#centers = [[-5, 0], [0, 1.5], [5, -1]]
#X, y = make_blobs(n_samples=1000, centers=centers, random_state=40)

#centers = [(-4, -6), (-1, 1), (-7, 5)]
#X, y = make_blobs(n_samples=50000, n_features=2, cluster_std=1.0, centers=centers, shuffle=False, random_state=30)

n_samples = 1500
random_state = 170
X, y = make_blobs(n_samples=n_samples, random_state=random_state)
transformation = [[0.5, -0.5], [-0.5, 1]]
X = np.dot(X, transformation)
				  
plt.scatter(X[:, 0], X[:, 1], marker='o', s=25, edgecolor='k')
plt.show()
				  
eps=0.3
min_samples=4
db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

print('eps = ' + str(eps))
print('min_samples = ' + str(min_samples))
print('Estimated number of clusters: %d' % n_clusters_)

unique, counts = np.unique(labels, return_counts=True)
print(dict(zip(unique, counts)))

fig = plt.figure()
plt.scatter(X[:, 0], X[:, 1], marker='o', c=labels, s=25, edgecolor='k')
plt.show()
fig.savefig('dbscan2.jpg', dpi=100)