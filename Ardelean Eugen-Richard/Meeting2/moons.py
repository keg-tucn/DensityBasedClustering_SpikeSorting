import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


X, y = make_moons(1000, noise=.05, random_state=0)
plt.scatter(X[:, 0], X[:, 1])
plt.show()

kmeans = KMeans(n_clusters=2)

# Fitting the input data
kmeans = kmeans.fit(X)

# Getting the cluster labels
labels = kmeans.predict(X)

# Centroid values
C = kmeans.cluster_centers_

print("Cluster Centers:")
print(C)

fig = plt.figure()
plt.scatter(X[:, 0], X[:, 1], marker='o', c=labels, s=25, edgecolor='k')
plt.show()
fig.savefig('kmeans.jpg', dpi=100)