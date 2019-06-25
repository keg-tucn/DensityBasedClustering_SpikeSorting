import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
import numbers
import math
import functions as fs


from sklearn.neighbors import NearestNeighbors


fig, ((plt1, plt2), (plt4,plt5)) = plt.subplots(2, 2, figsize=(15, 9))


# Importing the dataset
data = pd.read_csv('data.csv', skiprows=0)
f1 = data['F1'].values
f2 = data['F2'].values
f3 = data['F3'].values

#print("Shape:")
#print(data.shape)
#print("\n")

c1 = np.array([f1]).T
c2 = np.array([f2]).T
c3 = np.array([f3]).T


X = np.append(np.append(c1, c2, axis=1), c3, axis=1)
		
plt1.set_title('Original data')
plt1.scatter(X[:, 0], X[:, 1], marker='o', s=25, edgecolor='k')


newX = fs.approximationScheme(X)

plt2.set_title('Rarefied data')
plt2.scatter(newX[:, 0], newX[:, 1], marker='o', s=25, edgecolor='k')


NN1 = NearestNeighbors(n_neighbors=np.log(len(newX)).astype(int)).fit(newX)
distances1, indices1 = NN1.kneighbors(newX)

plt4.set_ylim(0, 1)
plt4.set_title('eps elbow')
plt4.plot(np.sort(distances1[:, distances1.shape[1]-1]), color='red')

eps=0.4
min_samples=np.log(len(newX))*10
db = DBSCAN(eps=eps, min_samples=min_samples).fit(newX)
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

print('Estimated number of clusters: %d' % n_clusters_)
unique, counts = np.unique(labels, return_counts=True)
print(dict(zip(unique, counts)))
plt5.set_title('DBSCAN')
plt5.scatter(newX[:, 0], newX[:, 1], marker='o', c=labels, s=25, edgecolor='k')

plt.show()