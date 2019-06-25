import numpy as np
import pandas as pd

from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN

from matplotlib import pyplot as plt


fig, ((plt1, plt2, plt3), (plt4,plt5, plt6), (plt7,plt8, plt9)) = plt.subplots(3, 3, figsize=(15, 9))

# Importing the dataset
data = pd.read_csv('data.csv', skiprows=0)
f1 = data['F1'].values
f2 = data['F2'].values
f3 = data['F3'].values

print("Shape:")
print(data.shape)
print("\n")

c1 = np.array([f1]).T
c2 = np.array([f2]).T
c3 = np.array([f3]).T


X = np.append(c1[1:25000], c3[1:25000], axis=1)

NN = NearestNeighbors(n_neighbors=int(np.log(len(X)))).fit(X)
distances, indices = NN.kneighbors(X)

plt4.set_ylim(0, 0.5)
plt4.set_title('eps elbow')
plt4.plot(np.sort(distances[:, distances.shape[1]-1]), color='red')



plt1.set_xlim(-5, 5)
plt1.set_ylim(-15, 15)
plt1.set_title('C1 C2')
plt1.scatter(c1, c2, marker='.', s=25, edgecolor='k')

plt2.set_xlim(-5, 5)
plt2.set_ylim(-15, 15)
plt2.set_title('C1 C3')
plt2.scatter(c1, c3, marker='.', s=25, edgecolor='k')

plt3.set_xlim(-5, 5)
plt3.set_ylim(-15, 15)
plt3.set_title('C2 C3')
plt3.scatter(c2, c3, marker='.', s=25, edgecolor='k')

plt5.set_xlim(-5, 5)
plt5.set_ylim(-15, 15)
plt5.set_title('25000 points')
plt5.scatter(X[:, 0], X[:, 1], marker='o', s=25, edgecolor='k')


eps = 0.35
min_samples=np.log(len(X))*10
db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

print('Estimated number of clusters: %d' % n_clusters_)

unique, counts = np.unique(labels, return_counts=True)
print(dict(zip(unique, counts)))

plt6.set_xlim(-5, 5)
plt6.set_ylim(-15, 15)
plt6.set_title('DBSCAN')
plt6.scatter(X[:, 0], X[:, 1], marker='o', c=labels, s=25, edgecolor='k')



newX = np.zeros((len(X),2))
k=0
for i in range(0, len(X)):
	if distances[i][int(np.log(len(X))-1)] < eps:
		newX[k]=X[i]
		k=k+1
newX = newX[:k]
print(len(newX))

NN = NearestNeighbors(n_neighbors=int(np.log(len(newX))*10)).fit(newX)
distances, indices = NN.kneighbors(newX)

plt7.set_ylim(0, 1)
plt7.set_title('eps elbow')
plt7.plot(np.sort(distances[:, distances.shape[1]-1]), color='red')

plt8.set_xlim(-5, 5)
plt8.set_ylim(-15, 15)
plt8.set_title('Reduced')
plt8.scatter(newX[:, 0], newX[:, 1], marker='o',s=25, edgecolor='k')

eps = 0.35
min_samples=np.log(len(newX))*10
db = DBSCAN(eps=eps, min_samples=min_samples).fit(newX)
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

print('Estimated number of clusters: %d' % n_clusters_)

unique, counts = np.unique(labels, return_counts=True)
print(dict(zip(unique, counts)))

plt9.set_xlim(-5, 5)
plt9.set_ylim(-15, 15)
plt9.set_title('DBSCAN')
plt9.scatter(newX[:, 0], newX[:, 1], marker='o', c=labels, s=25, edgecolor='k')


plt.show() 

