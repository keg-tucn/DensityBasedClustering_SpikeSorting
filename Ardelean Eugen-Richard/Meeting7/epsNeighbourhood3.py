import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
import numbers
import math
import random


import functions

from sklearn.neighbors import NearestNeighbors
from imblearn.pipeline import make_pipeline
from imblearn.under_sampling import (ClusterCentroids, RandomUnderSampler)



fig, ((plt1, plt2, plt3), (plt4,plt5, plt6), (plt7,plt8, plt9)) = plt.subplots(3, 3, figsize=(15, 9))


n_samples = 10000
random_state = 170
centers = [(-12, -7), (-4, -2), (2, 1)]
X, y = functions.generate_points(n_samples=n_samples, centers=centers, random_state=random_state)



for i in range(0, y.size):
	if y[i]==0:
		transformation = [[0.5, -0.25], [-0.5, 1]]
		X[i] = np.dot(X[i], transformation)	
	if y[i]==1:
		transformation = [[0.5, 0], [-0.5, 1]]
		X[i] = np.dot(X[i], transformation)	
#"""
n1=n_samples//4
n2=n_samples//5
n3=n_samples//16

X = np.vstack((X[y == 0][:n1], X[y == 1][:n2], X[y == 2][:n3]))
#"""
plt1.set_xlim(-5, 5)
plt1.set_ylim(-10, 5)
plt1.set_title('Original data')
plt1.scatter(X[:, 0], X[:, 1], marker='o', s=25, edgecolor='k')
##############################################
#----------------- PLT4 - ELBOW ----------------------
##############################################
# for no undersampling
NN1 = NearestNeighbors(n_neighbors=np.log(X.size).astype(int)).fit(X)
distances1, indices1 = NN1.kneighbors(X)

plt4.set_title('eps elbow')
plt4.plot(np.sort(distances1[:, distances1.shape[1]-1]), color='red', label = 'NoUndersample')
plt4.legend()

print(X.shape)
print(len(X))
eps=0.25
NN = NearestNeighbors(n_neighbors=len(X)).fit(X)
distances, indices = NN.kneighbors(X)


# Eps Distance Nearest Neighbour
# epsNeighbours[i] - nr of neighbours of point x, where distance is smaller than eps
epsNeighbours = np.zeros(len(X))
for i in range(0, len(X)):
	for j in range(0, len(X)):
		if distances[i][j]<eps:
			epsNeighbours[i] = epsNeighbours[i] +1

plt2.set_title('eps distance neighbours')
plt2.plot(epsNeighbours, color='red')


sortedEpsNeighbours = np.sort(epsNeighbours)
#print(sortedEpsNeighbours[len(sortedEpsNeighbours)-1])
plt3.set_title('eps distance neighbours sorted')
plt3.plot(sortedEpsNeighbours, color='red')


min_samples=np.log(n_samples)
db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
labels = db.labels_


# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

print('Estimated number of clusters: %d' % n_clusters_)
unique, counts = np.unique(labels, return_counts=True)
print(dict(zip(unique, counts)))

plt5.set_xlim(-5, 5)
plt5.set_ylim(-10, 5)
plt5.set_title('DBSCAN')
plt5.scatter(X[:, 0], X[:, 1], marker='o', c=labels, s=25, edgecolor='k')

newX = np.zeros((len(X),2))
k=0
heighestNeighbour = 0
averageDensity = 0

averageOutlierDensity = 0
q=0
for i in range(0, len(X)):
	if labels[i]==-1:
		averageOutlierDensity+=epsNeighbours[i]
		q=q+1
print(q)
print(averageOutlierDensity/q)
averageOutlierDensity /= q
		
for index, number in zip(unique, counts):
	if not index==-1:
		heighestNeighbour = 0
		averageDensity = 0
		q=0
		for i in range(0, len(X)):
			if labels[i]==index:
				if epsNeighbours[i]>heighestNeighbour:
					heighestNeighbour = epsNeighbours[i]
				averageDensity+=epsNeighbours[i]
				q=q+1
		print(index, heighestNeighbour, averageDensity/q)
		averageDensity /= q
		for i in range(0, len(X)):
			if labels[i]==index and random.random() > averageOutlierDensity/averageDensity:
				newX[k]=X[i]
				k=k+1
newX = newX[:k]
print(len(newX))


plt6.set_xlim(-5, 5)
plt6.set_ylim(-10, 5)
plt6.set_title('Rarefied')
plt6.scatter(newX[:, 0], newX[:, 1], marker='o', s=25, edgecolor='k')


NN2 = NearestNeighbors(n_neighbors=np.log(newX.size).astype(int)).fit(newX)
distances2, indices2 = NN2.kneighbors(newX)

plt7.set_title('eps elbow')
plt7.plot(np.sort(distances2[:, distances2.shape[1]-1]), color='red', label = 'NoUndersample')
plt7.legend()

eps=0.25

min_samples=np.log(len(newX))
db = DBSCAN(eps=eps, min_samples=min_samples).fit(newX)
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

print('Estimated number of clusters: %d' % n_clusters_)
unique, counts = np.unique(labels, return_counts=True)
print(dict(zip(unique, counts)))

plt8.set_xlim(-5, 5)
plt8.set_ylim(-10, 5)
plt8.set_title('DBSCAN')
plt8.scatter(newX[:, 0], newX[:, 1], marker='o', c=labels, s=25, edgecolor='k')


plt.show()
plt.show()