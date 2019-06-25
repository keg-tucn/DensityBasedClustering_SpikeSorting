import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
import seaborn as sns
import numbers
import math


import functions

from sklearn.neighbors import NearestNeighbors




fig, ((plt1, plt2, plt3), (plt4, plt5, plt6)) = plt.subplots(2, 3, figsize=(15, 9))


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


eps=0.3


neighbourMatrix, neighboursInEpsRadius, distances = functions.neighbours(X, eps)

plt3.set_title('eps elbow')
plt3.plot(np.sort(distances[:, distances.shape[1]-1]), color='red', label = 'NoUndersample')
plt3.legend()

sortedEpsNeighbours = np.sort(neighboursInEpsRadius)
plt2.set_title('eps distance neighbours sorted')
plt2.plot(sortedEpsNeighbours, color='red')





			

densityVector = np.zeros((len(X), 2))
for i in range(0, len(X)):
	densityVector[i] = [i, neighboursInEpsRadius[i]]
	
	
sortedDescDensityVector = densityVector[densityVector[:,1].argsort()[::-1]]
sortedDensityVector = densityVector[densityVector[:,1].argsort()]


print(sortedDensityVector[1000,1])



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

###############################################################
################ ------ RAREFICATION ------- ##################
###############################################################


newX = np.zeros((len(X),2))
k=0
for i in range(0, len(X)):
	if neighboursInEpsRadius[i]>neighboursInEpsRadius[1000]:
		newX[k]=X[i]
		k=k+1
newX = newX[:k]
print(len(newX))


min_samples=np.log(len(newX))
db = DBSCAN(eps=eps, min_samples=min_samples).fit(newX)
labels = db.labels_


# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

print('Estimated number of clusters: %d' % n_clusters_)
unique, counts = np.unique(labels, return_counts=True)
print(dict(zip(unique, counts)))


plt6.set_xlim(-5, 5)
plt6.set_ylim(-10, 5)
plt6.set_title('Rarefied data')
plt6.scatter(newX[:, 0], newX[:, 1], c=labels, marker='o', s=25, edgecolor='k')

plt.show()