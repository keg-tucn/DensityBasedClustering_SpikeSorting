import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
import seaborn as sns
import numbers
import math


from sklearn.neighbors import NearestNeighbors
from imblearn.pipeline import make_pipeline
from imblearn.under_sampling import (ClusterCentroids, RandomUnderSampler)

from sklearn.utils import check_array, check_random_state
from sklearn.utils import shuffle as shuffle_


fig, ((plt1, plt2, plt3), (plt4,plt5, plt6)) = plt.subplots(2, 3, figsize=(15, 9))


def generate_points(n_samples=100, centers=2, random_state=None):

    generator = check_random_state(random_state)

    centers = check_array(centers)
    n_features = centers.shape[1]

    X = []
    y = []

    n_centers = centers.shape[0]
    
    n_samples_per_center = [int(n_samples // n_centers)] * n_centers
    for i in range(n_samples % n_centers):
        n_samples_per_center[i] += 1
    
    for i, (n, std) in enumerate(zip(n_samples_per_center, np.ones(len(centers)))):
        X.append(centers[i] + generator.normal(scale=std, size=(n, n_features)))
        y += [i] * n

    X = np.concatenate(X)
    y = np.array(y)

    return X, y

n_samples = 10000
random_state = 170
centers = [(-9, -6), (-3, -3), (1, 0)]
X, y = generate_points(n_samples=n_samples, centers=centers, random_state=random_state)



for i in range(0, y.size):
	if y[i]==0:
		transformation = [[0.5, -0.25], [-0.5, 1]]
		X[i] = np.dot(X[i], transformation)	
	if y[i]==1:
		transformation = [[0.5, 0], [-0.5, 1]]
		X[i] = np.dot(X[i], transformation)	

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

eps=0.20
NN = NearestNeighbors(n_neighbors=len(X)).fit(X)
distances, indices = NN.kneighbors(X)


# Eps Distance Nearest Neighbour
# epsNeighbours[i] - nr of neighbours of point x, where distance is smaller than eps
epsNeighbours = np.zeros(len(X))
flagX = np.ones(len(X))
for i in range(0, len(X)-1):
	for j in range(0, len(X)-1):
		if distances[i][j]<eps:
			epsNeighbours[i] = epsNeighbours[i] +1
			flagX[j] = 0
plt2.set_title('eps distance neighbours')
plt2.plot(epsNeighbours, color='red')


sortedEpsNeighbours = np.sort(epsNeighbours)
plt3.set_title('eps distance neighbours sorted')
plt3.plot(sortedEpsNeighbours, color='red')

	
newX = np.zeros((len(X),2))
k=0
for i in range(0, len(X)):
	if flagX[i]==1:
		newX[k]=X[i]
		k=k+1
newX = newX[:k]

print(len(newX))

plt5.set_title('Rarefied data')
plt5.scatter(newX[:, 0], newX[:, 1], marker='o', s=25, edgecolor='k')

##############################################
#----------------- PLT6 ----------------------
##############################################

#plt6.set_title('DBSCAN')
#plt6.scatter(X[:, 0], X[:, 1], marker='o', c=labels, s=25, edgecolor='k')

plt.show()