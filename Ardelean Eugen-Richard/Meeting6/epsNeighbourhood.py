import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
import numbers
import math


from sklearn.neighbors import NearestNeighbors

from sklearn.utils import check_array, check_random_state
from sklearn.utils import shuffle as shuffle_


fig, ((plt1, plt2, plt3), (plt4, plt5, plt6)) = plt.subplots(2, 3, figsize=(15, 9))


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

#n_samples = 100
n_samples = 10000
#random_state = 1
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

plt1.set_xlim(-5, 5)
plt1.set_ylim(-10, 5)
plt1.set_title('Original data')
plt1.scatter(X[:, 0], X[:, 1], marker='o', s=25, edgecolor='k')




###############################################################
################ ------ EPS NEIGHBOR ------- ##################
###############################################################
#eps=3
eps=0.2
NN = NearestNeighbors(n_neighbors=len(X)).fit(X)
distances, indices = NN.kneighbors(X)


# Eps Distance Nearest Neighbour
# neighboursInEpsRadius[i] - nr of neighbours of point x, where distance is smaller than eps

neighbourMatrix = np.full((len(X), len(X)), int(-1))
	
neighboursInEpsRadius = np.zeros(len(X))
for i in range(0, len(X)):
	k=0
	for j in range(0, len(X)):
		if distances[i][j]<eps:
			neighboursInEpsRadius[i] = neighboursInEpsRadius[i] + 1
			neighbourMatrix[i][k] = j
			k=k+1

plt2.set_title('eps distance neighbours')
plt2.plot(neighboursInEpsRadius, color='red')
			
sortedEpsNeighbours = np.sort(neighboursInEpsRadius)
plt3.set_title('eps distance neighbours sorted')
plt3.plot(sortedEpsNeighbours, color='red')
			

densityVector = np.zeros((len(X), 2))
for i in range(0, len(X)):
	densityVector[i] = [i, neighboursInEpsRadius[i]]
	
	
sortedDensityVector = densityVector[densityVector[:,1].argsort()[::-1]]



###############################################################
################ ------ HIGH DENSITY ------- ##################
###############################################################
highDensity = np.zeros((5000, 2))
for i in range(0, len(highDensity)):
	highDensity[i] = X[int(sortedDensityVector[i, 0])]

plt4.set_xlim(-5, 5)
plt4.set_ylim(-10, 5)
plt4.set_title('High Density data')
plt4.scatter(highDensity[:, 0], highDensity[:, 1], marker='o', s=25, edgecolor='k')



###############################################################
################ ------ RAREFICATION ------- ##################
###############################################################

k=0
densityThreshold=sortedDensityVector[5000,1]
flagX = np.ones(len(X))
for i in range(0, len(X)):
	if flagX[int(sortedDensityVector[i,0])]==1:
		flagX[int(sortedDensityVector[i,0])]=2
	if int(sortedDensityVector[i,1])>densityThreshold:
		for j in range(0, len(X)):
			if neighbourMatrix[int(sortedDensityVector[i,0])][j]==-1:
				break
			if flagX[neighbourMatrix[int(sortedDensityVector[i,0])][j]] == 1:
				flagX[neighbourMatrix[int(sortedDensityVector[i,0])][j]] = 0

newX = np.zeros((len(X),2))
k=0
for i in range(0, len(X)):
	if flagX[i]!=0:
		newX[k]=X[i]
		k=k+1
newX = newX[:k]

print(len(newX))

plt5.set_xlim(-5, 5)
plt5.set_ylim(-10, 5)
plt5.set_title('Rarefied data')
plt5.scatter(newX[:, 0], newX[:, 1], marker='o', s=25, edgecolor='k')

plt.show()