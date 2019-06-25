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
centers = [(-12, -7), (-4, -2), (2, 1)]
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

NN = NearestNeighbors(n_neighbors=len(X)).fit(X)
distances, indices = NN.kneighbors(X)


maxDistance = distances[len(X)-1, len(X)-1]
scaledDistanceMatrix = np.full((len(X), len(X)), int(-1))
bandwidth = 0.1

# eps = bandwidth
def DScaleFunction(inputDistanceMatrix, eps, dataDimension):
	outputDistanceMatrix = np.full((len(X), len(X)), int(-1))

	k=0
	for i in range(0, len(X)):
		for j in range(0, len(X)):
			if inputDistanceMatrix[i,j]<eps:
				k++
		r = (maxDistance/eps) * ( k/len(X))**(1/dataDimension)
		for j in range(0, len(X)):
			if inputDistanceMatrix[i,j]<eps:
				outputDistanceMatrix[i,j] = inputDistanceMatrix[i,j] * r
			else
				outputDistanceMatrix[i,j] = (inputDistanceMatrix[i,j] - eps) * (m-eps*r)/(m-eps) + eps*r
	return outputDistanceMatrix

# min-max normalization
def min_max_normalization(matrix):
	result = np.zeros((len(matrix),2))
	for i in range(0, len(matrix)):
		result[i,0] = (matrix[i,0] - min(matrix[:,0])) / (max(matrix[:,0]) - min(matrix[:,0]))
		result[i,1] = (matrix[i,1] - min(matrix[:,1])) / (max(matrix[:,1]) - min(matrix[:,1]))
	return result
	
def CLFTS(inputData, eps, shiftThreshold):
	shiftData = np.zeros((len(X),2))
	normData = np.zeros((len(X),2))
	distanceMatrix = np.full((len(X), len(X)), int(-1))
	scaledDistanceMatrix = np.full((len(X), len(X)), int(-1))
	
	normData = min_max_normalization(inputData)
	normDataSaved = normData
	
	delta = Integer.MAX_VALUE
	t = 1

	while delta > shiftThreshold: 
		#calculate distance matrix for inputData
		for i in range(0, len(normData)):
			for j in range(0, len(normData)):
				distanceMatrix[i,j] = math.sqrt(
							(normData[i,1]-normData[j,1])*(normData[i,1]-normData[j,1])
							+(normData[i,0]-normData[j,0])*(normData[i,0]-normData[j,0]))
			
		scaledDistanceMatrix = DScaleFunction(distanceMatrix, eps, len(inputData[0]))
		
		normData = min_max_normalization(normData)
		
		sum = 0
		for i in range(0, len(inputData)):
			for j in range(0, len(inputData)):
				sum = sum + abs(
		delta = 1 / (len(inputData)*len(inputData[0])) * 
		t=t+1
	shiftData = normData	
	return shiftData
	
plt.show()