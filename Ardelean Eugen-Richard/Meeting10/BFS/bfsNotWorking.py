import numpy as np
import pandas as pd
import queue
import math
import random
import sys
sys.setrecursionlimit(2000)

from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN

from matplotlib import pyplot as plt

import functions as fs

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


X = np.append(c1, c3, axis=1)
X = fs.approximationScheme(X)

fig = plt.figure()
plt.scatter(X[:, 0], X[:, 1], marker='o', s=25, edgecolor='k')

n=25
#matrix = fs.densityTable(X, n)
#np.savetxt("foo.csv", matrix, fmt="%1.0f", delimiter=",")
matrix = np.genfromtxt("foo.csv", delimiter=",")
print(matrix.astype(int))


def pad_with(vector, pad_width, iaxis, kwargs):
	pad_value = kwargs.get('padder', 10)
	vector[:pad_width[0]] = pad_value
	vector[-pad_width[1]:] = pad_value
	return vector
	
paddedMatrix = np.pad(matrix, 1, pad_with, padder = -1)
print(paddedMatrix.astype(int))

def stopCondition(matrix, currentX, currentY):
	flag = 1
	for i in range(-1, 2):
		for j in range(-1, 2):
			if not (i==0 and j ==0):
				if matrix[currentX , currentY] <= matrix[currentX+i, currentY+j] or matrix[currentX+i, currentY+j] ==0:
					flag = 0
	return flag

def BFS(queue=None, condition=1):
	
	current_index = queue.get()
	current_x, current_y = current_index[0],current_index[1]
	
	currentSize = queue.qsize()
	if condition == 1:
		if stopCondition(paddedMatrix, current_x + 1, current_y + 1) == 1: 
			return current_x, current_y
	elif condition == 2:
		if matrix[current_x, current_y] < 4 or clustersMatrix[current_x, current_y]!=0:
			return queue

	for n in range(current_x-1, current_x+2):
		for m in range(current_y-1, current_y+2):
			if not (n==current_x and m==current_y)  and n>-1 and m>-1  and n<matrix.shape[0] and m<matrix.shape[1]  and (n,m) not in queue.queue:
				queue.put((n,m))
	
	if condition == 2:
		if currentSize == queue.qsize():
			return queue
	
	return BFS(queue, condition)


	
val = 20
clusterCenters = np.zeros((val, 2))
for i in range(0, 20):
	start_x,start_y = np.random.randint(25), np.random.randint(25)
	start_queue = queue.Queue()
	start_queue.put((start_x,start_y))
	clusterCenters[i] = BFS(start_queue, 1)
	
clusterCenters =  np.unique(clusterCenters, axis=0)
print(clusterCenters)

clustersMatrix = np.zeros((n,n), dtype=int)
for i in range(0, len(clusterCenters)):
	start_queue = queue.Queue()
	start_queue.put((int(clusterCenters[i,0]), int(clusterCenters[i,1])))
	q = BFS(start_queue, 2)
	for q_item in q.queue:
		clustersMatrix[q_item] = i+1

np.savetxt("labels.csv", clustersMatrix, fmt="%1.0f", delimiter=",")



plt.show()