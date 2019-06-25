import numpy as np
import pandas as pd

from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN

from matplotlib import pyplot as plt

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


X = np.append(c1, c3, axis=1)

fig = plt.figure()
plt.scatter(X[:, 0], X[:, 1], marker='o', s=25, edgecolor='k')
plt.show()
sortedX = np.sort(X, axis=0)

distributionX = np.zeros(len(X))
k=0
for i in range(0, len(X)-1):
	distributionX[k] += 1
	if not sortedX[i,0]+1>sortedX[i+1,0]:
		k += 1
distributionX = distributionX[:k]

print(k)
fig = plt.figure()
plt.plot(distributionX, color='red')
plt.show() 

sortedY = np.sort(X, axis=1)
distributionY = np.zeros(len(X))
k=0
for i in range(0, len(X)):
	distributionY[k] += 1
	if not sortedY[i-1,1]+0.5>sortedY[i,1]:
		k += 1
distributionY = distributionY[:k]
fig = plt.figure()
plt.plot(distributionY, color='red')
plt.show() 