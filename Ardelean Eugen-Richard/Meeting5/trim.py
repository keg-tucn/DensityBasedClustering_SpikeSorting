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


fig, ((plt1, plt2), (plt4,plt5)) = plt.subplots(2, 2, figsize=(15, 9))


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
#centers = [(-9, -6), (-3, -3), (1, 0)]
centers = [(-12, -7), (-4, -2), (2, 1)]
X, y = generate_points(n_samples=n_samples, centers=centers, random_state=random_state)




for i in range(0, y.size):
	if y[i]==0:
		transformation = [[0.5, -0.25], [-0.5, 1]]
		X[i] = np.dot(X[i], transformation)	
	if y[i]==1:
		transformation = [[0.5, 0], [-0.5, 1]]
		X[i] = np.dot(X[i], transformation)	

n1=n_samples//4
n2=n_samples//5
n3=n_samples//16

X = np.vstack((X[y == 0][:n1], X[y == 1][:n2], X[y == 2][:n3]))
		
plt1.set_title('Original data')
plt1.scatter(X[:, 0], X[:, 1], marker='o', s=25, edgecolor='k')


# trimming

	
newX = np.zeros((len(X),2))
k=0
newX[k] = X[0]
last=newX[k]
for i in range(1, len(X)):
	if math.sqrt((X[i][0]-last[0])**2+(X[i][1]-last[1])**2) > 2.5:
		k=k+1
		newX[k]=X[i]
		last = newX[k]
newX = newX[:k]
print(len(newX))

plt2.set_title('Rarefied data')
plt2.scatter(newX[:, 0], newX[:, 1], marker='o', s=25, edgecolor='k')


NN1 = NearestNeighbors(n_neighbors=np.log(len(newX)).astype(int)).fit(newX)
distances1, indices1 = NN1.kneighbors(newX)

plt4.set_ylim(0, 1)
plt4.set_title('eps elbow')
plt4.plot(np.sort(distances1[:, distances1.shape[1]-1]), color='red')

eps=0.75
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