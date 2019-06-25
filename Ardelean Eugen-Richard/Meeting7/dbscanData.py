from copy import deepcopy
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import scipy.cluster.hierarchy as sch
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')



fig, ((plt1, plt2, plt3)) = plt.subplots(1, 3, figsize=(15, 9))

# Importing the dataset
data = pd.read_csv('data.csv', skiprows=0)
f1 = data['F1'].values
f2 = data['F2'].values
f3 = data['F3'].values

print("Shape:")
print(data.shape)
print("\n")

c1 = np.array([f1]).T
c2 = np.array([f1]).T
c3 = np.array([f1]).T


#X = StandardScaler().fit_transform(data)
X = np.append(c1, np.append(c2, c3, axis=1), axis=1)




# for no undersampling
NN1 = NearestNeighbors(n_neighbors=np.log(X.size).astype(int)).fit(X)
distances1, indices1 = NN1.kneighbors(X)

plt2.set_title('eps elbow')
plt2.plot(np.sort(distances1[:, distances1.shape[1]-1]), color='red')








eps=0.1
min_samples=np.log(len(X)*10)
db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

print('Estimated number of clusters: %d' % n_clusters_)

unique, counts = np.unique(labels, return_counts=True)
print(dict(zip(unique, counts)))



newX = np.zeros((len(X),3))
k=0
for index, number in zip(unique, counts):
	if number > len(X)/20 and not index==-1:
		for i in range(0, len(X)):
			if labels[i]==index:
				newX[k]=X[i]
				k=k+1
newX = newX[:k]
print(len(newX))


NN2 = NearestNeighbors(n_neighbors=np.log(newX.size).astype(int)).fit(newX)
distances2, indices2 = NN2.kneighbors(newX)

plt3.set_title('eps elbow')
plt3.plot(np.sort(distances2[:, distances2.shape[1]-1]), color='red')
plt3.legend()

eps=0.25
min_samples=np.log(len(newX))
db = DBSCAN(eps=eps, min_samples=min_samples).fit(newX)
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

print('Estimated number of clusters: %d' % n_clusters_)
unique, counts = np.unique(labels, return_counts=True)
print(dict(zip(unique, counts)))


fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(f1,f2,f3, c=labels.astype(np.float))
plt.show() 
