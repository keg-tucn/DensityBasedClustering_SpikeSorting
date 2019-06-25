
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import scipy.cluster.hierarchy as sch
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


from timeit import default_timer as timer



# Importing the dataset
data = pd.read_csv('data.csv', skiprows=0)
f1 = data['F1'].values
f2 = data['F2'].values
f3 = data['F3'].values

c1 = np.array([f1]).T
c2 = np.array([f2]).T
c3 = np.array([f3]).T


X = np.append(c1, c3, axis=1)
#X = StandardScaler().fit_transform(data)





# for no undersampling
NN1 = NearestNeighbors(n_neighbors=np.log(X.size).astype(int)*10).fit(X)
distances1, indices1 = NN1.kneighbors(X)
fig = plt.figure()
plt.plot(np.sort(distances1[:, distances1.shape[1]-1]), color='red', label = 'NoUndersample')
plt.show()


eps=0.2
min_samples=np.log(len(X))*10
start = timer()
db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
labels = db.labels_
end = timer()
print("KMEANS TIME: "+str(end-start))

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

print('Estimated number of clusters: %d' % n_clusters_)

unique, counts = np.unique(labels, return_counts=True)
print(dict(zip(unique, counts)))

fig = plt.figure()
plt.scatter(X[:, 0], X[:, 1], marker='o', c=labels, s=25, edgecolor='k')
plt.show()





