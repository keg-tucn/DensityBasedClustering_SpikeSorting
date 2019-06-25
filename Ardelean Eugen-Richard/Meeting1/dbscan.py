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



fig, ((plt1, plt2, plt3, plt4)) = plt.subplots(1, 4, figsize=(18, 9))

# Importing the dataset
data = pd.read_csv('data.csv', skiprows=0)
f1 = data['F1'].values
f2 = data['F2'].values
f3 = data['F3'].values

print("Shape:")
print(data.shape)
print("\n")

#X = StandardScaler().fit_transform(data)
X = data





# for no undersampling
NN1 = NearestNeighbors(n_neighbors=np.log(X.size).astype(int)*10).fit(X)
distances1, indices1 = NN1.kneighbors(X)
plt2.set_ylim(-0, 1)
plt2.set_title('eps elbow')
plt2.plot(np.sort(distances1[:, distances1.shape[1]-1]), color='red', label = 'NoUndersample')








eps=0.25
min_samples=np.log(len(X))*10
db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

print('Estimated number of clusters: %d' % n_clusters_)

unique, counts = np.unique(labels, return_counts=True)
print(dict(zip(unique, counts)))

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(f1,f2,f3, c=labels.astype(np.float))


eps=0.4
min_samples=np.log(len(X))*10
db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
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
fig.savefig('dbscan-'+str(eps)+'-'+str(min_samples)+'.jpg', dpi=100) 


