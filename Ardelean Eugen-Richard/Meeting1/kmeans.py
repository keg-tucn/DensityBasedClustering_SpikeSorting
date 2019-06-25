import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')

# Importing the dataset
data = pd.read_csv('data.csv', skiprows=0)

f1 = data['F1'].values
f2 = data['F2'].values
f3 = data['F3'].values

print("Shape:")
print(data.shape)
print("\n")
print("Head:")
print(data.head())
print("\n")
print("Describe:")
print(data.describe())
print("\n")
print(data.isna().sum())

kmeans = KMeans(n_clusters=4)

# Fitting the input data
kmeans = kmeans.fit(data)

# Getting the cluster labels
labels = kmeans.predict(data)
labels = kmeans.labels_

# Centroid values
C = kmeans.cluster_centers_

print("Cluster Centers:")
print(C)
print("\n")

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(f1, f2, f3, c=labels.astype(np.float))
ax.scatter(C[:, 0], C[:, 1], C[:, 2], marker='*', c='#050505', s=1000)
plt.show()
fig.savefig('cluster4.jpg', dpi=100)