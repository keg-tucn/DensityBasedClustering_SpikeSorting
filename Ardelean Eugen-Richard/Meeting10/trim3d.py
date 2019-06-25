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


import functions as fs


# Importing the dataset
data = pd.read_csv('data.csv', skiprows=0)
f1 = data['F1'].values
f2 = data['F2'].values
f3 = data['F3'].values

c1 = np.array([f1]).T
c2 = np.array([f2]).T
c3 = np.array([f3]).T



X = np.append(np.append(c1, c2, axis=1), c3, axis=1)

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(X[:,0] , X[:,1], X[:,2])

newX = fs.approximationScheme(X)

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(newX[:,0] , newX[:,1], newX[:,2])
plt.show() 



