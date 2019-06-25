import numpy as np
import pandas as pd
import math
import random

from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN

from matplotlib import pyplot as plt

from scipy.spatial import Voronoi, voronoi_plot_2d

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
initialX = X

vor = Voronoi(X)
voronoi_plot_2d(vor)
plt.show()