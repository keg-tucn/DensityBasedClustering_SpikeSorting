import numpy as np
import pandas as pd
import math
import random

from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN

from matplotlib import pyplot as plt


import Meeting10.BFS.functions as fs

#X = fs.getTINSData()
#labelsMatrix = np.genfromtxt("TINSlabels.csv", delimiter=",")

X = fs.getGenData()
labelsMatrix = np.genfromtxt("GENlabels.csv", delimiter=",")


n=25
xStart = -5
#xEnd = 5
xEnd = 15
yStart = 15
yEnd = -10
xIteration = (xEnd - xStart) / n
yIteration = (yEnd - yStart) / n
labelsX = fs.delabeling(X, labelsMatrix, n, xStart, xEnd, yStart, yEnd)

LABEL_COLOR_MAP = {-1: 'white',
				   0: 'w',
				   1: 'r',
				   2: 'b',
				   3: 'g',
				   4: 'k',
				   5: 'c',
				   6: 'y',
				   7: 'm',
				   8: 'tab:orange',
				   9: 'tab:brown',
				   10: 'tab:pink',
				   11: 'tab:gray',

				   }

label_color = [LABEL_COLOR_MAP[l] for l in labelsX]

fig = plt.figure()
ax=fig.gca()
ax.set_xticks(np.arange(xStart, xEnd, xIteration))
ax.set_yticks(np.arange(yStart, yEnd, yIteration))
plt.scatter(X[:, 0], X[:, 1], marker='o', c=label_color, s=25, edgecolor='k')
plt.grid(True)
plt.show()
plt.savefig('GEN.png')