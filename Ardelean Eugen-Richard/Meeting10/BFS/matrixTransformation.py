import numpy as np
import sys
sys.setrecursionlimit(5000)

from matplotlib import pyplot as plt

import Meeting10.BFS.functions as fs

#X = fs.getTINStrimmedData()
X = fs.getGenData()


fig = plt.figure()
plt.scatter(X[:, 0], X[:, 1], marker='o', s=25, edgecolor='k')

n=25
xStart = -5
#xEnd = 5
xEnd = 15
yStart = 15
yEnd = -10
matrix = fs.densityTable(X, n, xStart, xEnd, yStart, yEnd)

#np.savetxt("TINSmatrix.csv", matrix, fmt="%1.0f", delimiter=",")
np.savetxt("GENmatrix.csv", matrix, fmt="%1.0f", delimiter=",")

plt.show()