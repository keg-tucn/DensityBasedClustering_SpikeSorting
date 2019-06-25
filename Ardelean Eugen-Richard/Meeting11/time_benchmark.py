import numpy as np
import sys
from timeit import default_timer as timer

from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans


sys.setrecursionlimit(100000)

import Meeting11.alg as alg
import Meeting11.functions as fs


dataName = ["S1", "S2", "U", "UO"]
files = ["s1_labeled.csv", "s2_labeled.csv", "unbalance.csv"]
kmeansValues = [15, 15, 8, 6]
epsValues = [27000, 45000, 18000, 0.5]
n = 25
numberOfIterations = 100
for i in range(0, 4):
    if i<3:
        X = np.genfromtxt(files[i], delimiter=",")
        X, y = X[:, [0, 1]], X[:, 2]
    else:
        X, y = fs.getGenData()

    sum = 0
    for j in range(0, numberOfIterations):
        start = timer()
        kmeans = KMeans(n_clusters=kmeansValues[i]).fit(X)

        labels = kmeans.labels_
        end = timer()
        sum = sum + (end - start)
    print(dataName[i] + " - KMEANS TIME: "+str(sum/numberOfIterations))



    sum = 0
    min_samples = np.log(len(X))
    for j in range(0, numberOfIterations):
        start = timer()
        db = DBSCAN(eps=epsValues[i], min_samples=min_samples).fit(X)
        labels = db.labels_
        end = timer()
        sum = sum + (end - start)
    print(dataName[i] + " - DBSCAN TIME: "+str(sum/numberOfIterations))

    sum = 0
    for j in range(0, numberOfIterations):
        start = timer()
        labels = alg.nDimAlg(X, n, 1)
        end = timer()
        sum = sum + (end - start)
    print(dataName[i] + " - SBC V1 TIME: "+str(sum/numberOfIterations))

    sum = 0
    for j in range(0, numberOfIterations):
        start = timer()
        labels = alg.nDimAlg(X, n, 2)
        end = timer()
        sum = sum + (end - start)
    print(dataName[i] + " - SBC V2 TIME: "+str(sum/numberOfIterations))
