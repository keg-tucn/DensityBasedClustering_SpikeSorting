import numpy as np
import sys

from sklearn import metrics
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans

sys.setrecursionlimit(100000)

import matplotlib.pyplot as plt

import Meeting11.alg as alg
import Meeting11.benchmarking as f
import Meeting11.functions as fs

dataName = ["S1", "S2", "U", "UO"]
algName = ["KMMEANS", "DBSCAN", "SBC V1", "SBC V2"]
files = ["s1_labeled.csv", "s2_labeled.csv", "unbalance.csv"]
kmeansValues = [15, 15, 8, 6]
epsValues = [27000, 45000, 18000, 0.5]
n = 25
for i in range(0, 4):
    if i<3:
        X = np.genfromtxt(files[i], delimiter=",")
        X, y = X[:, [0, 1]], X[:, 2]
    else:
        X, y = fs.getGenData()

    if i == 1:
        for k in range(len(X)):
            y[k] = y[k] - 1

    for j in range(0, 4):
        if j == 0:
            kmeans = KMeans(n_clusters=kmeansValues[i]).fit(X)
            labels = kmeans.labels_
        elif j == 1:
            if i==1:
                min_samples = np.log(len(X)) * 10
            else:
                min_samples = np.log(len(X))
            db = DBSCAN(eps=epsValues[i], min_samples=min_samples).fit(X)
            labels = db.labels_
        elif j == 2:
            labels = alg.nDimAlg(X, n, 1)
        elif j == 3:
            labels = alg.nDimAlg(X, n, 2)

        label_color = [f.LABEL_COLOR_MAP[l] for l in labels]
        fig = plt.figure()
        plt.scatter(X[:, 0], X[:, 1], marker='o', c=label_color, s=25, edgecolor='k')
        plt.show()

        # f.benchmark(labels, y)
        print(dataName[i] + " - " + algName[j] + " - "+ "ARI:" + str(metrics.adjusted_rand_score(labels, y)))
        print(dataName[i] + " - " + algName[j] + " - "+ "AMI:" + str(metrics.adjusted_mutual_info_score(labels, y)))