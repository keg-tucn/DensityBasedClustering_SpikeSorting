import numpy as np
import sys


sys.setrecursionlimit(100000)
from sklearn import preprocessing

import matplotlib.pyplot as plt

import Meeting12.functions3 as fs

LABEL_COLOR_MAP = {-1: 'gray',
                   0: 'white',
                   1: 'r',
                   2: 'b',
                   3: 'g',
                   4: 'k',
                   5: 'c',
                   6: 'y',
                   7: 'tab:purple',
                   8: 'tab:orange',
                   9: 'tab:brown',
                   10: 'tab:pink',
                   11: 'lime',
                   13: 'cyan',
                   14: 'fuchsia',
                   15: 'lightgreen',
                   16: 'orangered',
                   17: 'salmon',
                   18: 'silver',
                   19: 'yellowgreen',
                   20: 'aqua',
                   21: 'beige',
                   22: 'crimson',
                   23: 'indigo',
                   24: 'darkblue',
                   25: 'gold',
                   26: 'ivory',
                   27: 'lavender',
                   28: 'lightblue',
                   29: 'olive',
                   30: 'sienna',
                   31: 'salmon',
                   32: 'teal',
                   33: 'turquoise',
                   34: 'wheat',
                   12: 'orchid'

                   }

def nDimAlg(X, pn, version=1, plot= False):
    X = preprocessing.MinMaxScaler((0, pn)).fit_transform(X)
    ndArray = fs.chunkify(X, pn)
    clusterCenters = fs.findClusterCenters(ndArray)
    #clusterCenters = fs.sortCenters(ndArray, clusterCenters)
    #print(clusterCenters)

    # labelsMatrix = np.full(np.shape(ndArray),-1, dtype=int)
    # labelsMatrix2 = np.full(np.shape(ndArray),-1, dtype=int)
    labelsMatrix = np.zeros_like(ndArray, dtype=int)
    labelsMatrix2 = np.zeros_like(ndArray, dtype=int)
    for labelM1 in range(len(clusterCenters)):
        point = clusterCenters[labelM1]
        if labelsMatrix[point] != 0:
            continue # cluster was already discovered
        labelsMatrix =fs.expand(ndArray, point,labelsMatrix,labelM1+1,clusterCenters, version=version)

    #bring cluster labels back to (-1) - ("nr of clusters"-2) range
    uniqueClusterLabels= np.unique(labelsMatrix)
    nrClust = len(uniqueClusterLabels)
    for label in range(len(uniqueClusterLabels)):
        if uniqueClusterLabels[label] == -1 or uniqueClusterLabels[label]==0: # don`t remark noise/ conflicta
            nrClust -=1
            continue
        labelsMatrix2[labelsMatrix == uniqueClusterLabels[label]] = label

    labels = fs.dechunkify(X,labelsMatrix,pn)#TODO 2

    #print("number of actual clusters: ", nrClust)

    if plot:#TODO
        nrDim = len(X[0])
        label_color = [LABEL_COLOR_MAP[l] for l in labels]
        fig = plt.figure()
        if nrDim == 2:
            ax = fig.gca()
            ax.set_xticks(np.arange(0, pn, 1))
            ax.set_yticks(np.arange(0, pn, 1))
            plt.scatter(X[:, 0], X[:, 1], marker='o', c=label_color, s=25, edgecolor='k')
            plt.grid(True)
        if nrDim == 3:
            from mpl_toolkits.mplot3d import Axes3D
            ax = Axes3D(fig)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            #ax.set_xticks(np.arange(0, pn, 1))
            #ax.set_zticks(np.arange(0, pn, 1))
            #ax.set_yticks(np.arange(0, pn, 1))
            ax.scatter(X[:, 0], X[:, 1],X[:, 2], marker='.', c=label_color, s=25, )
            #plt.grid(True)

#        plt.show()
    return labels

n=25
# X,_=fs.getGenData()
# labels = nDimAlg(X,n,plot=True)
# # plt.show()
X2=fs.getTINSData2()
labels2 = nDimAlg(X2,n,plot=True)
plt.show()

from sklearn import metrics
print("done")
print(metrics.silhouette_score(X2, labels2, metric='euclidean'))
# print("done")
