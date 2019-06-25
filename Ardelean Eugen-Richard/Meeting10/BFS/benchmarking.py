import numpy as np
import queue
import sys
from timeit import default_timer as timer

from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN

sys.setrecursionlimit(100000)
from sklearn import preprocessing

import matplotlib.pyplot as plt

import Meeting10.BFS.functions as fs

LABEL_COLOR_MAP = {-1: 'white',
                   0: 'm',
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
                   12: 'tab:gray',
                   13: 'cyan',
                   14: 'fuchsia',
                   15: 'lightgreen',
                   16: 'orangered',
                   17: 'salmon',
                   18: 'silver',
                   19: 'yellowgreen',

                   }


def squareBFS(X, n = 25, minClusterSize = 10, expansionLimit = 0):
    # fig = plt.figure()
    # plt.scatter(X[:, 0], X[:, 1], marker='o', s=25, edgecolor='k')
    # plt.show()
    X = preprocessing.MinMaxScaler().fit_transform(X)

    xIteration = 1 / n
    yIteration = 1 / n
    matrix = fs.densityTable(X, n, 0, 1, 1, 0)
    np.savetxt("muie.csv", matrix, fmt="%1.0f", delimiter=",")
    """
    paddedMatrix = np.pad(matrix, 1, fs.pad_with, padder=-1)

    val = n * 10
    clusterCenters = np.zeros((val, 2))
    for i in range(0, val):
        start_x, start_y = np.random.randint(n), np.random.randint(n)
        start_queue = queue.Queue()
        start_queue.put((start_x, start_y))
        clusterCenters[i] = fs.BFSLocalMaxima(matrix, paddedMatrix, start_queue)

    clusterCenters = np.unique(clusterCenters, axis=0)
    """

    clusterCenters = fs.findLocalMaxima(matrix, minClusterSize)
    labelsMatrix = np.zeros((n, n), dtype=int)
    for i in range(0, len(clusterCenters)):
        start_queue = queue.Queue()
        start_queue.put((int(clusterCenters[i, 0]), int(clusterCenters[i, 1])))
        end_queue = queue.Queue()
        q, labelsMatrix = fs.BFS(matrix, labelsMatrix, start_queue, end_queue, expansionLimit)
        for q_item in q.queue:
            if labelsMatrix[q_item] != -1:
                labelsMatrix[q_item] = i + 1

    labelsX = fs.delabeling(X, labelsMatrix, n, 0, 1, 1, 0)

    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 0.5, np.amax(labelsX) + 2)]

    label_color = [LABEL_COLOR_MAP[l] for l in labelsX]
    #label_color = [colors[l] for l in labelsX]

    fig = plt.figure()
    ax = fig.gca()
    ax.set_xticks(np.arange(0, 1, xIteration))
    ax.set_yticks(np.arange(0, 1, yIteration))
    plt.scatter(X[:, 0], X[:, 1], marker='o', c=label_color, s=25, edgecolor='k')
    plt.grid(True)
    plt.show()

    return labelsX

def benchmark(labels, y):
    averageCorrectness = 0
    ponderateAverage = 0
    count = 0
    q = queue.Queue()
    for i in range(0, int(np.amax(y)) + 1):
        unique, counts = np.unique(labels[y == i], return_counts=True)
        k = 0
        for j in range(0, len(y)):
            if y[j] == i:
                k += 1

        results = np.array(list(zip(unique, counts)))
        results = results[results[:, 1].argsort()[::-1]]
        if results[0][0] == -1:
            if len(results) == 1:
                correct = 0
            else:
                if results[1][0] in q.queue:
                    correct = 0
                else:
                    correct = results[1][1]
                    q.put(results[1][0])
        else:
            if results[0][0] in q.queue:
                correct = 0
            else:
                correct = results[0][1]
                q.put(results[0][0])

        print(correct, k, correct / k)

        ponderateAverage += correct
        averageCorrectness += correct / k
        count += 1
    print("Average:" + str(averageCorrectness / count))
    print("Weighted Average:" + str(ponderateAverage / len(labels)) + " (" + str(ponderateAverage) + " out of " + str(len(labels)) + ")")


X = fs.getTINSData()
n=25
#start = timer()
#squareBFS(X, n, 10, 5)
#end = timer()
#print("MyALG TIME: "+str(end-start))

#start = timer()
kmeans = KMeans(n_clusters=3).fit(X)
labels = kmeans.labels_
#end = timer()
#print("KMEANS TIME: "+str(end-start))
label_color = [LABEL_COLOR_MAP[l] for l in labels]
fig = plt.figure()
plt.scatter(X[:, 0], X[:, 1], marker='o', c=label_color, s=25, edgecolor='k')
plt.show()

eps = 0.5
min_samples=np.log(len(X))*10
start = timer()
db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
labels = db.labels_
end = timer()
print("KMEANS TIME: "+str(end-start))
label_color = [f.LABEL_COLOR_MAP[l] for l in labels]
fig = plt.figure()
plt.scatter(X[:, 0], X[:, 1], marker='o', c=label_color, s=25, edgecolor='k')
plt.show()