import numpy as np
import queue
import sys

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


def squareBFS(X, n):
    # fig = plt.figure()
    # plt.scatter(X[:, 0], X[:, 1], marker='o', s=25, edgecolor='k')
    # plt.show()

    X = preprocessing.MinMaxScaler().fit_transform(X)

    xIteration = 1 / n
    yIteration = 1 / n
    matrix = fs.densityTable(X, n, 0, 1, 1, 0)
    paddedMatrix = np.pad(matrix, 1, fs.pad_with, padder=-1)
    np.savetxt("muie.csv", matrix, fmt="%1.0f", delimiter=",")

    val = n * 4
    clusterCenters = np.zeros((val, 2))
    for i in range(0, val):
        start_x, start_y = np.random.randint(n), np.random.randint(n)
        start_queue = queue.Queue()
        start_queue.put((start_x, start_y))
        clusterCenters[i] = fs.BFSLocalMaxima(matrix, paddedMatrix, start_queue)

    clusterCenters = np.unique(clusterCenters, axis=0)

    labelsMatrix = np.zeros((n, n), dtype=int)
    for i in range(0, len(clusterCenters)):
        start_queue = queue.Queue()
        start_queue.put((int(clusterCenters[i, 0]), int(clusterCenters[i, 1])))
        end_queue = queue.Queue()
        q = fs.BFS(matrix, labelsMatrix, start_queue, end_queue)
        for q_item in q.queue:
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


X = fs.getGenData()
n = 25


"""
X = np.genfromtxt("s1_labeled.csv", delimiter=",")
X, y = X[:, [0,1]], X[:, 2]
fig = plt.figure()
plt.scatter(X[:, 0], X[:, 1], marker='o', s=25, edgecolor='k')
plt.show()

labels = squareBFS(X, n)

#missing label 2
for i in range(0, len(y)):
    if y[i] >= 3:
        y[i] = y[i] - 1

for i in range(0, int(np.amax(y))+1):
    unique, counts = np.unique(labels[y==i], return_counts=True)
    k=0
    for j in range(0, len(y)):
        if y[j] == i:
            k+=1
    correct = np.amax(counts)
    print(correct, k, correct/k)

X = np.genfromtxt("s2_labeled.csv", delimiter=",")
X, y = X[:, [0, 1]], X[:, 2]
fig = plt.figure()
plt.scatter(X[:, 0], X[:, 1], marker='o', s=25, edgecolor='k')
plt.show()

labels = squareBFS(X, n)

for i in range(0, int(np.amax(y)) + 1):
    unique, counts = np.unique(labels[y == i], return_counts=True)
    k = 0
    for j in range(0, len(y)):
        if y[j] == i:
            k += 1
    correct = np.amax(counts)
    print(correct, k, correct / k)
"""

"""
X = np.genfromtxt("unbalance.csv", delimiter=",")
X, y = X[:, [0,1]], X[:, 2]

labels = squareBFS(X, n)

for i in range(0, int(np.amax(y)) + 1):
    unique, counts = np.unique(labels[y == i], return_counts=True)
    k = 0
    for j in range(0, len(y)):
        if y[j] == i:
            k += 1
    correct = np.amax(counts)
    print(correct, k, correct / k)
"""