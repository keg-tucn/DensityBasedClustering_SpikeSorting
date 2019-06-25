import math
import queue
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from matplotlib import pyplot as plt

def adjust(x):
    if x < 0:
        x = np.floor(x)
        x = np.floor(x / 5)
    else:
        x = np.ceil(x)
        x = np.ceil(x / 5)
    x = 5 * x
    return x

def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 10)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value
    return vector


def countZeroNeighbours(matrix, currentX, currentY):
    k = 0
    for i in range(-1, 2):
        for j in range(-1, 2):
            if matrix[currentX + i, currentY + j] == 0:
                k += 1
    return k


def stopCondition(queue, matrix, currentX, currentY):
    flag = 1
    if countZeroNeighbours(matrix, currentX, currentY) < 6 and matrix[currentX, currentY] > 5:
        for i in range(-1, 2):
            for j in range(-1, 2):
                if not (i == 0 and j == 0):
                    if matrix[currentX, currentY] < matrix[currentX + i, currentY + j]:
                        # or matrix[currentX+i, currentY+j] ==0:
                        flag = 0
    else:
        flag = 0
    return flag


def BFSLocalMaxima(matrix, paddedMatrix, queue=None):
    current_index = queue.get()
    current_x, current_y = current_index[0], current_index[1]

    if stopCondition(queue, paddedMatrix, current_x + 1, current_y + 1) == 1:
        return current_x, current_y

    for n in range(current_x - 1, current_x + 2):
        for m in range(current_y - 1, current_y + 2):
            if not (n == current_x and m == current_y) and n > -1 and m > -1 and n < matrix.shape[0] and m < \
                    matrix.shape[1] and (n, m) not in queue.queue:
                queue.put((n, m))

    return BFSLocalMaxima(matrix, paddedMatrix, queue)


def BFS(matrix, labelsMatrix, queue=None, end=None):
    current_index = queue.get()
    end.put(current_index)
    current_x, current_y = current_index[0], current_index[1]

    currentSize = queue.qsize()

    for n in range(current_x - 1, current_x + 2):
        for m in range(current_y - 1, current_y + 2):
            if not (n == current_x and m == current_y) and n > -1 and m > -1 and n < matrix.shape[0] and m < \
                    matrix.shape[1] and (n, m) not in queue.queue:
                if matrix[current_x, current_y] >= matrix[n, m] and matrix[n, m] != 0 and labelsMatrix[n,m]==0:
                    end.put((n, m))
                    queue.put((n, m))

    if currentSize == queue.qsize():
        return end

    return BFS(matrix, labelsMatrix, queue, end)


def getTINSData():
    # Importing the dataset
    data = pd.read_csv('data.csv', skiprows=0)
    f1 = data['F1'].values
    f2 = data['F2'].values
    f3 = data['F3'].values

    # print("Shape:")
    # print(data.shape)
    # print("\n")

    c1 = np.array([f1]).T
    c2 = np.array([f2]).T
    c3 = np.array([f3]).T

    X = np.append(c1, c3, axis=1)
    return X


def getTINStrimmedData():
    # Importing the dataset
    data = pd.read_csv('data.csv', skiprows=0)
    f1 = data['F1'].values
    f2 = data['F2'].values
    f3 = data['F3'].values

    # print("Shape:")
    # print(data.shape)
    # print("\n")

    c1 = np.array([f1]).T
    c2 = np.array([f2]).T
    c3 = np.array([f3]).T

    X = np.append(c1, c3, axis=1)
    X = approximationScheme(X)
    return X


def getGenData():
    np.random.seed(0)
    avgPoints = 250
    C1 = [-2, 0] + .8 * np.random.randn(avgPoints * 2, 2)
    C4 = [-2, 3] + .3 * np.random.randn(avgPoints // 5, 2)
    # C3 = [1, -2] + .2 * np.random.randn(avgPoints*5, 2)
    C5 = [3, -2] + 1.0 * np.random.randn(avgPoints * 4, 2)
    C2 = [4, -1] + .1 * np.random.randn(avgPoints, 2)
    C6 = [5, 6] + 1.0 * np.random.randn(avgPoints * 5, 2)
    X = np.vstack((C1, C2, C4, C5, C6))
    return X


def densityTable(X, n, xStart, xEnd, yStart, yEnd):
    matrix = np.zeros((n, n), dtype=int)
    xIteration = (xEnd - xStart) / n
    yIteration = (yEnd - yStart) / n
    yCurrent = yStart
    i = 0
    while yCurrent > yEnd:
        j = 0
        xCurrent = xStart
        while xCurrent < xEnd:
            for k in range(0, len(X)):
                if xCurrent < X[k, 0] < xCurrent + xIteration and yCurrent + yIteration < X[k, 1] < yCurrent:
                    matrix[i, j] += 1
            xCurrent += xIteration
            j += 1
        yCurrent += yIteration
        i += 1
    return matrix


def delabeling(X, labelsMatrix, n, xStart, xEnd, yStart, yEnd):
    labelsX = np.full(len(X), -1)
    xIteration = (xEnd - xStart) / n
    yIteration = (yEnd - yStart) / n
    yCurrent = yStart
    i = 0
    while yCurrent > yEnd:
        j = 0
        xCurrent = xStart
        while xCurrent < xEnd:
            if labelsMatrix[i, j] != 0:
                for k in range(0, len(X)):
                    if xCurrent < X[k, 0] < xCurrent + xIteration and yCurrent + yIteration < X[k, 1] < yCurrent:
                        labelsX[k] = int(labelsMatrix[i, j]) - 1
            xCurrent += xIteration
            j += 1
        yCurrent += yIteration
        i += 1
    return labelsX


def distance(pointA, pointB):
    sum = 0
    for i in range(0, len(pointA)):
        sum += (pointA[i] - pointB[i]) ** 2
    return math.sqrt(sum)


def approximationScheme(X):
    newX = np.zeros((len(X), len(X[0])))
    k = 0
    newX[k] = X[0]
    last = newX[k]
    for i in range(1, len(X)):
        if distance(X[i], last) > 2.5:
            k = k + 1
            newX[k] = X[i]
            last = newX[k]
    newX = newX[:k]
    print('Initial length: ' + str(len(X)))
    print('Rarefied length: ' + str(len(newX)))
    return newX


def keepNoise(input, labels):
    output = np.zeros((len(input), 2))
    k = 0
    for i in range(0, len(input)):
        if labels[i] == -1:
            output[k] = input[i]
            k = k + 1
    output = output[:k]
    print('NOISE:' + str(len(output)))
    return output


def applyDBSCAN(X, eps, min_samples):
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    print('DBSCAN: Estimated number of clusters: %d' % n_clusters_)

    unique, counts = np.unique(labels, return_counts=True)
    print('DBSCAN:' + str(dict(zip(unique, counts))))

    return labels


def countOnes(list):
    k = 0
    for i in range(0, len(list)):
        if list[i] == 1:
            k += 1
    return k


def getIndice(list):
    for i in range(0, len(list)):
        if list[i] == 1:
            return i
