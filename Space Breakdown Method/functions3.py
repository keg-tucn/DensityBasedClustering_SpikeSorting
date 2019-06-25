import math
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from matplotlib import pyplot as plt
import sys

sys.setrecursionlimit(100000)


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


def correctCondition(matrix, currentX, currentY):
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


# TODO
def isValidCenter(value):
    return True  # TODO set the min value acceptable


def getNeighbours(p, shape):
    ndim = len(p)
    offsetIndexes = np.indices((3,) * ndim).reshape(ndim, -1).T
    offsets = np.r_[-1, 0, 1].take(offsetIndexes)
    offsets = offsets[np.any(offsets, 1)]

    neighbours = p + offsets

    valid = np.all((neighbours < np.array(shape)) & (neighbours >= 0), axis=1)
    neighbours = neighbours[valid]

    return neighbours


def isMaxima(array, point):
    neighbours = getNeighbours(point, np.shape(array))
    for neighbour in neighbours:
        if (array[tuple(neighbour)] > array[point]):
            return     False

    return True



def findClusterCenters(array, acceptanceTreshold= 5):
    clusterCenters = []
    for index, value in np.ndenumerate(array):
        if value>=acceptanceTreshold and isMaxima(array, index):  # TODO exclude neighbour centers
            clusterCenters.append(index)
    return clusterCenters


def findLocalMaxima(matrix):
    clusterCenters = np.zeros((len(matrix) ** 2, 2))

    k = 0
    for i in range(1, len(matrix) - 1):
        for j in range(1, len(matrix) - 1):
            if correctCondition(matrix, i, j) == 1:
                clusterCenters[k] = i, j
                k += 1
    clusterCenters = clusterCenters[:k]

    return clusterCenters


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


def getDropoff(ndArray, location):
    neighbours = getNeighbours(location,np.shape(ndArray))
    dropoff= 0
    for neighbour in neighbours:
        neighbourLocation = tuple(neighbour)
        dropoff+=((ndArray[location] - ndArray[neighbourLocation])**2)/ndArray[location]
    return math.sqrt(dropoff/len(neighbours))

def expand(array, start, labels, currentLabel, clusterCenters, version= 1):  # TODO
    visitet = np.zeros_like(array, dtype=bool)
    expansionQueue = []
    if labels[start] == 0:
        expansionQueue.append(start)
        labels[start] = currentLabel
    else:
        oldLabel = labels[start]
        disRez = dismbiguate(array,
                             start,
                             clusterCenters[currentLabel - 1],
                             clusterCenters[oldLabel - 1])
        if disRez == 1:
            labels[start] = currentLabel
            expansionQueue.append(start)
        elif disRez == 11:
            labels[labels == oldLabel] = currentLabel
            expansionQueue.append(start)
        elif disRez == 22:
            labels[labels == currentLabel] = oldLabel
            currentLabel = oldLabel
            expansionQueue.append(start)

    visitet[start] = True

    dropoff= getDropoff(array, start)

    while expansionQueue:
        point = expansionQueue.pop(0)
        neigbours = getNeighbours(point, np.shape(array))
        for neigbour in neigbours:
            location = tuple(neigbour)
            if array[location] == 0:
                pass
            if (location == (5, 14)):
                a = 1

            if (not visitet[location]) and (dropoff*math.sqrt(distance(start,point)) < array[location] <= array[point]):
                visitet[location] = True
                if labels[location] == currentLabel:
                    expansionQueue.append(location)
                elif labels[location] == 0:
                    expansionQueue.append(location)
                    labels[location] = currentLabel
                else:
                    if version == 2:
                        labels[location]=-1
                    else:
                        oldLabel = labels[location]
                        disRez = dismbiguate(array,
                                             location,
                                             clusterCenters[currentLabel - 1],
                                             clusterCenters[oldLabel - 1])
                        if disRez == 1:
                            labels[location] = currentLabel
                            expansionQueue.append(location)
                        elif disRez == 11:
                            labels[labels == oldLabel] = currentLabel
                            expansionQueue.append(location)
                        elif disRez == 22:
                            labels[labels == currentLabel] = oldLabel
                            currentLabel = oldLabel
                            expansionQueue.append(location)

    return labels


"""def expand(array, start): #TODO
    visitet= np.zeros_like(array,dtype=bool)
    q = []
    q.append(start)
    visitet[start] = True
    expandedCluster = []
    expandedCluster.append(start)
    while q:
        point = q.pop(0)
        neigbours = getNeighbours(point,np.shape(array))
        for neigbour in neigbours:
            neigbourTuple = tuple(neigbour)
            if (not visitet[neigbourTuple]) and 0 < array[neigbourTuple] <= array[point]:
                q.append(neigbourTuple)
                expandedCluster.append(neigbourTuple)
                visitet[neigbourTuple] = True

    return expandedCluster
"""


def BFS(matrix, labelsMatrix, queue=None, end=None):
    if queue.qsize() == 0:
        return end, labelsMatrix

    current_index = queue.get()
    end.put(current_index)
    current_x, current_y = current_index[0], current_index[1]

    for n in range(current_x - 1, current_x + 2):
        for m in range(current_y - 1, current_y + 2):
            if not (n == current_x and m == current_y) and n > -1 and m > -1 and n < matrix.shape[0] and m < \
                    matrix.shape[1] and (n, m) not in end.queue:
                if matrix[current_x, current_y] >= matrix[n, m] and matrix[n, m] != 0:
                    if labelsMatrix[n, m] != 0:
                        labelsMatrix[n, m] = -1
                    else:
                        end.put((n, m))
                        queue.put((n, m))

    return BFS(matrix, labelsMatrix, queue, end)


def getTINSData2():
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

def getTINSData2():
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

    X = np.hstack((c1,c2, c3))
    chanceKeep=1
    keep = np.random.choice(2, len(X), p=[1-chanceKeep, chanceKeep])
    keep = keep==1
    X=X[keep]
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


def getGenData(plotFig=False):
    np.random.seed(0)
    avgPoints = 250
    C1 = [-2, 0] + .8 * np.random.randn(avgPoints * 2, 2)

    C4 = [-2, 3] + .3 * np.random.randn(avgPoints // 5, 2)
    L4 = np.full(len(C4), 1).reshape((len(C4), 1))

    C3 = [1, -2] + .2 * np.random.randn(avgPoints * 5, 2)
    C5 = [3, -2] + 1.0 * np.random.randn(avgPoints * 4, 2)
    L5 = np.full(len(C5), 2)

    C2 = [4, -1] + .1 * np.random.randn(avgPoints, 2)
    # L2 = np.full(len(C2), 1).reshape((len(C2), 1))

    C6 = [5, 6] + 1.0 * np.random.randn(avgPoints * 5, 2)
    L6 = np.full(len(C6), 3).reshape((len(C6), 1))

    if plotFig:
        plt.plot(C1[:, 0], C1[:, 1], 'b.', alpha=0.3)
        plt.plot(C2[:, 0], C2[:, 1], 'r.', alpha=0.3)
        plt.plot(C3[:, 0], C3[:, 1], 'g.', alpha=0.3)
        plt.plot(C4[:, 0], C4[:, 1], 'c.', alpha=0.3)
        plt.plot(C5[:, 0], C5[:, 1], 'm.', alpha=0.3)
        plt.plot(C6[:, 0], C6[:, 1], 'y.', alpha=0.3)
        plt.figure()
        plt.plot(C1[:, 0], C1[:, 1],'b.',  alpha=0.3)
        plt.plot(C2[:, 0], C2[:, 1],'b.',  alpha=0.3)
        plt.plot(C3[:, 0], C3[:, 1],'b.',  alpha=0.3)
        plt.plot(C4[:, 0], C4[:, 1],'b.',  alpha=0.3)
        plt.plot(C5[:, 0], C5[:, 1],'b.',  alpha=0.3)
        plt.plot(C6[:, 0], C6[:, 1],'b.',  alpha=0.3)

    plt.show()
    X = np.vstack((C1, C2, C3, C4, C5, C6))

    c1Labels = np.full(len(C1),1)
    c2Labels = np.full(len(C2),2)
    c3Labels = np.full(len(C3),3)
    c4Labels = np.full(len(C4),4)
    c5Labels = np.full(len(C5),5)
    c6Labels = np.full(len(C6),6)

    y = np.hstack((c1Labels,c2Labels,c3Labels,c4Labels,c5Labels,c6Labels))
    return X, y


"""X is a 2D array with "number of points" lines and "number of dimensions" columns 
   X contains values between (including) 0 and pn """
def chunkify(X, pn, includeMax=False):
    nrDim = np.shape(X)[1]
    nArray = np.zeros((pn,) * nrDim, dtype=int)

    for point in X:
        if np.all(point<pn):
            location = tuple(np.floor(point).astype(int))
            nArray[location]+=1
        else:#TODO
            #print(point)
            pass
    return nArray

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


"""return array of "number of points" length with the label for each point"""
def dechunkify(X,labelsArray,pn, includeMax=False):
    pointLabels = np.zeros(len(X), dtype=int)

    for index in range(0,len(X)):
        point = X[index]
        if np.all(point<pn):
            location = tuple(np.floor(point).astype(int))
            pointLabels[index]=labelsArray[location]
        else:#TODO
            pointLabels[index] = -1
    return pointLabels




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
                        if labelsMatrix[i, j] == -1:
                            labelsX[k] = -1
                        else:
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

def dismbiguate(array, questionPoint, cluster1, cluster2, treshold=0):
    if (cluster1 == questionPoint) or (cluster2 == questionPoint):
        if array[cluster1] > array[cluster2]:
            return 11
        else:
            return 22

    # cluster 2 was expanded first, but it is actually connected to a bigger cluster
    if array[cluster2] == array[questionPoint]:
        return 11

    distanceToC1 = distance(questionPoint, cluster1)
    distanceToC2 = distance(questionPoint, cluster2)
    pointStrength = array[questionPoint]

    c1Strength = array[cluster1]/pointStrength - getDropoff(array,cluster1)*distanceToC1
    c2Strength = array[cluster2]/pointStrength - getDropoff(array,cluster2)*distanceToC1

    # c1Strength = array[cluster1]*distanceToC1/(array[cluster1] - pointStrength)
    # c2Strength = array[cluster2]*distanceToC2/(array[cluster2] - pointStrength)
    #c2Strength = (array[cluster2] / pointStrength) / distanceToC2
    if(questionPoint==(5,14)):
        a=1

    if (abs(c1Strength - c2Strength) < treshold):
        return 0
    if c1Strength > c2Strength:
        return 1
    else:
        return 2


def sortCenters(matrix, clusterCenters2):
    l = []
    for point in clusterCenters2:
        l.append((matrix[point], point))
    l.sort(reverse=True)

    return [x[1] for x in l]
