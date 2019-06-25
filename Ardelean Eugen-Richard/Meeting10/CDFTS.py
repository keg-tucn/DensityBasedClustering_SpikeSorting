import numpy as np
import numpy.matlib
import math
import matplotlib.pyplot as plt

def normalize(Matrix):
    mins= np.amin(Matrix,axis=0)
    maxs= np.amax(Matrix,axis=0)
    dif= maxs-mins
    return (Matrix - mins)/dif


    """result = np.zeros((len(Matrix), 2))
    min0 = min(Matrix[:, 0])
    max0 = max(Matrix[:, 0])
    min1 = min(Matrix[:, 1])
    max1 = max(Matrix[:, 1])
    for i in range(0, len(Matrix)):
        result[i, 0] = (Matrix[i, 0] - min0) / (max0 - min0)
        result[i, 1] = (Matrix[i, 1] - min1) / (max1 - min1)
    return result
"""
def distances(Matrix):
    n= len(Matrix)
    distanceMatrix = np.zeros((n,n))
    for i in range(0, n):
        for j in range(0, i):
            distanceMatrix[i, j] =distanceMatrix[j,i] = math.sqrt(
                (Matrix[i, 1] - Matrix[j, 1]) * (Matrix[i, 1] - Matrix[j, 1])
                + (Matrix[i, 0] - Matrix[j, 0]) * (Matrix[i, 0] - Matrix[j, 0]))
    return distanceMatrix


def DScale(distanceMatrix,bandwidth,dataDimension):
    n = len(distanceMatrix)
    m = np.amax(distanceMatrix)
    newDistanceMatrix = distanceMatrix / m
    lb = newDistanceMatrix <= bandwidth
    rb = newDistanceMatrix > bandwidth
    den = np.sum(lb,axis= 1)
    rate = (den / n) ** (1 / dataDimension) / bandwidth
    rate = rate.reshape(n,1)
    rep = np.matlib.repmat(rate, 1, n)
    newDistanceMatrix[lb] = newDistanceMatrix[lb] * rep[lb]
    rep = rep * bandwidth
    newDistanceMatrix[rb] = (newDistanceMatrix[rb] - bandwidth) * (1 - rep[rb]) / (1 - bandwidth) + rep[rb]
    return newDistanceMatrix

def move(moved,movedTowards,magnitude):
        return moved + magnitude * (moved - movedTowards)


def CDFTS(Matrix, bandwidth, threshold, mt):
    processedMatrix = normalize(Matrix)
    delta = np.inf
    t = 1
    n,d = Matrix.shape
    tempMovedMatrix = np.zeros_like(processedMatrix)

    fig = plt.figure()
    plt.scatter(processedMatrix[:, 0], processedMatrix[:, 1], marker='.',  s=25, edgecolor='k', alpha=0.3)


    while delta > threshold and t<=mt:
        tempMatrix = np.zeros_like(processedMatrix)
        distanceMatrix = distances(processedMatrix)
        newDistanceMatrix = DScale(distanceMatrix,bandwidth,d)
        for z in range(n):
            for x in range(n):
                tempMovedMatrix[x] = move(processedMatrix[z],processedMatrix[x],newDistanceMatrix[z][x])
            tempMatrix[z] = np.sum(tempMovedMatrix,axis=0)/n
        tempMatrix = normalize(tempMatrix)
        delta= np.sum(np.abs(processedMatrix - tempMatrix))/(n*d)
        t+=1
        processedMatrix= tempMatrix

        fig = plt.figure()
        plt.scatter(processedMatrix[:, 0], processedMatrix[:, 1], marker='.', s=25, edgecolor='k', alpha=0.3)
        print("Delta: ",delta)
    print("Nr of iterations:",t-1)
    return processedMatrix