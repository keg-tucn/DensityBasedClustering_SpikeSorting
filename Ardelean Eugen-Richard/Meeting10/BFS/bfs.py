import numpy as np
import queue
import sys

sys.setrecursionlimit(5000)

from matplotlib import pyplot as plt

n = 25
#matrix = np.genfromtxt("TINSmatrix.csv", delimiter=",")
matrix = np.genfromtxt("GENmatrix.csv", delimiter=",")


def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 10)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value
    return vector


paddedMatrix = np.pad(matrix, 1, pad_with, padder=-1)


def countZeroNeighbours(matrix, currentX, currentY):
    k = 0
    for i in range(-1, 2):
        for j in range(-1, 2):
            if matrix[currentX + i, currentY + j] == 0:
                k += 1
    return k


def stopCondition(matrix, currentX, currentY):
    flag = 1
    if countZeroNeighbours(matrix, currentX, currentY) < 4 and matrix[currentX, currentY]>5:
        for i in range(-1, 2):
            for j in range(-1, 2):
                if not (i == 0 and j == 0):
                    if matrix[currentX, currentY] < matrix[currentX + i, currentY + j]:  # or matrix[currentX+i, currentY+j] ==0:
                        flag = 0
    else:
        flag = 0
    return flag

def BFS(queue=None, end=None, condition=1):
    current_index = queue.get()
    if condition == 2:
        end.put(current_index)  # ????
    current_x, current_y = current_index[0], current_index[1]

    currentSize = queue.qsize()
    if condition == 1:
        if stopCondition(paddedMatrix, current_x + 1, current_y + 1) == 1:
            return current_x, current_y

    for n in range(current_x - 1, current_x + 2):
        for m in range(current_y - 1, current_y + 2):
            if not (n == current_x and m == current_y) and n > -1 and m > -1 and n < matrix.shape[0] and m < \
                    matrix.shape[1] and (n, m) not in queue.queue:
                if condition == 1:
                    queue.put((n, m))
                else:
                    if matrix[current_x, current_y] > matrix[n, m] and matrix[n, m] != 0 and clustersMatrix[n,m]==0:
                        end.put((n, m))
                        queue.put((n, m))

    if condition == 2:
        if currentSize == queue.qsize():
            return end

    return BFS(queue, end, condition)

val = n * 4
clusterCenters = np.zeros((val, 2))
for i in range(0, val):
    start_x, start_y = np.random.randint(n), np.random.randint(n)
    start_queue = queue.Queue()
    start_queue.put((start_x, start_y))
    clusterCenters[i] = BFS(start_queue, None, 1)

clusterCenters = np.unique(clusterCenters, axis=0)
print(clusterCenters)
print(len(clusterCenters))

clustersMatrix = np.zeros((n, n), dtype=int)
for i in range(0, len(clusterCenters)):
    start_queue = queue.Queue()
    start_queue.put((int(clusterCenters[i, 0]), int(clusterCenters[i, 1])))
    end_queue = queue.Queue()
    q = BFS(start_queue, end_queue, 2)
    for q_item in q.queue:
        clustersMatrix[q_item] = i + 1



#np.savetxt("TINSlabels.csv", clustersMatrix, fmt="%1.0f", delimiter=",")
np.savetxt("GENlabels.csv", clustersMatrix, fmt="%1.0f", delimiter=",")

plt.show()
