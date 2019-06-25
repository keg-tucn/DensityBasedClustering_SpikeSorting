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


#start = timer()
#squareBFS(X, n, 10, 5)
#end = timer()
#print("MyALG TIME: "+str(end-start))

# #start = timer()
# kmeans = KMeans(n_clusters=3).fit(X)
# labels = kmeans.labels_
# #end = timer()
# #print("KMEANS TIME: "+str(end-start))
# label_color = [LABEL_COLOR_MAP[l] for l in labels]
# fig = plt.figure()
# plt.scatter(X[:, 0], X[:, 1], marker='o', c=label_color, s=25, edgecolor='k')
# plt.show()

# eps = 0.25
# min_samples=np.log(len(X))*10
# start = timer()
# db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
# labels = db.labels_
# end = timer()
# print("KMEANS TIME: "+str(end-start))
# label_color = [LABEL_COLOR_MAP[l] for l in labels]
# fig = plt.figure()
# plt.scatter(X[:, 0], X[:, 1], marker='o', c=label_color, s=25, edgecolor='k')
# plt.show()