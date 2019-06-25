import numpy as np
import sys
from timeit import default_timer as timer

from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans

sys.setrecursionlimit(100000)

import Meeting10.BFS.functions as fs
import Meeting10.BFS.benchmarking as f

n=25

# X = np.genfromtxt("unbalance.csv", delimiter=",")
# X = np.genfromtxt("s1_labeled.csv", delimiter=",")
# X = np.genfromtxt("s2_labeled.csv", delimiter=",")
# X, y = X[:, [0,1]], X[:, 2]
X, y = fs.getGenData()



start = timer()
# kmeans = KMeans(n_clusters=8).fit(X) #unbalanced
kmeans = KMeans(n_clusters=15).fit(X) #s1, s2
kmeans = KMeans(n_clusters=4).fit(X) #gen

labels = kmeans.labels_
end = timer()
f.benchmark(labels, y)
print("KMEANS TIME: "+str(end-start))


#eps = 5000 #unbalanced
#min_samples=np.log(len(X)) #unbalanced
# eps = 45000 #s2
# min_samples=np.log(len(X))*10 #s2
eps = 27000 #s1
min_samples=np.log(len(X)) #s1
# eps = 0.5 #gen
# min_samples=np.log(len(X)) #gen   8

start = timer()
db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
labels = db.labels_
end = timer()
f.benchmark(labels, y)
print("DBSCAN TIME: "+str(end-start))

start = timer()
labels = f.squareBFS(X, n)
end = timer()
f.benchmark(labels, y)
print("MyALG TIME: "+str(end-start))