import math
import numpy as np
from sklearn.cluster import DBSCAN
from matplotlib import pyplot as plt


def keepNoise(input, labels):
	output = np.zeros((len(input),2))
	k=0
	for i in range(0, len(input)):
		if labels[i]==-1:
			output[k]=input[i]
			k=k+1
	output=output[:k]
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
	k=0
	for i in range(0, len(list)):
		if list[i] == 1:
			k+=1
	return k

def getIndice(list):
	for i in range(0, len(list)):
		if list[i] == 1:
			return i