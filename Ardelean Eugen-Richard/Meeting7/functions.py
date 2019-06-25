import numpy as np

from sklearn.neighbors import NearestNeighbors

from sklearn.utils import check_array, check_random_state
from sklearn.utils import shuffle as shuffle_

def generate_points(n_samples=100, centers=2, random_state=None):

    generator = check_random_state(random_state)

    centers = check_array(centers)
    n_features = centers.shape[1]

    X = []
    y = []

    n_centers = centers.shape[0]
    
    n_samples_per_center = [int(n_samples // n_centers)] * n_centers
    for i in range(n_samples % n_centers):
        n_samples_per_center[i] += 1
    
    for i, (n, std) in enumerate(zip(n_samples_per_center, np.ones(len(centers)))):
        X.append(centers[i] + generator.normal(scale=std, size=(n, n_features)))
        y += [i] * n

    X = np.concatenate(X)
    y = np.array(y)

    return X, y
	
def neighbours(X, eps):
	neighbourMatrix = np.full((len(X), len(X)), int(-1))
	neighboursInEpsRadius = np.zeros(len(X))
	
	NN = NearestNeighbors(n_neighbors=len(X)).fit(X)
	distances, indices = NN.kneighbors(X)
	
	for i in range(0, len(X)):
		k=0
		for j in range(0, len(X)):
			if distances[i][j]<eps:
				neighboursInEpsRadius[i] = neighboursInEpsRadius[i] + 1
				neighbourMatrix[i][k] = j
				k=k+1
	return neighbourMatrix, neighboursInEpsRadius, distances