import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
import seaborn as sns
import numbers

from sklearn.neighbors import NearestNeighbors
from imblearn.pipeline import make_pipeline
from imblearn.under_sampling import (ClusterCentroids, RandomUnderSampler)

from sklearn.utils import check_array, check_random_state
from sklearn.utils import shuffle as shuffle_


fig, ((plt1, plt2, plt3), (plt4,plt5, plt6)) = plt.subplots(2, 3, figsize=(15, 9))

def make_blobs(n_samples=100, n_features=2, centers=2, cluster_std=1.0, center_box=(-10.0, 10.0), shuffle=True, random_state=None):
    """Generate isotropic Gaussian blobs for clustering.
    Read more in the :ref:`User Guide <sample_generators>`.
    Parameters
    ----------
    n_samples : int, or tuple, optional (default=100)
        The total number of points equally divided among clusters.
    n_features : int, optional (default=2)
        The number of features for each sample.
    centers : int or array of shape [n_centers, n_features], optional
        (default=3)
        The number of centers to generate, or the fixed center locations.
    cluster_std: float or sequence of floats, optional (default=1.0)
        The standard deviation of the clusters.
    center_box: pair of floats (min, max), optional (default=(-10.0, 10.0))
        The bounding box for each cluster center when centers are
        generated at random.
    shuffle : boolean, optional (default=True)
        Shuffle the samples.
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    Returns
    -------
    X : array of shape [n_samples, n_features]
        The generated samples.
    y : array of shape [n_samples]
        The integer labels for cluster membership of each sample.
    Examples
    --------
    >>> from sklearn.datasets.samples_generator import make_blobs
    >>> X, y = make_blobs(n_samples=10, centers=3, n_features=2,
    ...                   random_state=0)
    >>> print(X.shape)
    (10, 2)
    >>> y
    array([0, 0, 1, 0, 2, 2, 2, 1, 1, 0])
    See also
    --------
    make_classification: a more intricate variant
    """
    generator = check_random_state(random_state)

    if isinstance(centers, numbers.Integral):
        centers = generator.uniform(center_box[0], center_box[1],
                                    size=(centers, n_features))
    else:
        centers = check_array(centers)
        n_features = centers.shape[1]

    if isinstance(cluster_std, numbers.Real):
        cluster_std = np.ones(len(centers)) * cluster_std

    X = []
    y = []

    n_centers = centers.shape[0]
    if isinstance(n_samples, numbers.Integral):
        n_samples_per_center = [int(n_samples // n_centers)] * n_centers
        for i in range(n_samples % n_centers):
            n_samples_per_center[i] += 1
    else:
        n_samples_per_center = n_samples

    for i, (n, std) in enumerate(zip(n_samples_per_center, cluster_std)):
        X.append(centers[i] + generator.normal(scale=std, size=(n, n_features)))
        y += [i] * n

    X = np.concatenate(X)
    y = np.array(y)

    if shuffle:
        X, y = shuffle_(X, y, random_state=generator)

    return X, y

n_samples = 10000
random_state = 170
centers = [(-9, -6), (-3, -3), (1, 0)]
X, y = make_blobs(n_samples=n_samples, centers=centers, random_state=random_state)


for i in range(0, y.size):
	if y[i]==0:
		transformation = [[0.5, -0.25], [-0.5, 1]]
		X[i] = np.dot(X[i], transformation)	
	if y[i]==1:
		transformation = [[0.5, 0], [-0.5, 1]]
		X[i] = np.dot(X[i], transformation)	

n1=n_samples//40
n2=n_samples//50
n3=n_samples//160

plt1.set_title('Original data')
plt1.scatter(X[:, 0], X[:, 1], marker='o', s=25, edgecolor='k')

X = np.vstack((X[y == 0][:n1], X[y == 1][:n2], X[y == 2][:n3]))


newy = np.concatenate((np.full((n1,1),0), np.full((n2,1),1), np.full((n3,1),2)))


colors = ['#ef8a62' if v == 0 else '#f7f7f7' if v == 1 else '#67a9cf' for v in newy]		
plt2.set_title('Different density data')
plt2.scatter(X[:, 0], X[:, 1], marker='o',c=colors,  s=25, edgecolor='k')


sampler = RandomUnderSampler(random_state=0)
X_res, y_res = sampler.fit_resample(X, newy)
print(X.shape)
print(X_res.shape)
colors = ['#ef8a62' if v == 0 else '#f7f7f7' if v == 1 else '#67a9cf' for v in y_res]
plt3.set_title('Undersampled data')
plt3.scatter(X_res[:, 0], X_res[:, 1], c=colors, linewidth=0.5, edgecolor='black')


NN = NearestNeighbors(n_neighbors=len(X)).fit(X)
distances, indices = NN.kneighbors(X)

print(distances)


plt4.set_title('eps elbow')
plt4.plot(np.sort(distances[:, distances.shape[0]-1]))

eps=0.5
min_samples=np.log10(n_samples)
db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

print('Estimated number of clusters: %d' % n_clusters_)

unique, counts = np.unique(labels, return_counts=True)
print(dict(zip(unique, counts)))
plt5.set_title('Different density data DBSCAN')
plt5.scatter(X[:, 0], X[:, 1], marker='o', c=labels, s=25, edgecolor='k')



eps=0.38
min_samples=np.log10(n_samples)
db = DBSCAN(eps=eps, min_samples=min_samples).fit(X_res)
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

print('Estimated number of clusters: %d' % n_clusters_)

unique, counts = np.unique(labels, return_counts=True)
print(dict(zip(unique, counts)))


plt6.set_title('US DBSCAN')
plt6.scatter(X_res[:, 0], X_res[:, 1], marker='o', c=labels, s=25, edgecolor='k')
plt.show()
#fig.savefig('dbscanGenUs2.jpg', dpi=100)