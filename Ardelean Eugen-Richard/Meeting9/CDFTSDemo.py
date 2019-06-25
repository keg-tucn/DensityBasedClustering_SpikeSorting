# Authors: Shane Grigsby <refuge@rocktalus.com>
#          Amy X. Zhang <axz@mit.edu>
# License: BSD 3 clause


import matplotlib.gridspec as gridspec
from matplotlib import colors as mcolors


import numpy as np

import matplotlib.pyplot as plt
#import Lic2.OPTICS as op
from sklearn.cluster import DBSCAN

import CDFTS as ct
# Generate sample data

np.random.seed(0)
avgPoints = 250

colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
colors.pop('w')
colors.pop('k')

C1 = [-2, 0] + .8 * np.random.randn(avgPoints*2, 2)
C4 = [-2, 3] + .3 * np.random.randn(avgPoints//5, 2)
# C3 = [1, -2] + .2 * np.random.randn(avgPoints*5, 2)
C5 = [3, -2] + 1.6 * np.random.randn(avgPoints, 2)
C2 = [4, -1] + .1 * np.random.randn(avgPoints//2, 2)
C6 = [5, 6] + 2 * np.random.randn(avgPoints, 2)
X = np.vstack((C1, C2, C4, C5, C6))

plt.plot(C1[:, 0], C1[:, 1], 'b.', alpha=0.3)
plt.plot(C2[:, 0], C2[:, 1], 'r.', alpha=0.3)
# plt.plot(C3[:, 0], C3[:, 1], 'g.', alpha=0.3)
plt.plot(C4[:, 0], C4[:, 1], 'c.', alpha=0.3)
plt.plot(C5[:, 0], C5[:, 1], 'm.', alpha=0.3)
plt.plot(C6[:, 0], C6[:, 1], 'y.', alpha=0.3)


#plt.figure(figsize=(10, 7))
#G = gridspec.GridSpec(2, 3)
#ax1 = plt.subplot(G[0, :3])
#ax2 = plt.subplot(G[0, 2])
#ax3 = plt.subplot(G[1, 0])
#ax4 = plt.subplot(G[1, 1])
#ax5 = plt.subplot(G[1, 2])
eps = 0.2
minPts = 5
"""
newX = ct.CDFTS(X,0.2,0.005,10)


#print(opres[0])
#opres[1][opres[1]==np.inf] =3
db = DBSCAN(eps=eps, min_samples=minPts).fit(newX)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

print('eps = ' + str(eps))
print('min_samples = ' + str(minPts))
print('Estimated number of clusters: %d' % n_clusters_)

unique, counts = np.unique(labels, return_counts=True)
print(dict(zip(unique, counts)))

fig = plt.figure()
plt.scatter(newX[:, 0], newX[:, 1], marker='.', c=labels, s=25, edgecolor='k',alpha=0.3)
"""
plt.show()
