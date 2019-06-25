import numpy as np
from matplotlib import pyplot as plt

# X = np.genfromtxt("unbalance.csv", delimiter=",")
# X = np.genfromtxt("s1_labeled.csv", delimiter=",")
#X = np.genfromtxt("s2_labeled.csv", delimiter=",")
#X, y = X[:, [0,1]], X[:, 2]

#plt.scatter(X[:, 0], X[:, 1], marker='o', s=25, edgecolor='k')
#plt.show()


np.random.seed(0)
avgPoints = 250
C1 = [-2, 0] + .8 * np.random.randn(avgPoints * 2, 2)

C4 = [-1, 1.5] + .3 * np.random.randn(avgPoints // 5, 2)
# C4 = [-2, 3] + .3 * np.random.randn(avgPoints // 5, 2)

# C3 = [1, -2] + .2 * np.random.randn(avgPoints*5, 2)
C5 = [3, -2] + 1.0 * np.random.randn(avgPoints * 4, 2)

# C2 = [4, -1] + .1 * np.random.randn(avgPoints, 2)

C6 = [5, 6] + 1.0 * np.random.randn(avgPoints * 5, 2)
X = np.vstack((C1, C4, C5, C6))

plt.plot(C1[:, 0], C1[:, 1], 'b.', alpha=0.3)
# plt.plot(C2[:, 0], C2[:, 1], 'c.', alpha=1)
# plt.plot(C3[:, 0], C3[:, 1], 'g.', alpha=0.3)
plt.plot(C4[:, 0], C4[:, 1], 'r.', alpha=1)
plt.plot(C5[:, 0], C5[:, 1], 'm.', alpha=0.3)
plt.plot(C6[:, 0], C6[:, 1], 'y.', alpha=0.3)

plt.show()