import numpy as np
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys

g_norm = np.load(sys.argv[1])
loss = np.load(sys.argv[2])

fig = plt.figure()

fig.suptitle(sys.argv[3] + ' Gradient Norm', fontsize=14)
plt.plot(np.array([i+1 for i in range(len(g_norm))]), g_norm, '-o', linewidth=1, markersize=0)
plt.xlabel('Iteration', fontsize=10)
plt.ylabel('Grad_Norm', fontsize=10)
fig = plt.gcf()
fig.savefig(sys.argv[3] + '_grad_norm.png', dpi=200)

fig = plt.figure()

fig.suptitle(sys.argv[3] + ' Loss', fontsize=14)
plt.plot(np.array([i+1 for i in range(len(loss))]), loss, '-o', linewidth=1, markersize=0)
plt.xlabel('Iteration', fontsize=10)
plt.ylabel('Loss', fontsize=10)
fig = plt.gcf()
fig.savefig(sys.argv[3] + '_loss.png', dpi=200)


