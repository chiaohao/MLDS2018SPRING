import numpy as np
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

weights = np.load('mnist_weights.npy')
accs = np.load('mnist_accs.npy')

weights = weights.reshape((240, -1))
accs = accs.reshape((8, 30))

print(weights.shape)
print(accs.shape)

pca = PCA(n_components=2)
x = pca.fit(weights).transform(weights).reshape((8, 30, 2))

fig = plt.figure()
fig.suptitle('Param-Loss', fontsize=14)
colors = [(0,0,0), (1,0,0), (0,1,0), (0,0,1), (1,1,0), (1,0,1), (0,1,1), (0.5,0.5,0.5)]

for i in range(8):
    plt.scatter(x[:, :, 0], x[:, :, 1], marker='.', alpha=0.0)
    for j in range(30):
        plt.text(x[i, j, 0], x[i, j, 1], str(round(accs[i, j], 2)), color=colors[i])

fig = plt.gcf()
fig.savefig('output1-2.1_mnist.jpg', dpi=200)
