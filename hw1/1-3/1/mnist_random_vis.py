import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

matplotlib.interactive(True)
accs = np.load('./mnist_rand_accs.npy')
losss = np.load('./mnist_rand_losss.npy')
epoch = [i+1 for i in range(len(accs[0]))]

plt.plot(epoch, accs[0], 'o', label="train_acc", markersize=2)
plt.plot(epoch, accs[1], 'o', label="test_acc", markersize=2)
plt.xlabel('epoch')
plt.ylabel('acc')
plt.title('Model Accuracy by Epoch')
plt.legend()
plt.savefig('./accs.png')
plt.close()

plt.plot(epoch, losss[0], 'o', label="train_loss", markersize=2)
plt.plot(epoch, losss[1], 'o', label="test_loss", markersize=2)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('Model loss by Epoch')
plt.legend()
plt.savefig('./losss.png')
plt.close()