import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

matplotlib.interactive(True)
params = np.load('./mnist_gen_params.npy')
accs = np.load('./mnist_gen_accs.npy')
losss = np.load('./mnist_gen_losss.npy')

hist = []
new_accs = [[],[]]
new_losss = [[],[]]
for i,v in enumerate(params):
	if v not in hist:
		hist.append(v)
		new_accs[0].append(accs[0][i])
		new_accs[1].append(accs[1][i])
		new_losss[0].append(losss[0][i])
		new_losss[1].append(losss[1][i])

print("Acutual Model Num: {}".format(len(hist)))

plt.plot(hist, new_accs[0], 'o', label="train_acc")
plt.plot(hist, new_accs[1], 'o', label="test_acc")
plt.xlabel('parameters')
plt.ylabel('acc')
plt.title('Model Accuracy by Patameters')
plt.legend()
plt.savefig('./accs.png')
plt.close()

plt.plot(hist, new_losss[0], 'o', label="train_loss")
plt.plot(hist, new_losss[1], 'o', label="test_loss")
plt.xlabel('parameters')
plt.ylabel('loss')
plt.title('Model loss by Patameters')
plt.legend()
plt.savefig('./losss.png')
plt.close()
