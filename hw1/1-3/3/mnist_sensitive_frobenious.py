import os, math
import numpy as np
import urllib.request
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable, grad
from mnist_sensitive_model import Net, batch_sizes
import matplotlib
import matplotlib.pyplot as plt

def sensitive(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    optimizer = optim.Adagrad(model.parameters())
    for data, target in test_loader:
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, requires_grad=True), Variable(target)
        output = model(data)
        test_loss = F.nll_loss(output, target, size_average=False)
        _grads = grad(test_loss, data, create_graph=True)
        for _grad in _grads:
            for g in _grad:
                sensitivities[i].append(g.norm().cpu().data.numpy())         
    
if __name__ == "__main__":
    models = []
    models.append(torch.load('mnist_s64_e200_a98.19_l0.0759.pt'))
    models.append(torch.load('mnist_s128_e200_a98.25_l0.0702.pt'))
    models.append(torch.load('mnist_s256_e200_a98.04_l0.0709.pt'))
    models.append(torch.load('mnist_s512_e200_a98.15_l0.0694.pt'))
    models.append(torch.load('mnist_s1024_e200_a98.07_l0.0735.pt'))
    hists = []
    hists.append(np.load('mnist_hist_s64.npy'))
    hists.append(np.load('mnist_hist_s128.npy'))
    hists.append(np.load('mnist_hist_s256.npy'))
    hists.append(np.load('mnist_hist_s512.npy'))
    hists.append(np.load('mnist_hist_s1024.npy'))
    sensitivities = [[] for i in range(len(models))]
    
    for i,batch_size in enumerate(batch_sizes):
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data', train=True, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
           ])),
           batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=batch_size, shuffle=False)
        sensitive(models[i], test_loader)

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ls1 = ax2.plot([i*64 for i in range(len(models))], [np.mean(s) for s in sensitivities], 'r-', label='sensitivity')
    ls2 = ax1.plot([i*64 for i in range(len(models))], [hist[0] for hist in hists], 'b-', label='train_acc')
    ls3 = ax1.plot([i*64 for i in range(len(models))], [hist[1] for hist in hists], 'b--', label='test_acc')
    ax1.set_ylabel('accuracy')
    ax1.set_xlabel('batch size')
    ax2.set_ylabel('sensitivity')
    ax1.tick_params(axis='y', color='blue')
    ax2.tick_params(axis='y', color='red')
    ax2.yaxis.label.set_color('red')
    ax1.yaxis.label.set_color('blue')
    ax1.set_title('Accuracy and sensitivity by training batch size')
    lbs = [l.get_label() for l in ls1+ls2+ls3]
    ax1.legend(ls1+ls2+ls3, lbs, loc='upper left', bbox_to_anchor=(1.08,1))
    fig.savefig('part2-acc_sens.png', bbox_inches='tight')
    plt.close()
    
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ls1 = ax2.plot([i*64 for i in range(len(models))], [np.mean(s) for s in sensitivities], 'r-', label='sensitivity')
    ls2 = ax1.plot([i*64 for i in range(len(models))], [hist[2] for hist in hists], 'b-', label='train_loss')
    ls3 = ax1.plot([i*64 for i in range(len(models))], [hist[3] for hist in hists], 'b--', label='test_loss')
    ax1.set_ylabel('loss')
    ax1.set_xlabel('batch size')
    ax2.set_ylabel('sensitivity')
    ax1.tick_params(axis='y', color='blue')
    ax2.tick_params(axis='y', color='red')
    ax2.yaxis.label.set_color('red')
    ax1.yaxis.label.set_color('blue')
    lbs = [l.get_label() for l in ls1+ls2+ls3]
    ax1.legend(ls1+ls2+ls3, lbs, loc='upper left', bbox_to_anchor=(1.08,1))
    ax1.set_title('Loss and sensitivity by training batch size')
    fig.savefig('part2-loss_sens.png', bbox_inches='tight')
    plt.close()