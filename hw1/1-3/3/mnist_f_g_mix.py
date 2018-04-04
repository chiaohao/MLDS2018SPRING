import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from mnist_f_g import Net, batch_size
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def mix_model(alpha, model1, model2, model):
   model.fc1.weight.data =  (1-alpha)*model1.fc1.weight.data + alpha*model2.fc1.weight.data
   model.fc2.weight.data =  (1-alpha)*model1.fc2.weight.data + alpha*model2.fc2.weight.data

def train_acc_loss(model):
    train_loss = 0
    correct = 0
    for data, target in train_loader:
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        train_loss += F.nll_loss(output, target, size_average=False).data[0]/len(train_loader.dataset)
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()
    print('Training Acc: {}, Training Loss: {}'.format(correct/len(train_loader.dataset), train_loss))
    return correct/len(train_loader.dataset), train_loss

def test_acc_loss(model):
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0]/len(test_loader.dataset)
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()
    print('Testing Acc: {}, Testing Loss: {}'.format(correct/len(test_loader.dataset), test_loss))
    return correct/len(test_loader.dataset), test_loss

if __name__ == "__main__":
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
        batch_size=batch_size, shuffle=True)

    alphas = []
    for i in range(0,31):
        alphas.append((i-10)/10)
    sorted(alphas)

    model_64 = torch.load('./mnist_s64_e200_a97.21_l0.0966.pt')
    model_1024 = torch.load('./mnist_s1024_e250_a97.19_l0.0945.pt')
    model = Net()
    train_accs = []
    test_accs = []
    train_losss = []
    test_losss = []
    for alpha in alphas:
        mix_model(alpha, model_64, model_1024, model)
        if torch.cuda.is_available():
            model.cuda()
        train_acc, train_loss = train_acc_loss(model)
        test_acc, test_loss = test_acc_loss(model)
        train_accs.append(train_acc)
        train_losss.append(train_loss)
        test_accs.append(test_acc)
        test_losss.append(test_loss)

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ls1 = ax1.plot(alphas, train_losss, 'b-', label='train_loss')
    ls2 = ax1.plot(alphas, test_losss, 'b--', label='test_loss')
    ls3 = ax2.plot(alphas, train_accs, 'r-', label='train_acc')
    ls4 = ax2.plot(alphas, test_accs, 'r--', label='test_acc')
    labels = [l.get_label() for l in ls1+ls2+ls3+ls4]

    ax1.set_ylabel('loss')
    ax2.set_ylabel('accuracy')
    ax1.set_xlabel('alpha')
    ax1.yaxis.label.set_color('blue')
    ax2.yaxis.label.set_color('red')
    ax1.tick_params(axis='y', colors='blue')
    ax2.tick_params(axis='y', colors='red')
    ax1.legend(ls1+ls2+ls3+ls4, labels, loc='upper left', bbox_to_anchor=(1.08, 1))
    ax1.set_title('Interpolation of model weight')
    fig.savefig('part1.png', bbox_inches='tight')