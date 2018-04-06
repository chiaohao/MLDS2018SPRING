import numpy as np
import matplotlib.pyplot as plt
import os
import urllib.request
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

batch_size = 16384
epoch = 80

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=True, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
   ])),
   batch_size=batch_size, shuffle=True)

# Dense model with 2 layer
class Model_0(nn.Module):
    def __init__(self):
        super(Model_0, self).__init__()
        self.name = "Shallow"
        self.fcn1 = nn.Linear(28*28, 26)
        self.fcn2 = nn.Linear(26, 10)
    def forward(self, _x):
        _x = _x.view(-1, 28*28)
        _x = F.relu(self.fcn1(_x))
        _x = self.fcn2(_x)
        return F.log_softmax(_x, dim=1)
    def params(self):
        params = filter(lambda p: p.requires_grad, self.parameters())
        return sum([np.prod(p.size()) for p in params])

# Dense model with 3 layer
class Model_1(nn.Module):
    def __init__(self):
        super(Model_1, self).__init__()
        self.name = "Dense 3 Layer"
        self.fcn1 = nn.Linear(28*28, 25)
        self.fcn2 = nn.Linear(25, 29)
        self.fcn3 = nn.Linear(29, 10)
    def forward(self, _x):
        _x = _x.view(-1, 28*28)
        _x = F.relu(self.fcn1(_x))
        _x = F.relu(self.fcn2(_x))
        _x = self.fcn3(_x)
        return F.log_softmax(_x, dim=1)
    def params(self):
        params = filter(lambda p: p.requires_grad, self.parameters())
        return sum([np.prod(p.size()) for p in params])
    
# Dense model with 4 layer
class Model_2(nn.Module):
    def __init__(self):
        super(Model_2, self).__init__()
        self.name = "Dense 4 Layer"
        self.fcn1 = nn.Linear(28*28, 24)
        self.fcn2 = nn.Linear(24, 28)
        self.fcn3 = nn.Linear(28, 29)
        self.fcn4 = nn.Linear(29, 10)
    def forward(self, _x):
        _x = _x.view(-1, 28*28)
        _x = F.relu(self.fcn1(_x))
        _x = F.relu(self.fcn2(_x))
        _x = F.relu(self.fcn3(_x))
        _x = self.fcn4(_x)
        return F.log_softmax(_x, dim=1)
    def params(self):
        params = filter(lambda p: p.requires_grad, self.parameters())
        return sum([np.prod(p.size()) for p in params])

# Instantiate model
model_0 = Model_0()
model_1 = Model_1()
model_2 = Model_2()

# Move model and datas to gpu if possible
if torch.cuda.is_available():
    model_0, model_1, model_2 = model_0.cuda(), model_1.cuda(), model_2.cuda()
    model_0.optimizer = optim.Adagrad(model_0.parameters())
    model_1.optimizer = optim.Adagrad(model_1.parameters())
    model_2.optimizer = optim.Adagrad(model_2.parameters())
models = [model_0, model_1, model_2]

def data_len(data_loader):
    s = 0
    for data, target in data_loader:
        s += len(data)
    return s

def train(_model, epoch):
    _model.train()
    data_size = data_len(train_loader)
    for batch_idx, (data, target) in enumerate(train_loader):
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        output = _model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        _model.optimizer.step()
        print('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}\r'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.data[0]), end='')
    print('')
    train_loss = 0
    correct = 0
    for data, target in train_loader:
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        train_loss += F.nll_loss(output, target, size_average=False).data[0]/data_size
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()
    return correct/data_size, train_loss

losses = []
accuracies = []
for model in models:
    model_loss = []
    model_accuracy = []
    for e in range(1, epoch+1):
        accuracy, loss = train(model, e)
        model_loss.append(loss)
        model_accuracy.append(accuracy)
    losses.append(model_loss)
    accuracies.append(model_accuracy)

# plot loss and accuracy chart.
for index, model in enumerate(models):
    plt.plot(range(1, epoch+1), losses[index], label=model.name + ' parameters: {}'.format(model.params()))
plt.legend(loc='best')
plt.gcf().savefig('loss.png', dpi=200)
plt.clf()
for index, model in enumerate(models):
    plt.plot(range(1, epoch+1), accuracies[index], label=model.name + ' parameters: {}'.format(model.params()))
plt.legend(loc='best')
plt.gcf().savefig('accuracy.png', dpi=200)
plt.clf()

