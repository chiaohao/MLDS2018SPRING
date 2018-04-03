import os, math
import urllib.request
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

batch_size = 256
epoch = 1
max_epoch =  epoch
num_models = 40

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

class Net(nn.Module):
    def __init__(self, s):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28*28, s)
        self.fc2 = nn.Linear(s, 10)
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

model_params = []
train_accs = [[] for i in range(num_models)]
train_losss = [[] for i in range(num_models)]
test_accs = [[] for i in range(num_models)]
test_losss = [[] for i in range(num_models)]

def shuffle_target(data_loader, is_shuffle=True):
    np.random.seed(0)
    shuffled_data_loader = []
    for data, target in data_loader:
        target_ = target.numpy()
        if is_shuffle:
            np.random.shuffle(target_)
        target_ = torch.LongTensor(target_)
        shuffled_data_loader.append((data, target_))
    return shuffled_data_loader

def data_len(data_loader):
    s = 0
    for data, target in data_loader:
        s += len(data)
    return s

def train(epoch, r):
    model.train()
    data_size = data_len(train_loader)
    for batch_idx, (data, target) in enumerate(train_loader):
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)

        output = model(data)
        loss = F.nll_loss(output, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}\r'.format(
            epoch, batch_idx * len(data), data_size,
            100. * batch_idx / len(train_loader), loss.data[0]), end='')


    if epoch == max_epoch:
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
        train_accs[r].append(correct / data_size)
        train_losss[r].append(train_loss)
    print('')

def val(epoch, r):
    model.eval()
    test_loss = 0
    correct = 0
    data_size = data_len(test_loader)
    for data, target in test_loader:
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0]/data_size
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()
        print('Test set: loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\r'.format(
            test_loss, correct, data_size,
            100. * correct / data_size), end='')
    if epoch == max_epoch:
        test_accs[r].append(correct / data_size)
        test_losss[r].append(test_loss)
        print(correct / data_size)
    print('')

def count_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params

train_loader = shuffle_target(train_loader, False)
test_loader = shuffle_target(test_loader, False)

for i in range(num_models):
    hidden_size = [int(math.pow(1.2,i))  for v in range(1,num_models+1)]
    model = Net(hidden_size[i])
    print("Model {} paramters: {}".format(i, count_parameters(model)))
    model_params.append(count_parameters(model))
    if torch.cuda.is_available():
        model.cuda()
    
    optimizer = optim.Adagrad(model.parameters())

    for e in range(1, epoch + 1):
        train(e, i)
        val(e, i)

params = np.array(model_params)
accs = np.array([train_accs, test_accs])
losss = np.array([train_losss, test_losss])
print(accs.shape)
print(losss.shape)
np.save('mnist_gen_accs.npy', accs)
np.save('mnist_gen_losss.npy', losss)
np.save('mnist_gen_params.npy', params)