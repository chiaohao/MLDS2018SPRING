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
epoch = 200
max_epoch =  epoch

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
        self.fc2 = nn.Linear(s, 2*s)
        self.fc3 = nn.Linear(2*s, 4*s)
        self.fc4 = nn.Linear(4*s, 2*s)
        self.fc5 = nn.Linear(2*s, s)
        self.fc6 = nn.Linear(s, 10)
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.fc6(x)
        return F.log_softmax(x, dim=1)

train_accs = []
train_losss = []
test_accs = []
test_losss = []

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

def train(epoch):
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
    train_accs.append(correct / data_size)
    if epoch % 10 == 0:
        print("Epoch {} - Training Acc: {}".format(epoch, correct/data_size))
    train_losss.append(train_loss)
    print('')

def val(epoch):
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

    test_accs.append(correct / data_size)
    test_losss.append(test_loss)
    print('')

def count_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params

train_loader = shuffle_target(train_loader, True)
test_loader = shuffle_target(test_loader, True)


model = Net(512)
print("Model paramters: {}".format(count_parameters(model)))
if torch.cuda.is_available():
    model.cuda()
    
optimizer = optim.Adagrad(model.parameters())

for e in range(1, epoch + 1):
    train(e)
    val(e)

accs = np.array([train_accs, test_accs])
losss = np.array([train_losss, test_losss])
print("Model Parameters: {}".format(count_parameters(model)))
print("Training Acc: {} | Testing Acc: {}".format(accs[0][len(accs[0])-1],accs[1][len(accs[1])-1]))
np.save('mnist_rand_accs.npy', accs)
np.save('mnist_rand_losss.npy', losss)