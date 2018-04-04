import os
import urllib.request
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28*28, 256)
        self.fc2 = nn.Linear(256, 10)
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)   
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        print('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}\r'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.data[0]), end='')
    if epoch % 10 == 0:
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
        print('\nTraining Acc: {}, Training Loss: {}'.format(correct/len(train_loader.dataset), train_loss))
        if epoch == max_epoch:
            train_accs.append(correct / len(train_loader.dataset))
            train_losss.append(train_loss)
    else:
        print('')

def val(epoch):
    model.eval()
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
        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\r'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)), end='')
    if epoch == max_epoch:
        test_accs.append(correct / len(test_loader.dataset))
        test_losss.append(test_loss)
    print('')
    model_save(model, epoch, max_epoch, 100. * correct / len(test_loader.dataset), test_loss)

def model_save(model, epoch, interval, acc, loss):
    if epoch % interval == 0:
        torch.save(model, './mnist_s{}_e{}_a{:.2f}_l{:.4f}.pt'.format(batch_size, epoch, acc, loss))
        np.save('./mnist_hist_s{}'.format(batch_size), np.array([train_accs, test_accs, train_losss, test_losss]))
        print('Model saved')

def count_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params

batch_sizes = [64, 128, 256, 512, 1024]
max_epoch = 200

if __name__ == "__main__":
    for batch_size in batch_sizes:
        train_accs = []
        test_accs = []
        train_losss = []
        test_losss = []

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

    
        model = Net()
        print(count_parameters(model))
        if torch.cuda.is_available():
            model.cuda()

        optimizer = optim.Adagrad(model.parameters())

        for e in range(1, max_epoch + 1):
            train(e)
            val(e)
