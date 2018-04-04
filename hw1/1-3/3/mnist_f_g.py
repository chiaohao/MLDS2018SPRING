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
        self.fc1 = nn.Linear(28*28, 50)
        self.fc2 = nn.Linear(50, 10)
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
            data, target = Variable(data, volatile=True), Variable(target)
            output = model(data)
            train_loss += F.nll_loss(output, target, size_average=False).data[0]/len(train_loader.dataset)
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()
        print('\nTraining Acc: {}, Training Loss: {}'.format(correct/len(train_loader.dataset), train_loss))
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
    print('')
    model_save(model, epoch, 10, 100. * correct / len(test_loader.dataset), test_loss)

def model_save(model, epoch, interval, acc, loss):
    if epoch % interval == 0 and epoch > 200:
        torch.save(model, './mnist_s{}_e{}_a{:.2f}_l{:.4f}.pt'.format(batch_size, epoch, acc, loss))
        print('Model saved')


batch_size = 1024
epoch = 1000

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

    model = Net()
    if torch.cuda.is_available():
        model.cuda()

    optimizer = optim.Adagrad(model.parameters())

    for e in range(1, epoch + 1):
        train(e)
        val(e)

# b_s 64 e 200 train_l 0.0266
# b_s 1024 220