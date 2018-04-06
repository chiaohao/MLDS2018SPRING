import numpy as np
import os, random, math
import matplotlib.pyplot as plt
# torch library
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

sampleSize = 140
epoch = 12000
def generate_data(size, simulate=1):
    # e^(sin(3x)+cos(2pix))/2pi + sin(3pi)
    def simulate_function1(_input):
        return math.exp(math.sin(3*_input)+math.cos(2*math.pi*_input))/(2*math.pi) + math.sin(3*math.pi)
    # sin(3x)+cos(2pix) + e^(sin(2x))
    def simulate_function2(_input):
        return math.sin(3*_input)+math.cos(2*math.pi*_input) + math.exp(math.sin(2*_input))
    x = sorted([random.random()*4 for i in range(size)])
    if simulate == 1:
        y = [simulate_function1(e) for e in x]
    elif simulate == 2:
        y = [simulate_function2(e) for e in x]
    return x, y

# Dense model with 2 layer
class Model_0(nn.Module):
    def __init__(self):
        super(Model_0, self).__init__()
        self.name = "Shallow"
        self.fcn1 = nn.Linear(1, 175)
        self.fcn2 = nn.Linear(175, 1)
        self.optimizer = torch.optim.Adam(self.parameters())
    def forward(self, _x):
        _x = F.sigmoid(self.fcn1(_x))
        return self.fcn2(_x)
    def params(self):
        params = filter(lambda p: p.requires_grad, self.parameters())
        return sum([np.prod(p.size()) for p in params])

# Dense model with 3 layer
class Model_1(nn.Module):
    def __init__(self):
        super(Model_1, self).__init__()
        self.name = "Dense 3 Layer"
        self.fcn1 = nn.Linear(1, 21)
        self.fcn2 = nn.Linear(21, 21)
        self.fcn3 = nn.Linear(21, 1)
        self.optimizer = torch.optim.Adam(self.parameters())
    def forward(self, _x):
        _x = F.sigmoid(self.fcn1(_x))
        _x = F.sigmoid(self.fcn2(_x))
        return self.fcn3(_x)
    def params(self):
        params = filter(lambda p: p.requires_grad, self.parameters())
        return sum([np.prod(p.size()) for p in params])
    
# Dense model with 6 layer
class Model_2(nn.Module):
    def __init__(self):
        super(Model_2, self).__init__()
        self.name = "Dense 6 Layer"
        self.fcn1 = nn.Linear(1, 11)
        self.fcn2 = nn.Linear(11, 11)
        self.fcn3 = nn.Linear(11, 11)
        self.fcn4 = nn.Linear(11, 10)
        self.fcn5 = nn.Linear(10, 10)
        self.fcn6 = nn.Linear(10, 1)
        self.optimizer = torch.optim.Adam(self.parameters())
    def forward(self, _x):
        _x = F.sigmoid(self.fcn1(_x))
        _x = F.sigmoid(self.fcn2(_x))
        _x = F.sigmoid(self.fcn3(_x))
        _x = F.sigmoid(self.fcn4(_x))
        _x = F.sigmoid(self.fcn5(_x))
        return self.fcn6(_x)
    def params(self):
        params = filter(lambda p: p.requires_grad, self.parameters())
        return sum([np.prod(p.size()) for p in params])

def train(model, epoch, x, y, lossFunction):
    output = model(x)
    loss = lossFunction(output, y)
    if epoch%2000 == 0:
        print("Epoch: {}, Loss:{}".format(epoch, loss.data[0]))
    model.optimizer.zero_grad()
    loss.backward()
    model.optimizer.step()
    return loss.data[0]

for simulate_index in range(1,3):
    # Transform data to tesor format
    x, y = generate_data(sampleSize, simulate=simulate_index) # decide what simulate function
    _x = Variable(torch.FloatTensor(x).view(sampleSize,1))
    _y = Variable(torch.FloatTensor(y).view(sampleSize,1))

    # Instantiate model and 
    model_0 = Model_0()
    model_1 = Model_1()
    model_2 = Model_2()

    # Move model and datas to gpu if possible
    if torch.cuda.is_available():
        _x, _y = _x.cuda(), _y.cuda()
        model_0, model_1, model_2 = model_0.cuda(), model_1.cuda(), model_2.cuda()
    models = [model_0, model_1, model_2]
    calcuate_MSE = nn.MSELoss()
    losses = []
    # Train
    for model in models:
        print("Model: {} Start Training".format(model.name))
        modelLoss = []
        for i in range(1, epoch+1):
            if i >= 2000:
                modelLoss.append(train(model, i, _x, _y, calcuate_MSE))
            else:
                train(model, i, _x, _y, calcuate_MSE)
        losses.append(modelLoss)

    for index, model in enumerate(models):
        plt.plot(range(2000, epoch+1), losses[index], label=model.name + ' parameters: {}'.format(model.params()))
    plt.legend(loc='best')
    plt.gcf().savefig('loss_{}.png'.format(simulate_index), dpi=200)
    plt.clf()
    
    # Predict
    for model in models:
        output = model(_x).cpu()
        plt.plot(x, output.data.numpy().reshape(sampleSize), label=model.name + ' parameters: {}'.format(model.params()))
    x, y = generate_data(1000, simulate=simulate_index)
    plt.plot(x, y, label='Data set')
    plt.legend(loc='best')
    plt.gcf().savefig('func_curve_{}.png'.format(simulate_index), dpi=200)
    plt.clf()
