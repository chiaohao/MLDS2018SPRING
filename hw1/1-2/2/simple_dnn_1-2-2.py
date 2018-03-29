import numpy as np
import os, random, math
# torch library
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

sampleSize = 1000
epoch = 10000
def generate_data(size):
    # e^(sin(pix)+cos(3pix))/2 + sin(2x)
    def simulate_function(_input):
        return math.exp(math.sin(math.pi*_input)+math.cos(3*math.pi*_input))/2 + math.sin(2*_input)
    x = sorted([random.random()*4 for i in range(size)])
    y = [simulate_function(e) for e in x]
    return x,y
                            
x, y = generate_data(sampleSize)
# simulated function
print("Simulated F ,input random sampled in [0,4]")
#plt.plot(x,y)
#plt.show()

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.name = "Dense 5 Layer"
        self.fcn1 = nn.Linear(1, 10)
        self.fcn2 = nn.Linear(10, 10)
        self.fcn3 = nn.Linear(10, 1)
        self.optimizer = torch.optim.Adam(self.parameters())
        
    def forward(self, _x):
        _x = F.sigmoid(self.fcn1(_x))
        _x = F.sigmoid(self.fcn2(_x))
        return self.fcn3(_x)

def train(model, epoch, x, y, lossFunction):
    output = model(x)
    loss = lossFunction(output, y)
    if i%1000 == 0:
        print("Epoch: {}, Loss:{}".format(i, loss.data[0]))
    model.optimizer.zero_grad()
    loss.backward()
    model.optimizer.step()
    return loss.data[0]

def pnorm(model):
    grad_all = 0.0
    for p in model.parameters():
        grad = 0.0
        if p.grad is not None:
            grad = (p.grad.cpu().data.numpy() ** 2).sum()
        grad_all += grad
    grad_norm = grad_all ** 0.5
    return grad_norm

# Transform data to tesor format
_x = Variable(torch.FloatTensor(x).view(sampleSize,1))
_y = Variable(torch.FloatTensor(y).view(sampleSize,1))

model = Model()
if torch.cuda.is_available():
    _x, _y = _x.cuda(), _y.cuda()
    model = model.cuda()

t_loss = []
g_norm = []

lossF = nn.MSELoss()
for i in range(1, epoch+1):
    lo = train(model, i, _x, _y, lossF)
    t_loss.append(lo)
    g_norm.append(pnorm(model))

t_loss = np.array(t_loss)
print(t_loss.shape)
np.save('loss_simple.npy', t_loss)

g_norm = np.array(g_norm)
print(g_norm.shape)
np.save('g_norm_simple.npy', g_norm)
