import numpy as np
from numpy.linalg import eigvals
import os, random, math
# torch library
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, grad
#matplot lib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sampleSize = 1000
epoch = 100000
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

def pnorm(model, loss):
    loss_grad = grad(loss, model.parameters(), create_graph=True)
    return sum([lg.norm() ** 2 for lg in loss_grad]) ** 0.5

def hessian(model, loss):
    loss_grad = grad(loss, model.parameters(), create_graph=True)
    var_len = sum([i.view(-1).shape[0] for i in loss_grad])
    grad_mat = np.zeros((var_len, var_len))
    i = 0
    for _lg in loss_grad:
        gs = _lg.view(-1)
        for g in gs:
            sec_grad = grad(g, model.parameters(), create_graph=True)
            j = 0
            for _sg in sec_grad:
                sgs = _sg.contiguous().view(-1)
                for sg in sgs:
                    grad_mat[i, j] = sg
                    grad_mat[j, i] = sg
                    j += 1
            i += 1
    return grad_mat

def p_eig_percentage(mat):
    _e = eigvals(mat)
    c = 0
    for e in _e:
        if e > 0.0:
            c += 1
    return c / len(_e)

def train(model, epoch, x, y, lossFunction, is_grad_zero):
    output = model(x)
    loss = lossFunction(output, y)
    g_norm = pnorm(model, loss)
    if i%1000 == 0:
        print("Epoch: {}, Loss:{}".format(i, loss.data[0]))
    model.optimizer.zero_grad()
    p = -1
    if not is_grad_zero:
        loss.backward()
    else:
        if g_norm.data.cpu().numpy()[0] < 0.003:
            p = p_eig_percentage(hessian(model, loss))
        g_norm.backward()
    model.optimizer.step()
    return loss.data[0], p, g_norm.data.cpu().numpy()[0]

# Transform data to tesor format
_x = Variable(torch.FloatTensor(x).view(sampleSize,1))
_y = Variable(torch.FloatTensor(y).view(sampleSize,1))

fig = plt.figure()
fig.suptitle('Loss-Minimal Ratio', fontsize=14)
plt.xlabel('Minimal Ratio', fontsize=10)
plt.ylabel('Loss', fontsize=10)

px = []
py = []
for t in range(100):
    print('==========Start Turn ' + str(t) + '==========')
    model = Model()
    if torch.cuda.is_available():
        _x, _y = _x.cuda(), _y.cuda()
        model = model.cuda()

    lossF = nn.MSELoss()
    is_grad_zero = False
    countdown = 1000
    for i in range(1, 100000):
        loss, p, g_norm = train(model, i, _x, _y, lossF, is_grad_zero)
        is_grad_zero = is_grad_zero or g_norm < 0.0025
        if p > 0:
            print('plt')
            px.append(p)
            py.append(loss)
        if is_grad_zero:
            countdown -= 1
        if countdown == 0:
            print('break at epoch ' + str(i))
            break

plt.scatter(px, py)
fig = plt.gcf()
fig.savefig('Loss-Minimal Ratio.png', dpi=200)
