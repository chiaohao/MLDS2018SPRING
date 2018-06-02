import matplotlib
matplotlib.use('Agg')
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import torch.utils.data as Data
import numpy as np
from skimage import io as skio
from skimage.transform import resize

import os
import matplotlib.pyplot as plt

use_cuda = torch.cuda.is_available()

import time
import random

from pprint import pprint

EPOCHS = 1000
BATCH_SIZE = 256
NOISE_SIZE = 100
LEARNING_RATE = 0.0002
LEARNING_RATE_BETA = (0.5, 0.999)
USE_EXTRA_DATA = True
#USE_ADDITIVE_NOISE = True

def save_imgs(generator, _iter):
    r, c = 5, 5
    noise = Variable(torch.from_numpy(np.random.normal(0, 1, (r * c, NOISE_SIZE)))).float()
    if use_cuda:
        noise = noise.cuda()
    # gen_imgs should be shape (25, 64, 64, 3)
    gen_imgs = generator(noise).transpose(1, 2).transpose(2, 3)
    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow((gen_imgs[cnt].cpu().data.numpy()/255).clip(0,1))
            #axs[i,j].imshow((gen_imgs[cnt].cpu().data.numpy()+1)/2)
            axs[i,j].axis('off')
            cnt += 1
    fig.savefig("output_imgs_nn/output_%d.png" % _iter)
    plt.close()

def get_train_loader(images_path, ex_images_path=''):
    print('Start pack to DataLoader... ', end='')
    xs = np.array([resize(skio.imread(images_path + '/' + image_name), (64, 64)) for image_name in os.listdir(images_path)], dtype=np.float32)
    if USE_EXTRA_DATA:
        xs_ = np.array([resize(skio.imread(ex_images_path + '/' + image_name), (64, 64)) for image_name in os.listdir(ex_images_path)], dtype=np.float32)
        xs = np.concatenate((xs, xs_), axis=0)
    #rescale xs to 0-255
    xs = torch.from_numpy(xs*255).transpose(2, 3).transpose(1, 2)
    ys = torch.ones(xs.size()[0])
    #soft label
    #ys = torch.from_numpy(np.random.uniform(low=0.7, high=1.2, size=xs.size()[0]))
    
    torch_dataset = Data.TensorDataset(data_tensor=xs, target_tensor=ys)
    train_loader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2
    )
    print('Done!')
    return train_loader

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.prelinear = nn.Linear(NOISE_SIZE, 8192)
        self.preselu = nn.SELU(True)
        self.conv1 = nn.ConvTranspose2d(512, 256, 4, 2, 1, bias = False)
        self.batch1 = nn.BatchNorm2d(256)
        self.selu1 = nn.SELU(True)
        self.conv2 = nn.ConvTranspose2d(256, 128, 4, 2, 1, bias = False)
        self.batch2 = nn.BatchNorm2d(128)
        self.selu2 = nn.SELU(True)
        self.conv3 = nn.ConvTranspose2d(128, 64, 4, 2, 1, bias = False)
        self.batch3 = nn.BatchNorm2d(64)
        self.selu3 = nn.SELU(True)
        self.conv4 = nn.ConvTranspose2d(64, 3, 4, 2, 1, bias = False)
        self.out = nn.ReLU()

    def forward(self, input):
        output = self.preselu(self.prelinear(input))
        output = output.view(input.size(0), 512, 4, 4)
        output = self.selu1((self.batch1(self.conv1(output))))
        output = self.selu2((self.batch2(self.conv2(output))))
        output = self.selu3((self.batch3(self.conv3(output))))
        output = self.out(self.conv4(output))
        return output

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 4, 2, 1, bias = False)
        self.batch1 = nn.BatchNorm2d(32)
        self.selu1 = nn.SELU(True)
        self.conv2 = nn.Conv2d(32, 64, 4, 2, 1, bias = False)
        self.batch2 = nn.BatchNorm2d(64)
        self.selu2 = nn.SELU(True)
        self.conv3 = nn.Conv2d(64, 128, 4, 2, 1, bias = False)
        self.batch3 = nn.BatchNorm2d(128)
        self.selu3 = nn.SELU(True)
        self.conv4 = nn.Conv2d(128, 256, 4, 2, 1, bias = False)
        self.batch4 = nn.BatchNorm2d(256)
        self.selu4 = nn.SELU(True)
        self.linear = nn.Linear(4096, 1)
        self.out = nn.Sigmoid()

    def forward(self, input):
        output = self.selu1((self.batch1(self.conv1(input))))
        output = self.selu2((self.batch2(self.conv2(output))))
        output = self.selu3((self.batch3(self.conv3(output))))
        output = self.selu4((self.batch4(self.conv4(output))))
        output = self.linear(output.view(input.size()[0], -1))
        output = self.out(output)
        return output.view(-1)

def train(_iter, _generator, _discriminator, _optimG, _optimD, _criterion, _train_loader):
    prev_errD = 0.0
    prev_errG = 0.0
    errD_all = Variable(torch.zeros(1))
    errG_all = Variable(torch.zeros(1))
    for idx, (_xs, _ys) in enumerate(_train_loader):
        # Train Discriminator
        _discriminator.zero_grad()
        reals = Variable(_xs)
        labels = Variable(_ys).float()
        #additive_noise = Variable(torch.FloatTensor(_xs.size()[0], 3, 96, 96))

        if use_cuda:
            reals = reals.cuda()
            labels = labels.cuda()
            #additive_noise = additive_noise.cuda()
        #if USE_ADDITIVE_NOISE:
        #    additive_noise.data.resize_(reals.size()).normal_(0, 0.005)
        #    reals.data.add_(additive_noise.data)
        output = _discriminator(reals)
        errD_real = _criterion(output, labels)
    
        noise = torch.from_numpy(np.random.normal(0, 1, (_xs.size()[0], NOISE_SIZE)))
        #noise = torch.randn(_xs.size()[0], NOISE_SIZE)
        noise = Variable(noise).float()
        fakes_labels = Variable(torch.zeros(_xs.size()[0]))
        #soft fake labels
        #fakes_labels = Variable(torch.from_numpy(np.random.uniform(low=0, high=0.3, size=_xs.size()[0]))).float()
        if use_cuda:
            noise = noise.cuda()
            fakes_labels = fakes_labels.cuda()
        fakes = _generator(noise)
        output = _discriminator(fakes.detach())
        errD_fake = _criterion(output, fakes_labels)

        errD_all = errD_real + errD_fake
        #if prev_errG > prev_errD or prev_errD > 1.0:
        errD_all.backward()
        _optimD.step()
        prev_errD = errD_all.data[0]
    
        # Train Generator
        _generator.zero_grad()
        fakes_labels = Variable(torch.ones(_xs.size()[0]))
        if use_cuda:
            fakes_labels = fakes_labels.cuda()
        fakes = _generator(noise)
        output = _discriminator(fakes)
        errG = _criterion(output, fakes_labels)
        errG_all = errG
        #if prev_errD > prev_errG or prev_errG > 1.0:
        errG_all.backward()
        _optimG.step()
        prev_errG = errG_all.data[0]
    
    return errD_all.data[0], errG_all.data[0]

def trainIters(_generator, _discriminator, _optimG, _optimD, _criterion, _train_loader, print_every=1):
    print('Start training...')
    start = time.time()
    
    for iter in range(1, EPOCHS + 1):
        lossD, lossG = train(iter, _generator, _discriminator, _optimG, _optimD, _criterion, _train_loader)
        
        if iter % print_every == 0:
            avg_epoch_time = (time.time() - start) / print_every
            start = time.time()
            print('Epoch: %d, Avg_Epoch_Time: %.4f, LossD: %.4f, LossG: %.4f' % (iter, avg_epoch_time, lossD, lossG))
        if iter % 50 == 0:
            save_imgs(_generator, iter)
            torch.save(generator, 'generator.pt')
            torch.save(discriminator, 'discriminator.pt')
        
train_loader = get_train_loader('AnimeDataset/faces/', 'AnimeDataset/extra_data/images/')
generator = Generator()
generator.apply(weights_init)
discriminator = Discriminator()
discriminator.apply(weights_init)

if use_cuda:
    generator = generator.cuda()
    discriminator = discriminator.cuda()

criterion = nn.BCELoss()
optimizerG = optim.Adam(generator.parameters(), lr = LEARNING_RATE, betas = LEARNING_RATE_BETA)
optimizerD = optim.Adam(discriminator.parameters(), lr = LEARNING_RATE, betas = LEARNING_RATE_BETA)

trainIters(generator, discriminator, optimizerG, optimizerD, criterion, train_loader)

