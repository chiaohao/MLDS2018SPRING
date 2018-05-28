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

import sys
import matplotlib.pyplot as plt

use_cuda = torch.cuda.is_available()

import random

NOISE_SIZE = 100

generator_path = sys.argv[1]
noise_path = sys.argv[2]
save_image_path = sys.argv[3]

def save_imgs(generator):
    r, c = 5, 5
    noise = Variable(torch.from_numpy(np.load(noise_path))).float()
    if use_cuda:
        noise = noise.cuda()
    # gen_imgs should be shape (25, 64, 64, 3)
    gen_imgs = generator(noise).transpose(1, 2).transpose(2, 3)
    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow((gen_imgs[cnt].cpu().data.numpy() + 1.0) / 2.0)
            axs[i,j].axis('off')
            cnt += 1
    fig.savefig(save_image_path)
    plt.close()

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
        self.out = nn.Tanh()

    def forward(self, input):
        output = self.preselu(self.prelinear(input))
        output = output.view(input.size(0), 512, 4, 4)
        output = self.selu1(self.batch1(self.conv1(output)))
        output = self.selu2(self.batch2(self.conv2(output)))
        output = self.selu3(self.batch3(self.conv3(output)))
        output = self.out(self.conv4(output))
        return output

generator = Generator()
generator = torch.load(generator_path)
if use_cuda:
    generator = generator.cuda()

save_imgs(generator)
