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
#from skimage.transform import resize

import os
import sys
import matplotlib.pyplot as plt

use_cuda = torch.cuda.is_available()

import random
import sentences_process

NOISE_SIZE = 100
generator_path = sys.argv[1]
noise_path = sys.argv[2]
dict_hair_path = sys.argv[3]
dict_eyes_path = sys.argv[4]
testing_path = sys.argv[5]
output_path = sys.argv[6]

wd_hair = sentences_process.Word_dict(True)
wd_eyes = sentences_process.Word_dict(True)
wd_hair.load_dict(dict_hair_path)
wd_eyes.load_dict(dict_eyes_path)

def save_imgs(generator, _testing_tags):
    r, c = 5, 5
    noise = Variable(torch.from_numpy(np.load(noise_path))).float()
    tags = Variable(torch.from_numpy(_testing_tags))
    hair = tags[:, 0]
    eyes = tags[:, 1]
    if use_cuda:
        noise = noise.cuda()
        hair = hair.cuda()
        eyes = eyes.cuda()
    # gen_imgs should be shape (25, 64, 64, 3)
    gen_imgs = generator(noise, hair, eyes).transpose(1, 2).transpose(2, 3)
    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow((gen_imgs[cnt].cpu().data.numpy() + 1.0) / 2.0)
            axs[i,j].axis('off')
            cnt += 1
    fig.savefig(output_path)
    plt.close()

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.embed_hair = nn.Embedding(len(wd_hair.w2n), 5)
        #self.gru_hair = nn.GRU(EMBED_DIM, GRU_HIDDEN_SIZE, batch_first=True)
        self.embed_eyes = nn.Embedding(len(wd_eyes.w2n), 5)
        #self.gru_eyes = nn.GRU(EMBED_DIM, GRU_HIDDEN_SIZE, batch_first=True)

        self.prelinear = nn.Linear(NOISE_SIZE + 5 * 2, 8192)
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

    def forward(self, input, hair, eyes):
        hair_output = self.embed_hair(hair).view(hair.size()[0], -1)
        eyes_output = self.embed_eyes(eyes).view(eyes.size()[0], -1)
        output = torch.cat((input, hair_output), 1)
        output = torch.cat((output, eyes_output), 1)

        output = self.preselu(self.prelinear(output))
        output = output.view(input.size(0), 512, 4, 4)
        
        output = self.selu1(self.batch1(self.conv1(output)))
        output = self.selu2(self.batch2(self.conv2(output)))
        output = self.selu3(self.batch3(self.conv3(output)))
        output = self.out(self.conv4(output))
        return output

testing_pre_tags = [l.replace('\n', '').split(',') for l in open(testing_path, 'r').readlines()]
testing_tags = [[t[0], t[1].split(' ')] for t in testing_pre_tags]
testing_tags = [[t[1][0], t[1][2]] for t in testing_tags]
testing_tags = np.array([np.concatenate((wd_hair.sentence2number(t[0], 1), wd_eyes.sentence2number(t[1], 1))) for t in testing_tags])

generator = Generator()
generator = torch.load(generator_path)
if use_cuda:
    generator = generator.cuda()

save_imgs(generator, testing_tags)
