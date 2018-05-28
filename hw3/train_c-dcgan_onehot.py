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

import re
import time
import random
import sentences_process
from pprint import pprint

EPOCHS = 300
BATCH_SIZE = 256
NOISE_SIZE = 100
LEARNING_RATE = 0.0002
LEARNING_RATE_BETA = (0.5, 0.999)
USE_EXTRA_DATA = True
#USE_ADDITIVE_NOISE = True
wd_hair = sentences_process.Word_dict(one_hot=True)
wd_eyes = sentences_process.Word_dict(one_hot=True)

def save_imgs(generator, _testing_sents, _iter):
    r, c = 5, 5
    noise = Variable(torch.from_numpy(np.random.normal(0, 1, (r * c, NOISE_SIZE)))).float()
    sents = Variable(torch.from_numpy(_testing_sents)).float()
    if use_cuda:
        noise = noise.cuda()
        sents = sents.cuda()
    # gen_imgs should be shape (25, 64, 64, 3)
    gen_imgs = generator(noise, sents).transpose(1, 2).transpose(2, 3)
    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow((gen_imgs[cnt].cpu().data.numpy() + 1.0) / 2.0)
            axs[i,j].axis('off')
            cnt += 1
    fig.savefig("output_imgs/output_%d.png" % _iter)
    plt.close()

def get_train_loader(images_path, tags_path, ex_images_path='', ex_tags_path=''):
    print('Start pack to DataLoader... ', end='')
    pre_tags = [re.split(',|\t', l.replace('\n', ' ')) for l in open(tags_path, 'r').readlines()]
    tags = [[i[0], [s.split(':')[0] for s in i if 'hair' in s and 'long' not in s and 'short' not in s], [s.split(':')[0] for s in i if 'eyes' in s]] for i in pre_tags]
    #tags = [i for i in tags if [] not in i]
    tags = [[i[0], i[1] if i[1] != '' else '{UNK}', i[2] if i[2] != '' else '{UNK}'] for i in tags]
    tags = [[i[0], ' '.join([j.split(' ')[0] for j in i[1]]), ' '.join([j.split(' ')[0] for j in i[2]])] for i in tags]

    hair_sents = [t[1] for t in tags]
    eyes_sents = [t[2] for t in tags]
    wd_hair.add_sentences(hair_sents)
    wd_eyes.add_sentences(eyes_sents)
    tmp_tags = [[t[0], np.concatenate((wd_hair.sentence2onehot(t[1]), wd_eyes.sentence2onehot(t[2])))] for t in tags]
    xs = np.array([resize(skio.imread(images_path + '/' + tag[0] + '.jpg') / 255.0, (64, 64)) for tag in tags], dtype=np.float32)
    ys = np.array([t[1] for t in tmp_tags])
    if USE_EXTRA_DATA:
        pre_tags_ = [re.split(',|\t', l.replace('\n', ' ')) for l in open(ex_tags_path, 'r').readlines()]
        tags_ = [[i[0], [s.split(':')[0] for s in i if 'hair' in s and 'long' not in s and 'short' not in s], [s.split(':')[0] for s in i if 'eyes' in s]] for i in pre_tags_]
        #tags_ = [i for i in tags_ if [] not in i]
        tags_ = [[i[0], i[1] if i[1] != '' else '{UNK}', i[2] if i[2] != '' else '{UNK}'] for i in tags_]
        tags_ = [[i[0], ' '.join([j.split(' ')[0] for j in i[1]]), ' '.join([j.split(' ')[0] for j in i[2]])] for i in tags_]
        hair_sents_ = [t[1] for t in tags_]
        eyes_sents_ = [t[2] for t in tags_]
        wd_hair.add_sentences(hair_sents_)
        wd_eyes.add_sentences(eyes_sents_)
        tags_ = [[t[0], np.concatenate((wd_hair.sentence2onehot(t[1]), wd_eyes.sentence2onehot(t[2])))] for t in tags_]
        xs_ = np.array([resize(skio.imread(ex_images_path + '/' + tag_[0] + '.jpg') / 255.0, (64, 64)) for tag_ in tags_], dtype=np.float32)
        ys_ = np.array([t[1] for t in tags_])
        xs = np.concatenate((xs, xs_), axis=0)
        # Dict updated, reprocess ys
        tmp_tags = [[t[0], np.concatenate((wd_hair.sentence2onehot(t[1]), wd_eyes.sentence2onehot(t[2])))] for t in tags]
        ys = np.array([t[1] for t in tmp_tags])
        ys = np.concatenate((ys, ys_), axis=0)

    xs = torch.from_numpy(xs * 2.0 - 1.0).transpose(2, 3).transpose(1, 2)
    ys = torch.from_numpy(ys)
    wd_hair.save_dict('dict_hair.txt') #len:16
    wd_eyes.save_dict('dict_eyes.txt') #len:16
    
    torch_dataset = Data.TensorDataset(data_tensor=xs, target_tensor=ys)
    train_loader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2
    )
    print('Done!')
    onehot_len = len(wd_hair.w2n) + len(wd_eyes.w2n)
    return train_loader, onehot_len

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class Generator(nn.Module):
    def __init__(self, onehot_len):
        super(Generator, self).__init__()
        self.sents_linear = nn.Linear(onehot_len, 32)
        self.sents_selu = nn.SELU(True)

        self.prelinear = nn.Linear(NOISE_SIZE, 8192)
        self.preselu = nn.SELU(True)

        self.conv1 = nn.ConvTranspose2d(512 + 2, 256, 4, 2, 1, bias = False)
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

    def forward(self, input, sents):
        sents_output = self.sents_selu(self.sents_linear(sents))
        sents_output = sents_output.view(input.size(0), 2, 4, 4)

        output = self.preselu(self.prelinear(input))
        output = output.view(input.size(0), 512, 4, 4)
        
        output = torch.cat((output, sents_output), 1)
        output = self.selu1(self.batch1(self.conv1(output)))
        output = self.selu2(self.batch2(self.conv2(output)))
        output = self.selu3(self.batch3(self.conv3(output)))
        output = self.out(self.conv4(output))
        return output

class Discriminator(nn.Module):
    def __init__(self, onehot_len):
        super(Discriminator, self).__init__()
        self.sents_linear = nn.Linear(onehot_len, 3 * 64 * 64)
        self.sents_selu1 = nn.SELU(True)
        self.sents_conv = nn.Conv2d(3, 1, 4, 2, 1, bias = False)
        self.sents_batch = nn.BatchNorm2d(1)
        self.sents_selu2 = nn.SELU(True)

        self.conv1 = nn.Conv2d(3, 32, 4, 2, 1, bias = False)
        self.batch1 = nn.BatchNorm2d(32)
        self.selu1 = nn.SELU(True)

        self.conv2 = nn.Conv2d(32 + 1, 64, 4, 2, 1, bias = False)
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

    def forward(self, input, sents):
        sents_output = self.sents_selu1(self.sents_linear(sents)).view(input.size()[0], 3, 64, 64)
        sents_output = self.sents_selu2(self.sents_batch(self.sents_conv(sents_output)))

        output = self.selu1(self.batch1(self.conv1(input)))

        output = torch.cat((output, sents_output), 1)
        output = self.selu2(self.batch2(self.conv2(output)))
        output = self.selu3(self.batch3(self.conv3(output)))
        output = self.selu4(self.batch4(self.conv4(output)))
        output = output.view(input.size()[0], -1)
        output = self.linear(output)
        output = self.out(output)
        return output.view(-1)

def train(_iter, _generator, _discriminator, _optimG, _optimD, _criterion, _train_loader, onehot_len):
    errD_all = Variable(torch.zeros(1))
    errG_all = Variable(torch.zeros(1))
    for idx, (_xs, _ys) in enumerate(_train_loader):
        # Train Discriminator
        _discriminator.zero_grad()
        reals = Variable(_xs)
        sents = Variable(_ys).float()
        labels = Variable(torch.ones(_xs.size()[0]))
        #additive_noise = Variable(torch.FloatTensor(_xs.size()[0], 3, 96, 96))

        if use_cuda:
            reals = reals.cuda()
            sents = sents.cuda()
            labels = labels.cuda()
            #additive_noise = additive_noise.cuda()
        #if USE_ADDITIVE_NOISE:
        #    additive_noise.data.resize_(reals.size()).normal_(0, 0.005)
        #    reals.data.add_(additive_noise.data)
        output = _discriminator(reals, sents)
        errD_real = _criterion(output, labels)

        fake_sents = Variable(torch.from_numpy(np.random.randint(2, size=(sents.size()[0] * onehot_len)))).float().view(sents.size()[0], -1)
        wrong_labels = Variable(torch.zeros(_xs.size()[0]))
        if use_cuda:
            fake_sents = fake_sents.cuda()
            wrong_labels = wrong_labels.cuda()
        output = _discriminator(reals, fake_sents)
        errD_wrong = _criterion(output, wrong_labels)
    
        noise = torch.from_numpy(np.random.normal(0, 1, (_xs.size()[0], NOISE_SIZE)))
        #noise = torch.randn(_xs.size()[0], NOISE_SIZE)
        noise = Variable(noise).float()
        fakes_labels = Variable(torch.zeros(_xs.size()[0]))
        if use_cuda:
            noise = noise.cuda()
            fakes_labels = fakes_labels.cuda()
        fakes = _generator(noise, sents)
        output = _discriminator(fakes.detach(), sents)
        errD_fake = _criterion(output, fakes_labels)

        errD_all = errD_real + errD_fake + errD_wrong
        errD_all.backward()
        _optimD.step()
        #for p in _discriminator.parameters():
        #    p.data.clamp_(-0.05, 0.05)
    
        # Train Generator
        _generator.zero_grad()
        fakes_labels = Variable(torch.ones(_xs.size()[0]))
        if use_cuda:
            fakes_labels = fakes_labels.cuda()
        fakes = _generator(noise, sents)
        output = _discriminator(fakes, sents)
        errG = _criterion(output, fakes_labels)
        errG_all = errG
        errG_all.backward()
        _optimG.step()
    
    return errD_all.data[0], errG_all.data[0]

def trainIters(_generator, _discriminator, _optimG, _optimD, _criterion, _train_loader, _testing_sents, onehot_len, print_every=1):
    print('Start training...')
    start = time.time()
    
    for iter in range(1, EPOCHS + 1):
        lossD, lossG = train(iter, _generator, _discriminator, _optimG, _optimD, _criterion, _train_loader, onehot_len)
        
        if iter % print_every == 0:
            avg_epoch_time = (time.time() - start) / print_every
            start = time.time()
            print('Epoch: %d, Avg_Epoch_Time: %.4f, LossD: %.4f, LossG: %.4f' % (iter, avg_epoch_time, lossD, lossG))
        if iter % 50 == 0:
            save_imgs(_generator, _testing_sents, iter)
            torch.save(generator, 'generator.pt')
            torch.save(discriminator, 'discriminator.pt')
        
train_loader, onehot_len = get_train_loader('AnimeDataset/faces/', 'AnimeDataset/tags_clean.csv', 'AnimeDataset/extra_data/images/', 'AnimeDataset/extra_data/tags.csv')
testing_data = [l.split(',')[1].replace('\n', '').split(' ') for l in list(open('AnimeDataset/testing_tags.txt').readlines())]
testing_hair = [i[0] for i in testing_data]
testing_eyes = [i[2] for i in testing_data]
testing_sents = np.array([np.concatenate((wd_hair.sentence2onehot(testing_hair[i]), wd_eyes.sentence2onehot(testing_eyes[i]))) for i in range(len(testing_data))])

generator = Generator(onehot_len)
generator.apply(weights_init)
discriminator = Discriminator(onehot_len)
discriminator.apply(weights_init)

if use_cuda:
    generator = generator.cuda()
    discriminator = discriminator.cuda()

criterion = nn.BCELoss()
optimizerG = optim.Adam(generator.parameters(), lr = LEARNING_RATE, betas = LEARNING_RATE_BETA)
optimizerD = optim.Adam(discriminator.parameters(), lr = LEARNING_RATE, betas = LEARNING_RATE_BETA)

trainIters(generator, discriminator, optimizerG, optimizerD, criterion, train_loader, testing_sents, onehot_len)

