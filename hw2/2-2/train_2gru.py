import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import torch.utils.data as Data
import numpy as np
import sys

use_cuda = torch.cuda.is_available()

import time
import random

import data_process
from pprint import pprint

wd = data_process.Word_dict()
MAX_LENGTH = 15
print('Start process raw data... ', end='')
sys.stdout.flush()
data = data_process.load_data('clr_conversation.txt', MAX_LENGTH, wd, min_word_freq=5)
print('Done!')
print('Words in dict: %d' % len(wd.w2n))
sys.stdout.flush()

wd.save_dict('word_dict.txt')

BOS_token = wd.w2n['{BOS}']
EOS_token = wd.w2n['{EOS}']
PAD_token = wd.w2n['{PAD}']

BATCH_SIZE = 100
hidden_size = 256
learning_rate = 0.001
EPOCHS = 7500

xs = []
ys = []
for sect in data:
    for i in range(len(sect)):
        if i < len(sect) - 1:
            xs.append(sect[i])
            ys.append(sect[i + 1])
data_count = len(xs)
#xs = torch.from_numpy(np.array(xs).astype(np.float64)).long()
#ys = torch.from_numpy(np.array(ys).astype(np.float64)).long()

def get_train_data():
    randints = random.sample(range(0, data_count), BATCH_SIZE)
    _xs = []
    _ys = []
    for i in randints:
        _xs.append(xs[i])
        _ys.append(ys[i])
    max_len_xs = max([(list(_x) + [PAD_token]).index(PAD_token) for _x in _xs])
    max_len_ys = max([(list(_y) + [PAD_token]).index(PAD_token) for _y in _ys])
    _xs = [_x[:max_len_xs] for _x in _xs]
    _ys = [_y[:max_len_ys] for _y in _ys]
    _xs = torch.from_numpy(np.array(_xs).astype(np.float64)).long()
    _ys = torch.from_numpy(np.array(_ys).astype(np.float64)).long()
    return _xs, _ys

class EncoderRNN(nn.Module):
    def __init__(self, hidden_size, voc_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(voc_size, hidden_size)
        self.gru1 = nn.GRU(hidden_size, hidden_size)
        self.gru2 = nn.GRU(hidden_size, hidden_size)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        #self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, 5)
    
    def forward(self, input, hidden1, hidden2):
        output = self.embedding(input)
        output = F.selu(output)
        output1, hidden1 = self.gru1(output, hidden1)
        output2, hidden2 = self.gru2(output1, hidden2)
        return hidden1, hidden2

    def initHidden(self, batch_size):
        result = Variable(torch.zeros(1, batch_size, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, voc_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(voc_size, hidden_size)
        self.gru1 = nn.GRU(hidden_size, hidden_size)
        self.gru2 = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, voc_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        #self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, 5)

    def forward(self, input, hidden1, hidden2):
        output = self.embedding(input)
        output = F.selu(output)
        output1, hidden1 = self.gru1(output, hidden1)
        output2, hidden2 = self.gru2(output1, hidden2)
        _output = self.softmax(self.out(output2[0]))
        return _output, hidden1, hidden2
    def initHidden(self):
        result = Variable(torch.zeros(1, BATCH_SIZE, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result

teacher_forcing_ratio = 0.5


def train(encoder, decoder, criterion, max_length=MAX_LENGTH):
    sys.stdout.flush()

    fs, ts = get_train_data()
    
    #encoder.scheduler.step()
    #decoder.scheduler.step()

    current_batch_size = fs.size()[0]
    encoder_hidden1 = encoder.initHidden(current_batch_size)
    encoder_hidden2 = encoder.initHidden(current_batch_size)
    
    encoder.optimizer.zero_grad()
    decoder.optimizer.zero_grad()
    
    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs
        
    loss = 0

    if use_cuda:
        fs, ts = fs.cuda(), ts.cuda()
    fs, ts = Variable(fs.transpose(0, 1)), Variable(ts.transpose(0, 1))
    encoder_hidden1, encoder_hidden2 = encoder(fs, encoder_hidden1, encoder_hidden2)
    
    decoder_input = Variable(torch.LongTensor(np.full((1, current_batch_size), BOS_token)))
    if use_cuda:
        decoder_input = decoder_input.cuda()
    
    decoder_hidden1 = encoder_hidden1
    decoder_hidden2 = encoder_hidden2

    ## Word Teacher Forcing
    for di in range(ts.size()[0]):
        decoder_output, decoder_hidden1, decoder_hidden2 = decoder(decoder_input, decoder_hidden1, decoder_hidden2)
        loss += criterion(decoder_output, ts[di])
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
        if use_teacher_forcing:
            decoder_input = ts[di].contiguous().view(1, -1)
        else:
            topv, topi = decoder_output.data.topk(1)
            decoder_input = Variable(topi.view(1, -1))
    loss.backward(retain_graph=True)
    encoder.optimizer.step()
    decoder.optimizer.step()
        
    return loss.data[0] / max_length

def trainIters(encoder, decoder, n_iters, print_every=1000, plot_every=100):
    print('Start training...')
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    
    criterion = nn.NLLLoss()
    for iter in range(1, n_iters + 1):
        loss = train(encoder, decoder, criterion)
        print_loss_total += loss
        
        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            avg_epoch_time = (time.time() - start) / print_every
            start = time.time()
            print('Epoch: %d, Avg_Epoch_Time: %.4f, Loss: %.4f' % (iter, avg_epoch_time, print_loss_avg))
        torch.save(encoder, 'encoder.pt')
        torch.save(decoder, 'decoder.pt')

encoder = EncoderRNN(hidden_size, len(wd.w2n))
decoder = DecoderRNN(hidden_size, len(wd.w2n))

if use_cuda:
    encoder = encoder.cuda()
    decoder = decoder.cuda()

trainIters(encoder, decoder, EPOCHS, print_every=1)
