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

hidden_size = 256
MAX_LENGTH = 15
BATCH_SIZE = 100
learning_rate = 0.001

wd = data_process.Word_dict()
wd.load_dict(sys.argv[1])
BOS_token = wd.w2n['{BOS}']
EOS_token = wd.w2n['{EOS}']

def read_data(file_name, _wd):
    f = open(file_name)
    lines = f.readlines()
    data = [_wd.sentence2number(line, MAX_LENGTH) for line in lines]
    return np.array(data).astype(np.float64)

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

def test(_td, encoder, decoder, max_length=MAX_LENGTH):
    td = Variable(torch.from_numpy(_td)).long()
    if use_cuda:
        td = td.cuda()
    
    current_batch_size = td.size()[0]
    encoder_hidden1 = encoder.initHidden(current_batch_size)
    encoder_hidden2 = encoder.initHidden(current_batch_size)
    
    td = td.transpose(0, 1)
    encoder_hidden1, encoder_hidden2 = encoder(td, encoder_hidden1, encoder_hidden2)
    
    decoder_input = Variable(torch.LongTensor(np.full((1, current_batch_size), BOS_token)))
    if use_cuda:
        decoder_input = decoder_input.cuda()
        
    decoder_hidden1 = encoder_hidden1
    decoder_hidden2 = encoder_hidden2
    
    r = decoder_input

    for di in range(td.size()[0]):
        decoder_output, decoder_hidden1, decoder_hidden2 = decoder(decoder_input, decoder_hidden1, decoder_hidden2)
        topv, topi = decoder_output.data.topk(1)
        decoder_input = Variable(topi.view(1, -1))
        r = torch.cat((r, decoder_input), dim=0)
    
    return r

def test_iter(_tds, encoder, decoder):
    iters = int(len(_tds) / BATCH_SIZE)
    if len(_tds) % BATCH_SIZE != 0:
        iters += 1
    result = []
    for i in range(iters):
        td = _tds[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
        r = test(td, encoder, decoder)
        result.append(r)
    
    result = [j for i in result for j in i.transpose(0, 1).cpu().data]

    return result

test_data = read_data(sys.argv[4], wd)

print('Load encoder/decoder... ', end='')
encoder = EncoderRNN(hidden_size, len(wd.w2n))
decoder = DecoderRNN(hidden_size, len(wd.w2n))
encoder = torch.load(sys.argv[2])
decoder = torch.load(sys.argv[3])
print('Done!')

if use_cuda:
    encoder = encoder.cuda()
    decoder = decoder.cuda()
print('Start testing...', end='')
result = test_iter(test_data, encoder, decoder)
print('Done!')

result = [[wd.number2word(j) for j in i] for i in result]
output_result = []
for r in result:
    o = []
    for i in r:
        if i != '{BOS}' and i != '{EOS}' and i != '{PAD}':
            o.append(i)
    o = ''.join(o)
    output_result.append(o)

print(output_result)
