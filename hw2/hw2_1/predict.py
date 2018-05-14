from data_process import *
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import optim
import torch.utils.data as Data
import random
import time
import numpy as np
import pickle as pk
import math
import sys

MAX_LENGTH = 15

with open('wd.pkl', 'rb') as word_dict:
    wd = pk.load(word_dict)

folder = sys.argv[1]

features_t, labels_t = load_test_data(folder)
BATCH_SIZE = 256

def get_test_loader(feats, labels):
    xs = torch.from_numpy(feats.astype(np.float32))
    ys = torch.from_numpy(np.array([ls[random.randint(0, len(ls) - 1)] for ls in labels]).astype(np.float64)).long()

    torch_dataset = Data.TensorDataset(data_tensor=xs, target_tensor=ys)
    test_loader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2
    )
    return test_loader

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, dropout=dropout)
        self.optimizer = optim.Adam(self.parameters())
    
    def forward(self, input):
        output, hidden = self.gru(input)
        return output, hidden

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout=0.1):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, 80)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.optimizer = optim.Adam(self.parameters())
        self.softmax = nn.LogSoftmax(dim=1)
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, input, hidden, encoder_outputs):
        output = self.embedding(input)
        attn_weights = F.softmax(
            self.attn(self.dropout(torch.cat((output[0], hidden[0]), 1))), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(1),
                                 encoder_outputs.transpose(0,1))
        output = torch.cat((output[0], attn_applied.transpose(0,1)[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)
        output = self.dropout(F.selu(output))
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

def evaluate(encoder, decoder, features, labels):
    result = []
    counter = 0
    loader = get_test_loader(features, labels)
    for batch_index, (_features, _labels) in enumerate(loader):
        batch_size = _features.size()[0]
        for i in range(batch_size):
            result.append([])
        if torch.cuda.is_available():
            _features, _labels = _features.cuda(), _labels.cuda()
        _features, _labels = Variable(_features.transpose(0,1)), Variable(_labels.transpose(0,1))
        encoder_output, encoder_hidden = encoder(_features)
        decoder_input = Variable(torch.LongTensor(np.full((1, batch_size), wd.w2n['{BOS}'])))
        decoder_hidden = encoder_hidden
        if torch.cuda.is_available():
            decoder_input = decoder_input.cuda()
        for _iter in range(_labels.size()[0]):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_output)
            val, idx = decoder_output.data.topk(1)
            decoder_input = Variable(idx.view(1,-1))
            for i in range(batch_size):
                result[counter+i].append(wd.number2word(torch.max(decoder_output, 1)[1][i].data.cpu().numpy()[0]))
        counter += batch_size
    return result

def result_save(text):
    with open(text, 'w') as file:
        with open(folder+'/testing_id.txt', 'r') as idFile:
            for line in idFile.readlines():
                line = line.strip()
                file.write(line + ',' + result_c[i]+'\n')

encoder = torch.load('./encoder_model')
decoder = torch.load('./decoder_model')
if torch.cuda.is_available:
    encoder, decoder = encoder.cuda(), decoder.cuda()

encoder.eval()
decoder.eval()
result = evaluate(encoder, decoder, features_t, labels_t)

result_c = []
for r in result:
    r = [w if w != '{EOS}' and w != '{PAD}' else '' for w in r]
    ptr = len(r)
    for i, w in enumerate(r):
        if i <= len(r) - 3 and i > 0:
            for t in range(i-1):
                if r[t] == r[i] and r[t+1] == r[i+1] and r[t+2] == r[i+2]:
                    ptr = i-1
                    break
    r = r[:ptr]
    result_c.append(' '.join(r).rstrip())
result_save(sys.argv[2])

