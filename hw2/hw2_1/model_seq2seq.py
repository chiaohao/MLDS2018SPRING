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

MAX_LENGTH = 15
wd = Word_dict()
folder = 'origin/'

features, labels = load_data('../../../../MLDS_hw2_1_data/training_label.json', '../../../../MLDS_hw2_1_data/training_data/feat', wd, MAX_LENGTH)

with open('wd.pkl', 'wb') as word_dict:
    pk.dump(wd, word_dict, pk.HIGHEST_PROTOCOL)

features_t, labels_t = load_data('../../../../MLDS_hw2_1_data/testing_label.json', '../../../../MLDS_hw2_1_data/testing_data/feat', wd, MAX_LENGTH, False)
INPUT_SIZE = features.shape[2]
HIDDEN_SIZE = 256
EPOCH = 200
BATCH_SIZE = 256

def get_train_loader(feats, labels, d=10):
    xs = torch.from_numpy(feats.astype(np.float32))
    torch_datasets = []
    for i in range(d):
        ys = torch.from_numpy(np.array([ls[i] if len(ls) > i else ls[len(ls)-1] for ls in labels]).astype(np.float64)).long()
        torch_datasets.append(Data.TensorDataset(data_tensor=xs, target_tensor=ys))
    torch_dataset = Data.ConcatDataset(torch_datasets)
    train_loader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2
    )
    return train_loader

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

def trainPerEpoch(loader, encoder, decoder, criterion, epoch, max_length=MAX_LENGTH):
    loss_all = 0
    for batch_index, (_features, _labels) in enumerate(loader):
        loss = 0
        batch_size = _features.size()[0]
        encoder.optimizer.zero_grad()
        decoder.optimizer.zero_grad()
        if torch.cuda.is_available():
            _features, _labels = _features.cuda(), _labels.cuda()
        _features, _labels = Variable(_features.transpose(0,1)), Variable(_labels.transpose(0,1))
        encoder_output, encoder_hidden = encoder(_features)
        decoder_input = Variable(torch.LongTensor(np.full((1, batch_size), wd.w2n['{BOS}'])))
        decoder_hidden = encoder_hidden
        if torch.cuda.is_available():
            decoder_input = decoder_input.cuda()
        threshold = reversed_sigmoid(epoch)
        for _iter in range(_labels.size()[0]):
            use_teaching_forcing = True if random.random() < threshold else False
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_output)
            loss += criterion(decoder_output, _labels[_iter])
            if use_teaching_forcing:
                decoder_input = _labels[_iter].contiguous().view(1, -1)
            else:
                val, idx = decoder_output.data.topk(1)
                decoder_input = Variable(idx.view(1,-1))
        loss_all += loss.data[0]
        loss.backward(retain_graph=True)
        decoder.optimizer.step()
        encoder.optimizer.step()
    return loss_all/len(loader)

def train(encoder, decoder, epoch, print_iter=1000, save_iter=200):
    start = time.time()
    criterion = nn.NLLLoss()
    history = {'epoch': [], 'loss':[] }
    for iteration in range(1, epoch+1):
        loader = get_train_loader(features, labels)
        loss = trainPerEpoch(loader, encoder, decoder, criterion, iteration)
        if iteration % print_iter == 0:
            end = time.time()
            avg_epoch_time = (end - start)/print_iter
            start = end
            history['loss'].append(loss)
            history['epoch'].append(iteration)
            print('Epoch: %d, Avg_Epoch_Time: %.4f, Loss: %.4f' % (iteration, avg_epoch_time, loss))
        if iteration % save_iter == 0:
            model_save(encoder, decoder, iteration)
#         if iteration == EPOCH:
            history_save(history, iteration)

def reversed_sigmoid(_epoch):
    return (1 - (1/ (1 + math.exp(-(_epoch - EPOCH/2)/EPOCH*6))))/1*0.6 + 0.4

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

def model_save(encoder, decoder, epoch, d=5, attn='attn', min_count=0):
    torch.save(encoder, '{}encoder_basic_{}-{}_{}-mc={}'.format(folder,epoch,d,attn, min_count))
    torch.save(decoder, '{}decoder_basic_{}-{}_{}_mc={}'.format(folder,epoch,d,attn, min_count))
    
def history_save(history, epoch, d=5, attn='attn', min_count=0):
    with open('{}history_{}-{}_{}-_mc={}'.format(folder,epoch, d, attn, min_count), 'wb') as hist_f:
        pk.dump(history, hist_f)

def result_save(result):
    text = folder + 'result.txt'
    with open( text, 'w') as file:
        labels = load_label('../../../../MLDS_hw2_1_data/testing_label.json')
        for i, lb in enumerate(labels):
            file.write(lb['id'] + ',' + result_c[i]+'\n')

encoder = EncoderRNN(INPUT_SIZE, HIDDEN_SIZE)
decoder = DecoderRNN(HIDDEN_SIZE, len(wd.w2n))
if torch.cuda.is_available:
    encoder, decoder = encoder.cuda(), decoder.cuda()

train(encoder, decoder, EPOCH, print_iter=5, save_iter=EPOCH/4)
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

result_save(result_c)

