import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import torch.utils.data as Data
import numpy as np

use_cuda = torch.cuda.is_available()

import time
import random

import data_process
from pprint import pprint

BATCH_SIZE = 128

feats, labels = data_process.load_data('MLDS_hw2_1_data/training_label.json', 'MLDS_hw2_1_data/training_data/feat')
def get_train_loader():
    xs = torch.from_numpy(feats.astype(np.float32))
    ys = torch.from_numpy(np.array([ls[random.randint(0, len(ls) - 1)] for ls in labels]).astype(np.float64)).long()

    torch_dataset = Data.TensorDataset(data_tensor=xs, target_tensor=ys)
    train_loader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2
    )
    return train_loader

BOS_token = data_process.wd.w2n['{BOS}']
EOS_token = data_process.wd.w2n['{EOS}']
MAX_LENGTH = 25

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size)
    
    def forward(self, input, hidden):
        output, hidden = self.gru(input, hidden)
        return output, hidden

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
        
        self.embedding = nn.Embedding(voc_size + 1, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, voc_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden
    def initHidden(self):
        result = Variable(torch.zeros(1, BATCH_SIZE, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result

teacher_forcing_ratio = 0.5


def train(train_loader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    loss_all = 0

    for batch_idx, (fs, ts) in enumerate(train_loader):
        current_batch_size = fs.size()[0]
        encoder_hidden = encoder.initHidden(current_batch_size)
    
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
    
        encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
        encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs
        
        loss = 0

        if use_cuda:
            fs, ts = fs.cuda(), ts.cuda()
        fs, ts = Variable(fs.transpose(0, 1)), Variable(ts.transpose(0, 1))
        encoder_output, encoder_hidden = encoder(fs, encoder_hidden)
        
        decoder_input = Variable(torch.LongTensor(np.full((1, current_batch_size), BOS_token)))
        if use_cuda:
            decoder_input = decoder_input.cuda()
    
        decoder_hidden = encoder_hidden
    
        #use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
        use_teacher_forcing = True
            
        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(ts.size()[0]):
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
                loss += criterion(decoder_output, ts[di])
                if di != ts.size()[0] - 1:
                    decoder_input = ts[di+1].contiguous().view(1, -1)  # Teacher forcing
        else:
            # -------------------------- HELP ---------------------------
            # Without teacher forcing: use its own predictions as the next input
            for di in range(ts.size()[0]):
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
                topv, topi = decoder_output.data.topk(1)
                ni = topi[0][0]
                decoder_input = Variable(torch.LongTensor([[ni]]))
                decoder_input = decoder_input.cuda() if use_cuda else decoder_input
                loss += criterion(decoder_output, ts[di])
                if ni == EOS_token:
                    break
            # ------------------------ END HELP -------------------------
        loss_all += loss.data[0]
        loss.backward(retain_graph=True)
        encoder_optimizer.step()
        decoder_optimizer.step()

    return loss_all / len(train_loader)

def trainIters(encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()
    
    for iter in range(1, n_iters + 1):
        train_loader = get_train_loader()
        
        loss = train(train_loader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        
        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            avg_epoch_time = (time.time() - start) / print_every
            start = time.time()
            print('Epoch: %d, Avg_Epoch_Time: %.4f, Loss: %.4f' % (iter, avg_epoch_time, print_loss_avg))

hidden_size = 256
encoder_input_size = feats.shape[2]
encoder = EncoderRNN(encoder_input_size, hidden_size)
decoder = DecoderRNN(hidden_size, len(data_process.wd.w2n))

if use_cuda:
    encoder = encoder.cuda()
    decoder = decoder.cuda()
            
trainIters(encoder, decoder, 75000, print_every=10)
