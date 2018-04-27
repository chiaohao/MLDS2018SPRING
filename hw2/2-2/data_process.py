import os
import numpy as np
import json
from pprint import pprint

class Word_dict:
    def __init__(self):
        self.w2n = {'{UNK}': 0, '{BOS}': 1, '{EOS}': 2, '{PAD}': 3}
        self.n2w = {0: '{UNK}', 1: '{BOS}', 2: '{EOS}', 3: '{PAD}'}
    def add_sentences(self, sentences):
        if type(sentences) == str:
            sentences = [sentences]
        for s in sentences:
            words = s.lower().replace('\n', '').split(' ')
            words = list(filter(None, words))
            for word in words:
                if word not in self.w2n:
                    _id = len(self.n2w)
                    self.w2n[word] = _id
                    self.n2w[_id] = word
    def word2number(self, word):
        return self.w2n[word] if word in self.w2n else self.w2n['{UNK}']
    def number2word(self, number):
        return self.n2w[number] if number < len(self.n2w) else ''
    def sentence2number(self, sentence, length):
        words = sentence.lower().replace('\n', '').split(' ')
        words = filter(None, words)
        result = [self.word2number(word) for word in words]
        for i in range(length - len(result)):
            if i == 0:
                result.append(self.word2number('{EOS}'))
            else:
                result.append(self.word2number('{PAD}'))
        return np.array(result[:length])
    def save_dict(self, file_name):
        f = open(file_name, 'w')
        for w in self.w2n:
            f.write(w + ' ' + str(self.w2n[w]) + '\n')
    def load_dict(self, file_name):
        self.w2n = {}
        self.n2w = {}
        f = open(file_name, 'r')
        lines = list(f.readlines())
        for l in lines:
            tmp = l.replace('\n').split(' ')
            if len(tmp) == 2:
                self.w2n[tmp[0]] = int(tmp[1])
                self.n2w[int(tmp[1])] = tmp[2]

def load_sentences(path):
    f = open(path, 'r')
    sentences_raw = list(f.readlines())
    sentences_group = []
    tmp = []
    for i, s in enumerate(sentences_raw):
        if s != '+++$+++\n':
            tmp.append(s)
            if i == len(sentences_raw) - 1:
                sentences_group.append(tmp)
                tmp = []
        else:
            sentences_group.append(tmp)
            tmp = []
    return sentences_group

def load_data(path, max_length, _wd, is_add_words=True):
    sentences = load_sentences(path)
    if is_add_words:
        for ss in sentences:
            _wd.add_sentences(ss)
    ps = []
    for ss in sentences:
        ps.append([_wd.sentence2number(s, max_length) for s in ss])
    ps = np.array(ps)
    return ps

'''
######################## usage ########################
wd = Word_dict()
MAX_LENGTH = 15

data = load_data('clr_conversation.txt', MAX_LENGTH, wd)
print(len(data)) #sections: 56523
print(sum([len(d) for d in data]) / len(data)) #avg sentences in a section: 49.2888735558976
'''
