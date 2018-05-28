import os
import numpy as np
import json
from pprint import pprint
import operator

class Word_dict:
    def __init__(self, one_hot=False):
        self.w2n = {'{UNK}': 0}
        self.n2w = {0: '{UNK}'}
        if not one_hot:
            self.w2n['{PAD}'] = 1
            self.n2w[1] = '{PAD}'
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
    def add_sentences_min_freq(self, sentences, min_word_freq):
        wc = {}
        for s in sentences:
            words = s.lower().replace('\n', '').split(' ')
            words = list(filter(None, words))
            for word in words:
                if word not in wc:
                    wc[word] = 1
                else:
                    wc[word] += 1
        for word in wc:
            if wc[word] >= min_word_freq:
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
            result.append(self.word2number('{PAD}'))
        return np.array(result[:length])
    def sentence2onehot(self, sentence):
        words = sentence.lower().replace('\n', '').split(' ')
        words = filter(None, words)
        numbers = [self.word2number(w) for w in words]
        result = np.zeros(len(self.w2n))
        for n in numbers:
            result[n] = 1
        return result
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
            tmp = l.replace('\n', '').split(' ')
            if len(tmp) == 2:
                self.w2n[tmp[0]] = int(tmp[1])
                self.n2w[int(tmp[1])] = tmp[0]
'''
######################## usage ########################
wd = Word_dict()
MAX_LENGTH = 15

data = load_data('clr_conversation.txt', MAX_LENGTH, wd)
print(len(data)) #sections: 56523
print(sum([len(d) for d in data]) / len(data)) #avg sentences in a section: 49.2888735558976
'''
