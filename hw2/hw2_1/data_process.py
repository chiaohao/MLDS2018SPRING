import os
import numpy as np
import json
from pprint import pprint
import pickle as pk

class Word_dict:
    def __init__(self):
        self.w2n = {'{UNK}': 0, '{BOS}': 1, '{EOS}': 2, '{PAD}': 3}
        self.n2w = {0: '{UNK}', 1: '{BOS}', 2: '{EOS}', 3: '{PAD}'}
        self.wc = {0: 999, 1: 999, 2: 999, 3: 999}
    def add_sentences(self, sentences):
        if type(sentences) == str:
            sentences = [sentences]
        for s in sentences:
            words = s.lower().replace('.', '').split(' ')
            words = list(filter(None, words))
            for word in words:
                if word not in self.w2n:
                    _id = len(self.n2w)
                    self.w2n[word] = _id
                    self.n2w[_id] = word
                    self.wc[_id] = 1
                else:
                    self.wc[self.w2n[word]] += 1
    def word2number(self, word):
        return self.w2n[word] if word in self.w2n else self.w2n['{UNK}']
    def number2word(self, number):
        return self.n2w[number] if number < len(self.n2w) else ''
    def sentence2number(self, sentence, length):
        words = sentence.lower().replace('.', '').split(' ')
        words = filter(None, words)
        result = [self.word2number(word) for word in words]
        for i in range(length - len(result)):
            if i == 0:
                result.append(self.word2number('{EOS}'))
            else:
                result.append(self.word2number('{PAD}'))
        return np.array(result[:length])
    def tighten(self, min_count):
        if min_count <= 0:
            return
        holes = []
        original_len = len(self.wc)
        for k in range(original_len):
            if self.wc[k] <= min_count:
                holes.append(k)
                w = self.n2w.pop(k)
                del self.wc[k]
                del self.w2n[w]
        new_len = len(self.wc)
        for k in range(original_len):
            if self.n2w.get(k) and k > new_len-1:
                hole = holes.pop(0)
                self.n2w[hole] = self.n2w.pop(k)
                self.wc[hole] = self.wc.pop(k)
                self.w2n[self.n2w[hole]] = hole

def load_label(label_path):
    raw = json.load(open(label_path))
    #pprint(raw[0]['caption'])
    #pprint(raw[0]['id'])
    return raw

def load_feature(dir_path, feat_path):
    #file_paths = os.listdir(dir_path)
    #data = []
    #for fp in file_paths:
    #    data.append(np.load(dir_path + '/' + fp))
    #return np.array(data)
    return np.load(dir_path + '/' + feat_path)

def load_data(label_path, dir_path, wd, MAX_LENGTH, is_train=True, min_count=0):
    labels = load_label(label_path)
    if is_train:
        for label in labels:
            wd.add_sentences(label['caption'])
    
    wd.tighten(min_count)

    _feats = []
    _labels = []
    for label in labels:
        _feats.append(load_feature(dir_path, label['id'] + '.npy'))
        _labels.append([wd.sentence2number(s, MAX_LENGTH) for s in label['caption']])
    _feats = np.array(_feats)
    #_labels = np.array(_labels)
    return _feats, _labels

def load_test_data(folder_path):
    _feats = []
    _labels = []
    with open(folder_path+'/testing_id.txt', 'r') as idFile:
        for line in idFile.readlines():
            line = line.strip()
            _feats.append(load_feature(folder_path+'/feat', line + '.npy'))
            _labels.append([[3,3,3,3,3,3,3,3,3,3,3,3,3,3,3]])
    _feats = np.array(_feats)
    #_labels = np.array(_labels)
    return _feats, _labels
######################## usage ########################
'''
feats, labels = load_data('MLDS_hw2_1_data/training_label.json', 'MLDS_hw2_1_data/training_data/feat', wd, MAX_LENGTH)
pprint(feats.shape) #(1450, 80, 4096)
pprint(labels) #1450 * n * MAX_LENGTH   ^n means n descriptions for one video
'''
