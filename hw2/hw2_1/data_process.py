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
            words = s.lower().replace('.', '').split(' ')
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
        words = sentence.lower().replace('.', '').split(' ')
        words = filter(None, words)
        result = [self.word2number(word) for word in words]
        for i in range(length - len(result)):
            if i == 0:
                result.append(self.word2number('{EOS}'))
            else:
                result.append(self.word2number('{PAD}'))
        return np.array(result[:length])

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

wd = Word_dict()
MAX_LENGTH = 25

def load_data(label_path, dir_path):
    labels = load_label(label_path)
    for label in labels:
        wd.add_sentences(label['caption'])

    _feats = []
    _labels = []
    for label in labels:
        _feats.append(load_feature(dir_path, label['id'] + '.npy'))
        _labels.append([wd.sentence2number(s, MAX_LENGTH) for s in label['caption']])
    _feats = np.array(_feats)
    #_labels = np.array(_labels)
    return _feats, _labels

######################## usage ########################
'''
feats, labels = load_data('MLDS_hw2_1_data/training_label.json', 'MLDS_hw2_1_data/training_data/feat')
pprint(feats.shape) #(1450, 80, 4096)
pprint(labels) #1450 * n * MAX_LENGTH   ^n means n descriptions for one video
'''
