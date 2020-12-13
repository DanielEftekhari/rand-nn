import sys
import os
import datetime
import shutil

import numpy as np

import torch


class AverageMeter():
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.avg = 0.
        self.var = 0.
        self.std = 0.
        self.n = 0
    
    def update(self, vals, m):
        sums = torch.sum(vals).item()
        squared_sums = torch.sum(torch.square(vals)).item()
        
        new_avg = (self.avg * self.n + sums) / (self.n + m)
        self.var = self.n / (self.n + m) * (self.var + abs(self.avg) ** 2) + 1 / (self.n + m) * squared_sums - abs(new_avg) ** 2
        self.std = self.var ** 0.5
        self.avg = new_avg
        self.n += m


def get_current_time():
    return str(datetime.datetime.utcnow()).replace(':', '-').replace(' ', '-')[0:-7]


def make_dirs(path, replace):
    if not os.path.isdir(path):
        os.makedirs(path)
    else:
        if replace:
            shutil.rmtree(path)
            os.makedirs(path)


def read_params(path, delimeter=' '):
    res = []
    with open(path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            line = line.split(delimeter)
            line = [int(line[i]) for i in range(len(line))]
            res.append(line)
    return res


def get_class_outputs(logits):
    return torch.argmax(logits, dim=-1)


def get_class_probs(logits):
    return torch.softmax(logits, dim=-1)


def to_one_hot(y, c_dim):
    y_one_hot = torch.zeros(size=(y.shape[0], c_dim))
    y_one_hot.scatter_(1, y.unsqueeze(-1), 1)
    return y_one_hot


def entropy(probs):
    return -torch.sum(probs * torch.log(probs), dim=-1)


def calculate_acc(matrix):
    return np.trace(matrix) / np.sum(matrix)


def confusion_matrix(y_hat, y, c_dim):
    matrix = np.zeros((c_dim, c_dim), dtype=np.uint32)
    for i in range(c_dim):
        for j in range(c_dim):
            matrix[i, j] = ((y_hat == j) * (y == i)).type(torch.FloatTensor).sum()
    return matrix


def append(*args):
    for arg in args:
        if arg[1]:
            arg[0].append(arg[1])


def flush():
    print('\n')
    sys.stdout.flush()
