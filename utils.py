import sys
import os
import shutil

import numpy as np

import torch


def make_dirs(path, replace):
    if not os.path.isdir(path):
        os.makedirs(path)
    else:
        if replace:
            shutil.rmtree(path)
            os.makedirs(path)


def get_class_outputs(logits):
    return torch.argmax(logits, dim=-1)


def get_class_probs(logits):
    return torch.softmax(logits, dim=-1)


def entropy(probs):
    return torch.mean(-torch.sum(probs * torch.log(probs), dim=-1))


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


class AverageMeter():
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.avg = 0
        self.count = 0
    
    def update(self, val, n):
        self.avg = (self.avg  * self.count + val * n) / (self.count + n)
        self.count += n


def flush():
    print('\n')
    sys.stdout.flush()
