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
    return torch.softmax(logits, dim=1)


def calculate_metrics(matrix):
    acc = np.trace(matrix) / np.sum(matrix)
    return acc


def confusion_matrix(y_hat, y, c_dim):
    matrix = np.zeros((c_dim, c_dim), dtype=np.uint32)
    for i in range(c_dim):
        for j in range(c_dim):
            matrix[i, j] = ((y_hat == j) * (y == i)).type(torch.FloatTensor).sum()
    return matrix


def get_average(a):
    return np.mean(np.asarray(a))


def get_sum(a):
    return np.sum(np.asarray(a))


def append(*args):
    for arg in args:
        if arg[1]:
            arg[0].append(arg[1])


def flush():
    print('\n')
    sys.stdout.flush()
