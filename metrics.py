import math
import numpy as np
import torch

import utils


def entropy(probs):
    return -torch.sum(utils.where(probs > 0, probs * torch.log(probs)), dim=-1)


def max_entropy(c_dim):
    return math.log(c_dim)


def accuracy(matrix):
    return np.trace(matrix) / np.sum(matrix)


def confusion_matrix(y_hat, y, c_dim):
    matrix = np.zeros((c_dim, c_dim), dtype=np.uint32)
    np.add.at(matrix, [y, y_hat], 1)
    return matrix
