import torch.nn as nn


def sigmoid():
    return nn.Sigmoid()


def tanh():
    return nn.Tanh()


def relu():
    return nn.ReLU()


def linear():
    return nn.Identity()
