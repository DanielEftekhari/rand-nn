import itertools

import numpy as np
import torch
import torch.nn as nn

import layers


class FCNet(nn.Module):
    def __init__(self, dims, c_dim, units, activation, norm):
        super(FCNet, self).__init__()
        
        all_units = [np.product(dims)] + list(itertools.chain.from_iterable(units))
        
        self.layers = nn.ModuleDict()
        for i in range(len(all_units)-1):
            self.layers['linear{}'.format(i+1)] = nn.Linear(all_units[i], all_units[i+1], bias=True)
            if norm:
                self.layers['norm{}'.format(i+1)] = norm(all_units[i+1])
            self.layers['activation{}'.format(i+1)] = activation()
        self.layers['fc'] = nn.Linear(all_units[-1], c_dim, bias=True)
        
        self.names = list(self.layers.keys())
    
    def forward(self, x):
        # flatten input
        x = x.view(x.shape[0], -1)
        
        for layer in self.layers:
            x = self.layers[layer](x)
        return x


class ConvNet(nn.Module):
    def __init__(self, dims, c_dim, units, activation, norm):
        super(ConvNet, self).__init__()
        
        units[0][0] = dims[0]
        h, w = dims[1], dims[2]
        
        self.layers = nn.ModuleDict()
        for i in range(len(units)):
            [c_in, c_out, k, s, p] = units[i]
            self.layers['conv{}'.format(i+1)] = nn.Conv2d(c_in, c_out, k, s, p, bias=True)
            if norm:
                self.layers['norm{}'.format(i+1)] = norm(c_out)
            self.layers['activation{}'.format(i+1)] = activation()
            h, w = int((h - k + 2*p) / s) + 1, int((w - k + 2*p) / s) + 1
        self.layers['fc'] = nn.Linear(c_out * h * w, c_dim, bias=True)
        
        self.names = list(self.layers.keys())
    
    def forward(self, x):
        for i in range(len(self.names)-1):
            x = self.layers[self.names[i]](x)
        
        # flatten for CONV -> FC layer
        x = x.view(x.shape[0], -1)
        x = self.layers[self.names[-1]](x)
        return x
