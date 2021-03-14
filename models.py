import itertools

import numpy as np
import torch
import torch.nn as nn

import layers


class LinearBlock(nn.Module):
    def __init__(self, in_features, out_features, activation, norm=None, bias=True):
        super(LinearBlock, self).__init__()
        
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.activation = activation
        if norm:
            self.norm = norm(out_features)
        else:
            self.norm = None
    
    def forward(self, x):
        x = self.linear(x)
        if self.norm:
            x = self.norm(x)
        x = self.activation(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, activation, norm=None, bias=True):
        super(ConvBlock, self).__init__()
        
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.activation = activation
        if norm:
            self.norm = norm(out_channels)
        else:
            self.norm = None
    
    def forward(self, x):
        x = self.conv2d(x)
        if self.norm:
            x = self.norm(x)
        x = self.activation(x)
        return x


class FCNet(nn.Module):
    def __init__(self, dims, c_dim, units, activation, norm):
        super(FCNet, self).__init__()
        
        all_units = [np.product(dims)] + list(itertools.chain.from_iterable(units))
        
        self.layers = nn.ModuleDict()
        for i in range(len(all_units)-1):
            self.layers['LinearBlock{}'.format(i+1)] = LinearBlock(all_units[i], all_units[i+1], activation, norm, bias=True)
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
            self.layers['ConvBlock{}'.format(i+1)] = ConvBlock(c_in, c_out, k, s, p, activation, norm, bias=True)
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
