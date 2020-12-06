import numpy as np
from scipy import special

import torch
import torch.nn as nn

import layers


class FCNet(nn.Module):
    def __init__(self, dim, c_dim, units, activation, norm, cfg):
        super(FCNet, self).__init__()
        self.cfg = cfg
        
        self.activation = activation
        self.norm = norm
        
        all_units = [dim] + units + [c_dim]
        
        self.norms, self.layers = nn.ModuleList(), nn.ModuleList()
        for i in range(len(all_units)-1):
            self.layers.append(nn.Linear(all_units[i], all_units[i+1], bias=True))
            if self.norm:
                self.norms.append(self.norm(all_units[i+1]))
    
    def forward(self, x):
        # flatten input
        x = x.view(x.shape[0], -1)
        
        for i in range(len(self.layers)-1):
            x = self.layers[i](x)
            if self.norm:
                x = self.norms[i](x)
            x = self.activation(x)
        x = self.layers[-1](x)
        return x


class ConvNet(nn.Module):
    def __init__(self, dims, c_dim, units, activation, norm, cfg):
        super(ConvNet, self).__init__()
        self.cfg = cfg
        
        self.activation = activation
        self.norm = norm
        
        c_in = units[0][0]
        assert (c_in == dims[0])
        h, w = dims[1], dims[2]
        
        self.norms, self.layers = nn.ModuleList(), nn.ModuleList()
        for i in range(len(units)):
            c_in, c_out, k, s, p = units[i][0], units[i][1], units[i][2], units[i][3], units[i][4]
            self.layers.append(nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=k, stride=s, padding=p, bias=True))
            h, w = int((h - k + 2*p) / s) + 1, int((w - k + 2*p) / s) + 1
            if self.norm:
                self.norms.append(self.norm(c_out))
        self.layers.append(nn.Linear(c_out * h * w, c_dim, bias=True))
    
    def forward(self, x):
        for i in range(len(self.layers)-1):
            x = self.layers[i](x)
            if self.norm:
                x = self.norms[i](x)
            x = self.activation(x)
        
        # flatten for CONV -> FC layer
        x = x.view(x.shape[0], -1)
        x = self.layers[-1](x)
        return x
