import torch
import torch.nn as nn


def additive_gaussian_noise_(x, mu=0., std=1.):
    with torch.no_grad():
        x.add_(torch.randn_like(x) * std + mu)


def multiplicative_gaussian_noise_(x, mu=1., std=1.):
    with torch.no_grad():
        x.mul_(torch.randn_like(x) * std + mu)


class LayerNorm1d(nn.Module):
    # source: <https://github.com/pytorch/pytorch/issues/1959#issuecomment-312364139>
    def __init__(self, features, eps=1e-6):
        super(LayerNorm1d, self).__init__()
        
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps
    
    def forward(self, x):
        mean = x.mean(1, keepdim=True)
        std = x.std(1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class LayerNorm2d(nn.Module):
    # source: <https://github.com/pytorch/pytorch/issues/1959#issuecomment-343435542>
    def __init__(self, num_features, eps=1e-6, affine=True):
        super(LayerNorm2d, self).__init__()
        
        if affine:
            self.gamma = nn.Parameter(torch.ones(num_features))
            self.beta = nn.Parameter(torch.zeros(num_features))
        self.affine = affine
        self.eps = eps
    
    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        mean = x.view(x.size(0), -1).mean(1).view(*shape)
        std = x.view(x.size(0), -1).std(1).view(*shape)
        
        y = (x - mean) / (std + self.eps)
        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            y = self.gamma.view(*shape) * y + self.beta.view(*shape)
        return y
