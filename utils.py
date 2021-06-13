import sys
import os
import datetime
import shutil
import json

import math
import numpy as np
import torch


class Meter():
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.avg = 0.
        self.var = 0.
        self.std = 0.
        self.n = 0
    
    def update(self, vals, m):
        sums = np.sum(vals)
        squared_sums = np.sum(np.square(vals))
        
        new_avg = (self.avg * self.n + sums) / (self.n + m)
        self.var = self.n / (self.n + m) * (self.var + math.pow(self.avg, 2)) + 1 / (self.n + m) * squared_sums - math.pow(new_avg, 2)
        self.var = max(0., self.var)
        self.std = math.sqrt(self.var)
        self.avg = new_avg
        self.n += m


def bootstrap_ci(data, num_sample=1000, alpha=0.025):
    assert (len(data.shape) == 1)
    n = data.shape[0]
    x_bar = np.mean(data)
    
    samples = np.random.choice(data, size=(num_sample, n), replace=True)
    delta = np.mean(samples, axis=1) - x_bar
    delta = np.sort(delta)
    
    l, r = delta[int((num_sample-1) * (1-alpha))], delta[int((num_sample-1) * alpha)]
    return x_bar-l, x_bar, x_bar-r


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


def load_json(filepath):
    if not os.path.isfile(filepath):
        return None
    with open(filepath) as data_file:
        data = json.load(data_file)
    return data


def save_json(data, filepath):
    with open(filepath, 'w') as outfile:
        json.dump(obj=data, fp=outfile, sort_keys=True, indent=4, separators=(',', ': '))


def save_array(data, filepath):
    np.save(file=filepath, arr=data)


def tensor2array(tensor):
    return tensor.cpu().detach().numpy()


def get_class_outputs(outputs):
    return torch.argmax(outputs, dim=-1)


def logits_to_probs(logits):
    c = torch.max(logits, dim=-1)[0]
    return torch.softmax(logits - c[:, None], dim=-1)


def to_one_hot(y, c_dim):
    y_one_hot = torch.zeros(size=(y.shape[0], c_dim))
    y_one_hot.scatter_(1, y.unsqueeze(-1), 1)
    return y_one_hot


def where(condition, x, y=0.):
    return torch.where(condition, x, torch.Tensor([y]).to(x.device))


def append(*args):
    for arg in args:
        if arg[1]:
            arg[0].append(arg[1])


def get_current_time():
    return str(datetime.datetime.utcnow()).replace(':', '-').replace(' ', '-')[0:-7]


def flush():
    print('\n')
    sys.stdout.flush()
