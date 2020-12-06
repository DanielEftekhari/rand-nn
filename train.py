import sys
import os
import collections
import copy
import random

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from torchvision.datasets import MNIST, CIFAR10
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from config import get_config
import activations
import layers
from models import FCNet, ConvNet
from plotting import plot_line, plot_hist
import utils


class Trainer():
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device
        
        # dataset parameters
        if self.cfg.dataset.lower() == 'mnist':
            self.dataset = MNIST
            self.data_path = r'./data/mnist'
            self.img_size = [1, 28, 28]
            self.normalize = [(0.1307,), (0.3081,)]
        else:
            self.dataset = CIFAR10
            self.data_path = r'./data/cifar10'
            self.img_size = [3, 32, 32]
            self.normalize = [(0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)]
        
        # datasets and dataloaders        
        # base transforms
        self.train_transforms = [transforms.ToTensor()]
        if self.cfg.normalize_input:
            self.train_transforms.append(transforms.Normalize(self.normalize[0], self.normalize[1]))
        self.val_transforms = copy.deepcopy(self.train_transforms)
        
        # (if applicable) additional training set transforms defined here
        # train_transforms.extend([])
        
        # in training, <drop_last> minibatch in an epoch set to <True> for simplicity in tracking training performance
        self.dataset_train = self.dataset(root=self.data_path, train=True, download=True,
                                transform=transforms.Compose(self.train_transforms),
                                target_transform=None)
        self.dataloader_train = DataLoader(dataset=self.dataset_train, batch_size=self.cfg.batch_size, shuffle=self.cfg.shuffle,
                                    num_workers=self.cfg.num_workers, pin_memory=True, drop_last=True)
        
        self.dataset_val = self.dataset(root=self.data_path, train=False, download=True,
                            transform=transforms.Compose(self.val_transforms),
                            target_transform=None)
        self.dataloader_val = DataLoader(dataset=self.dataset_val, batch_size=100, shuffle=False,
                                    num_workers=self.cfg.num_workers, pin_memory=True, drop_last=False)
        
        # number of output classes
        targets = np.asarray(self.dataset_train.targets)
        self.c_dim = np.unique(targets).shape[0]

        # define model
        # weights initialized using Kaiming uniform (He initialization)
        # parameters for each hidden layer is passed in as an argument
        self.activation = getattr(activations, self.cfg.activation.lower())
        if 'fc' in self.cfg.nn_type.lower():
            if self.cfg.norm.lower() == 'batch':
                self.norm = nn.BatchNorm1d
            elif self.cfg.norm.lower() == 'layer':
                self.norm = layers.LayerNorm1d
            else:
                self.norm = None
            self.net = FCNet(np.product(self.img_size), self.c_dim, self.cfg.fc_params, self.activation, self.norm, self.cfg).to(self.device)
        else:
            if self.cfg.norm.lower() == 'batch':
               self.norm = nn.BatchNorm2d
            elif self.cfg.norm.lower() == 'layer':
                self.norm = layers.LayerNorm2d
            else:
                self.norm = None
            self.net = ConvNet(self.img_size, self.c_dim, self.cfg.conv_params, self.activation, self.norm, self.cfg).to(self.device)
        
        self.criterion = nn.CrossEntropyLoss()
        
        if self.cfg.use_sgd:
            self.optimizer = optim.SGD(params=self.net.parameters(), lr=self.cfg.lr, momentum=self.cfg.momentum, nesterov=self.cfg.use_nesterov)
        else:
            self.optimizer = optim.Adam(params=self.net.parameters(), lr=self.cfg.lr, betas=(self.cfg.beta1, self.cfg.beta2))
    
    def train(self):
        # tracking training and validation stats over epochs
        self.metrics = collections.defaultdict(list)
        # best model is defined as model with best performing validation loss
        self.best_loss = float('inf')
        
        # measure performance before any training is done
        self.metrics['epochs'].append(0)
        self.validate(self.dataloader_train, is_val_set=False, measure_entropy=False)
        self.validate(self.dataloader_val, is_val_set=True, measure_entropy=True)
        
        for epoch in range(1, self.cfg.epochs+1):
            self.metrics['epochs'].append(epoch)
            
            self.train_one_epoch(self.dataloader_train)
            self.validate(self.dataloader_val)
            
            if self.cfg.plot:
                plot_line(self.metrics['epochs'], [self.metrics['train_loss'], self.metrics['val_loss']], ['Training', 'Validation'], 'Epoch Number', 'Loss', self.cfg)
                plot_line(self.metrics['epochs'], [self.metrics['train_acc'], self.metrics['val_acc']], ['Training', 'Validation'], 'Epoch Number', 'Accuracy', self.cfg)
                plot_line(self.metrics['epochs'], [self.metrics['val_entropy'], self.metrics['val_entropy_rand']], ['Inputs', 'Random Inputs'], 'Epoch Number', 'Entropy', self.cfg)
            
            if self.metrics['val_loss'][-1] < self.best_loss:
                self.best_loss = self.metrics['val_loss'][-1]
                print('New best model at epoch {:0=3d} with val_loss {:.4f}'.format(epoch, self.best_loss))
                utils.flush()
                
                if self.cfg.save_model:
                    # save model when validation loss improves
                    save_name = '{}-net_{}_epoch{:0=3d}_val_loss{:.4f}'.format(self.cfg.nn_type, self.cfg.model_name, epoch, self.best_loss)
                    torch.save(self.net.state_dict(), os.path.join(self.cfg.model_dir, self.cfg.nn_type, self.cfg.model_name, '{}.pth'.format(save_name)))
                    with open(os.path.join(self.cfg.model_dir, self.cfg.nn_type, self.cfg.model_name, '{}-net_{}.txt'.format(self.cfg.nn_type, self.cfg.model_name)), 'w') as file:
                        file.write('{}.pth'.format(save_name))
    
    def train_one_epoch(self, dataloader, is_val_set=False):
        self.net.train()
        
        prefix = self.get_prefix(is_val_set)
        metrics_epoch = collections.defaultdict(utils.AverageMeter)
        matrix = np.zeros((self.c_dim, self.c_dim), dtype=np.uint32)
        for i, (x, y) in enumerate(dataloader):
            x, y = x.to(self.device), y.to(self.device)
            
            self.optimizer.zero_grad()
            logits = self.net(x)
            loss = self.criterion(logits, y)
            loss.backward()
            self.optimizer.step()
            
            matrix = matrix + utils.confusion_matrix(utils.get_class_outputs(logits), y, self.c_dim)
            metrics_epoch['{}_loss'.format(prefix)].update(loss.item(), x.shape[0])
        self.summarize_metrics(metrics_epoch, matrix, prefix)
    
    def validate(self, dataloader, is_val_set=True, measure_entropy=True):
        self.net.eval()
        
        prefix = self.get_prefix(is_val_set)
        metrics_epoch = collections.defaultdict(utils.AverageMeter)
        matrix = np.zeros((self.c_dim, self.c_dim), dtype=np.uint32)
        with torch.no_grad():
            for i, (x, y) in enumerate(dataloader):
                x, y = x.to(self.device), y.to(self.device)
                
                logits = self.net(x)
                loss = self.criterion(logits, y)
                
                matrix = matrix + utils.confusion_matrix(utils.get_class_outputs(logits), y, self.c_dim)
                metrics_epoch['{}_loss'.format(prefix)].update(loss.item(), x.shape[0])
                
                if measure_entropy:
                    probs = utils.get_class_probs(logits)
                    entropy = utils.entropy(probs)
                    
                    x_rand = (torch.rand_like(x) - 0.5) / 0.5
                    logits_rand = self.net(x_rand)
                    probs_rand = utils.get_class_probs(logits_rand)
                    entropy_rand = utils.entropy(probs_rand)
                    
                    metrics_epoch['{}_entropy'.format(prefix)].update(entropy.item(), x.shape[0])
                    metrics_epoch['{}_entropy_rand'.format(prefix)].update(entropy_rand.item(), x.shape[0])
        self.summarize_metrics(metrics_epoch, matrix, prefix)
    
    @staticmethod
    def get_prefix(is_val_set):
        if is_val_set:
            prefix = 'val'
        else:
            prefix = 'train'
        return prefix
    
    def summarize_metrics(self, metrics_epoch, matrix, prefix):
        for key in sorted(metrics_epoch.keys()):
            self.metrics[key].append(metrics_epoch[key].avg)
            print('epoch{:0=3d}_{}{:.4f}'.format(self.metrics['epochs'][-1], key, self.metrics[key][-1]))
        print(matrix)
        self.metrics['{}_acc'.format(prefix)].append(utils.calculate_acc(matrix))
        print('epoch{:0=3d}_{}{:.4f}'.format(self.metrics['epochs'][-1], '{}_acc'.format(prefix), self.metrics['{}_acc'.format(prefix)][-1]))
        utils.flush()


def main(cfg):
    # setting up output directories, and writing to stdout
    utils.make_dirs(os.path.join(cfg.stdout_dir, cfg.nn_type), replace=False)
    sys.stdout = open(r'./{}/{}/stdout_{}_{}.txt'.format(cfg.stdout_dir, cfg.nn_type, cfg.nn_type, cfg.model_name), 'w')
    print(cfg)
    utils.flush()
    
    if cfg.plot:
        utils.make_dirs(os.path.join(cfg.plot_dir, cfg.nn_type, cfg.model_name), replace=True)
    if cfg.save_model:
        utils.make_dirs(os.path.join(cfg.model_dir, cfg.nn_type, cfg.model_name), replace=True)
    
    # set random seed
    if cfg.random_seed != 0:
        random_seed = cfg.random_seed
    else:
        random_seed = random.randint(1, 10000)
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    
    # set device as cuda or cpu
    if cfg.use_gpu and torch.cuda.is_available():
        # reproducibility using cuda
        torch.cuda.manual_seed(random_seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        if cfg.use_gpu:
            print('gpu option was set to <True>, but no cuda device was found')
            utils.flush()
        
    trainer = Trainer(cfg, device)
    trainer.train()


if __name__ == '__main__':
    cfg, unparsed = get_config()
    main(cfg)
