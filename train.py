import sys
import os
from argparse import Namespace
import datetime
import collections
import copy
import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from torchvision.datasets import MNIST, CIFAR10
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from config.config import get_config
import activations
import layers
import loss_fns
from models import FCNet, ConvNet
from plotting import plot_line, plot_hist
from db import dbTiny
import utils


class Trainer():
    def __init__(self, cfg):
        self.cfg = cfg
        
        self.db = dbTiny.init_db(self.cfg.db_path)
        self.init_post()
        
        self.device = torch.device(self.cfg.device)
        
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
        # train_transforms.extend([
        #                          ])
        
        self.dataset_train = self.dataset(root=self.data_path, train=True, download=True,
                                          transform=transforms.Compose(self.train_transforms),
                                          target_transform=None)
        self.dataloader_train = DataLoader(dataset=self.dataset_train, batch_size=self.cfg.batch_size, shuffle=self.cfg.shuffle,
                                           num_workers=self.cfg.num_workers, pin_memory=True, drop_last=False)
        
        self.dataset_val = self.dataset(root=self.data_path, train=False, download=True,
                                        transform=transforms.Compose(self.val_transforms),
                                        target_transform=None)
        self.dataloader_val = DataLoader(dataset=self.dataset_val, batch_size=self.cfg.batch_size, shuffle=False,
                                         num_workers=self.cfg.num_workers, pin_memory=True, drop_last=False)
        
        # number of output classes
        targets = np.asarray(self.dataset_train.targets)
        self.c_dim = np.unique(targets).shape[0]

        # entropy threshold (arbitrary value right now of (1 - 1/e) * h_max) for training with random inputs
        self.max_ent = math.log(self.c_dim)
        # self.thresh_ent = (1. - 1. / math.e) * self.max_ent
        self.thresh_ent = self.cfg.train_random * self.max_ent
        # self.thresh_ent = self.max_ent / math.e
        
        # define model
        # parameters for each hidden layer is passed in as an argument
        self.activation = getattr(activations, self.cfg.activation.lower())
        if self.cfg.nn_type.lower()  == 'fc':
            self.params = utils.read_params(self.cfg.fc_params)
            if self.cfg.norm.lower() == 'batch':
                self.norm = nn.BatchNorm1d
            elif self.cfg.norm.lower() == 'layer':
                self.norm = layers.LayerNorm1d
            else:
                self.norm = None
            net = FCNet
        else:
            self.params = utils.read_params(self.cfg.conv_params)
            if self.cfg.norm.lower() == 'batch':
               self.norm = nn.BatchNorm2d
            elif self.cfg.norm.lower() == 'layer':
                self.norm = layers.LayerNorm2d
            else:
                self.norm = None
            net = ConvNet
        self.net = net(self.img_size, self.c_dim, self.params, self.activation, self.norm).to(self.device)
        self.post['params'] = self.params
        
        # # weight initialization - if not specified, weights are initialized using Kaiming uniform (He) initialization by default        
        # self.net.apply(layers.weights_init, self.cfg.weights_init.lower())
        
        self.criterion = loss_fns.cross_entropy_loss
        
        if self.cfg.optim.lower() == 'sgd':
            self.optimizer = optim.SGD(params=self.net.parameters(), lr=self.cfg.lr, momentum=self.cfg.momentum, nesterov=self.cfg.nesterov)
            self.post['momentum'] = self.cfg.momentum
            self.post['nesterov'] = self.cfg.nesterov
        else:
            self.optimizer = optim.Adam(params=self.net.parameters(), lr=self.cfg.lr, betas=(self.cfg.beta1, self.cfg.beta2))
            self.post['beta1'] = self.cfg.beta1
            self.post['beta2'] = self.cfg.beta2
    
    def train(self):
        # tracking training and validation stats over epochs
        self.metrics = collections.defaultdict(list)
        self.metrics['epochs'].append(0)
        
        # best model is defined as model with best performing validation loss
        self.best_loss = float('inf')
        
        # # fixed noise input
        # self.fixed_noise = torch.randn(size=(self.cfg.batch_size, *self.img_size)).to(self.device)
        
        # measure performance before any training is done
        self.validate(self.dataloader_train, is_val_set=False, measure_entropy=True)
        self.validate(self.dataloader_val, is_val_set=True, measure_entropy=True)
        
        # save initial weights
        self.save_model(epoch=0)
        
        for epoch in range(1, self.cfg.epochs+1):
            self.metrics['epochs'].append(epoch)
            
            self.train_one_epoch(self.dataloader_train)
            self.validate(self.dataloader_train, is_val_set=False, measure_entropy=True)
            self.validate(self.dataloader_val, is_val_set=True, measure_entropy=True)
            
            if self.cfg.plot:
                plot_line(self.metrics['epochs'], [self.metrics['train_loss_avg'], self.metrics['val_loss_avg']], [self.metrics['train_loss_std'], self.metrics['val_loss_std']], ['Training', 'Validation'], 'Epoch Number', 'Loss', self.cfg)
                plot_line(self.metrics['epochs'], [self.metrics['train_acc'], self.metrics['val_acc']], None, ['Training', 'Validation'], 'Epoch Number', 'Accuracy', self.cfg)
                plot_line(self.metrics['epochs'], [self.metrics['train_entropy_avg'], self.metrics['val_entropy_avg'], self.metrics['entropy_rand_avg']], [self.metrics['train_entropy_std'], self.metrics['val_entropy_std'], self.metrics['entropy_rand_std']], ['Training', 'Validation', 'Random'], 'Epoch Number', 'Entropy', self.cfg)
            
            if self.metrics['val_loss_avg'][-1] < self.best_loss:
                self.best_loss = self.metrics['val_loss_avg'][-1]
                print('New best model at epoch {:0=3d} with val_loss {:.4f}'.format(epoch, self.best_loss))
                utils.flush()
            
            self.save_model(epoch)
            self.update_post()
        dbTiny.insert(self.db, self.post)
    
    def save_model(self, epoch):
        if self.cfg.save_model:
            save_name = '{}-net_{}_epoch{:0=3d}_val_loss{:.4f}'.format(self.cfg.nn_type, self.cfg.model_name, epoch, self.metrics['val_loss_avg'][-1])
            torch.save(self.net.state_dict(), os.path.join(self.cfg.model_dir, self.cfg.nn_type, self.cfg.model_name, '{}.pth'.format(save_name)))
            with open(os.path.join(self.cfg.model_dir, self.cfg.nn_type, self.cfg.model_name, '{}-net_{}.txt'.format(self.cfg.nn_type, self.cfg.model_name)), 'w') as file:
                file.write('{}.pth'.format(save_name))
    
    def train_one_epoch(self, dataloader):
        self.net.train()
        
        for i, (x, y) in enumerate(dataloader):
            x, y_one_hot = x.to(self.device), utils.to_one_hot(y, self.c_dim).to(self.device)
            if self.cfg.train_random and (i+1) % 10 == 0:
                with torch.no_grad():
                    x_rand = torch.randn(size=x.shape).to(self.device)
                    logits_rand = self.net(x_rand)
                    entropy_rand = utils.entropy(logits_rand)
                if torch.mean(entropy_rand).item() <= self.thresh_ent:
                    print('training on random inputs & random labels for minibatch {}'.format(i))
                    # x = (torch.rand(size=x.shape).to(self.device) - 0.5) / 0.5
                    x = torch.randn(size=x.shape).to(self.device)
                    y_one_hot = torch.ones(size=(x.shape[0], self.c_dim)).to(self.device) / self.c_dim
            
            self.optimizer.zero_grad()
            logits = self.net(x)
            losses = self.criterion(logits, y_one_hot)
            torch.mean(losses).backward()
            self.optimizer.step()
    
    def validate(self, dataloader, is_val_set=True, measure_entropy=True):
        self.net.eval()
        
        prefix = self.get_prefix(is_val_set)
        self.metrics_epoch = collections.defaultdict(utils.Meter)
        matrix = np.zeros((self.c_dim, self.c_dim), dtype=np.uint32)
        with torch.no_grad():
            for i, (x, y) in enumerate(dataloader):
                x, y_one_hot = x.to(self.device), utils.to_one_hot(y, self.c_dim).to(self.device)
                y = y.to(self.device)
                
                logits = self.net(x)
                losses = self.criterion(logits, y_one_hot)
                
                matrix = matrix + utils.confusion_matrix(utils.get_class_outputs(logits).cpu().detach().numpy(), y.cpu().detach().numpy(), self.c_dim)
                self.metrics_epoch['{}_loss'.format(prefix)].update(losses.cpu().detach().numpy(), x.shape[0])
                
                if measure_entropy:
                    entropy = utils.entropy(logits)
                    self.metrics_epoch['{}_entropy'.format(prefix)].update(entropy.cpu().detach().numpy(), x.shape[0])
                    
                    if is_val_set:
                        # x_rand = (torch.rand(size=x.shape).to(self.device) - 0.5) / 0.5
                        x_rand = torch.randn(size=x.shape).to(self.device)
                        logits_rand = self.net(x_rand)
                        entropy_rand = utils.entropy(logits_rand)
                        self.metrics_epoch['entropy_rand'].update(entropy_rand.cpu().detach().numpy(), x.shape[0])
        self.summarize_metrics(matrix, prefix)
    
    @staticmethod
    def get_prefix(is_val_set):
        if is_val_set:
            prefix = 'val'
        else:
            prefix = 'train'
        return prefix
    
    def summarize_metrics(self, matrix, prefix):
        for key in sorted(self.metrics_epoch.keys()):
            self.metrics['{}_{}'.format(key, 'avg')].append(self.metrics_epoch[key].avg)
            self.metrics['{}_{}'.format(key, 'std')].append(self.metrics_epoch[key].std)
            print('epoch{:0=3d}_{}{:.4f}'.format(self.metrics['epochs'][-1], '{}_{}'.format(key, 'avg'), self.metrics['{}_{}'.format(key, 'avg')][-1]))
            print('epoch{:0=3d}_{}{:.4f}'.format(self.metrics['epochs'][-1], '{}_{}'.format(key, 'std'), self.metrics['{}_{}'.format(key, 'std')][-1]))
        print(matrix)
        self.metrics['{}_acc'.format(prefix)].append(utils.calculate_acc(matrix))
        print('epoch{:0=3d}_{}{:.4f}'.format(self.metrics['epochs'][-1], '{}_acc'.format(prefix), self.metrics['{}_acc'.format(prefix)][-1]))
        utils.flush()
    
    # TODO: modularize
    def init_post(self):
        last_run = dbTiny.get_last(self.db, 'run')
        if last_run:
            run = last_run + 1
        else:
            run = 1
        self.post = {'run': run}
        
        self.post['model_name'] = self.cfg.model_name.lower()
        
        self.post['nn_type'] = self.cfg.nn_type.lower()
        
        self.post['activation'] = self.cfg.activation.lower()
        self.post['norm'] = self.cfg.norm.lower()
        self.post['weights_init'] = self.cfg.weights_init.lower()
        
        self.post['dataset'] = self.cfg.dataset.lower()
        self.post['normalize_input'] = self.cfg.normalize_input
        
        self.post['epochs'] = self.cfg.epochs
        self.post['batch_size'] = self.cfg.batch_size
        self.post['optim'] = self.cfg.optim.lower()
        self.post['lr'] = self.cfg.lr
        self.post['shuffle'] = self.cfg.shuffle
        
        self.post['num_workers'] = self.cfg.num_workers
        self.post['device'] = self.cfg.device.lower()
        self.post['random_seed'] = self.cfg.random_seed
        
        self.post['timestamp'] = self.cfg.time
    
    # TODO: modularize
    def update_post(self):
        self.post['train_loss_avg'] = self.metrics['train_loss_avg']
        self.post['train_loss_std'] = self.metrics['train_loss_std']
        self.post['val_loss_avg'] = self.metrics['val_loss_avg']
        self.post['val_loss_std'] = self.metrics['val_loss_std']
        
        self.post['train_acc'] = self.metrics['train_acc']
        self.post['val_acc'] = self.metrics['val_acc']
        
        best_epoch_train_loss = int(np.argmin(np.asarray(self.metrics['train_loss_avg'])))
        best_epoch_train_acc = int(np.argmax(np.asarray(self.metrics['train_acc'])))
        best_epoch_val_loss = int(np.argmin(np.asarray(self.metrics['val_loss_avg'])))
        best_epoch_val_acc = int(np.argmax(np.asarray(self.metrics['val_acc'])))
        
        self.post['best_epoch_train_loss'] = best_epoch_train_loss
        self.post['best_epoch_train_acc'] = best_epoch_train_acc
        self.post['best_epoch_val_loss'] = best_epoch_val_loss
        self.post['best_epoch_val_acc'] = best_epoch_val_acc
        
        self.post['train_loss_at_best_train_loss'] = self.metrics['train_loss_avg'][best_epoch_train_loss]
        self.post['train_acc_at_best_train_loss'] = self.metrics['train_acc'][best_epoch_train_loss]
        self.post['val_loss_at_best_train_loss'] = self.metrics['val_loss_avg'][best_epoch_train_loss]
        self.post['val_acc_at_best_train_loss'] = self.metrics['val_acc'][best_epoch_train_loss]
        
        self.post['train_loss_at_best_train_acc'] = self.metrics['train_loss_avg'][best_epoch_train_acc]
        self.post['train_acc_at_best_train_acc'] = self.metrics['train_acc'][best_epoch_train_acc]
        self.post['val_loss_at_best_train_acc'] = self.metrics['val_loss_avg'][best_epoch_train_acc]
        self.post['val_acc_at_best_train_acc'] = self.metrics['val_acc'][best_epoch_train_acc]
        
        self.post['train_loss_at_best_val_loss'] = self.metrics['train_loss_avg'][best_epoch_val_loss]
        self.post['train_acc_at_best_val_loss'] = self.metrics['train_acc'][best_epoch_val_loss]
        self.post['val_loss_at_best_val_loss'] = self.metrics['val_loss_avg'][best_epoch_val_loss]
        self.post['val_acc_at_best_val_loss'] = self.metrics['val_acc'][best_epoch_val_loss]
        
        self.post['train_loss_at_best_val_acc'] = self.metrics['train_loss_avg'][best_epoch_val_acc]
        self.post['train_acc_at_best_val_acc'] = self.metrics['train_acc'][best_epoch_val_acc]
        self.post['val_loss_at_best_val_acc'] = self.metrics['val_loss_avg'][best_epoch_val_acc]
        self.post['val_acc_at_best_val_acc'] = self.metrics['val_acc'][best_epoch_val_acc]
        
        self.post['train_entropy_avg'] = self.metrics['train_entropy_avg']
        self.post['train_entropy_std'] = self.metrics['train_entropy_std']
        self.post['val_entropy_avg'] = self.metrics['val_entropy_avg']
        self.post['val_entropy_std'] = self.metrics['val_entropy_std']
        
        self.post['entropy_rand_avg'] = self.metrics['entropy_rand_avg']
        self.post['entropy_rand_std'] = self.metrics['entropy_rand_std']


def main(cfg):
    current_time = utils.get_current_time()
    
    # override default-config parameters, with command-line-provided parameters
    preset_cfg = utils.load_json(cfg.config)
    cfg_json = vars(cfg)
    for key in cfg_json:
        if cfg_json[key] is not None:
            preset_cfg[key] = cfg_json[key]
    cfg = Namespace(**preset_cfg)
    
    cfg_json = vars(cfg)
    utils.make_dirs('./config/save/', replace=False)
    utils.save_json(cfg_json, './config/save/config_{}.json'.format(current_time))
    
    cfg.time = current_time
    cfg.model_name = '{}_{}'.format(cfg.model_name, current_time)
    
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
    if cfg.random_seed == 0:
        cfg.random_seed = random.randint(1, 10000)
        print('random seed set to {}'.format(cfg.random_seed))
        utils.flush()
    random.seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)
    torch.manual_seed(cfg.random_seed)
    
    # set device as cuda or cpu
    if cfg.device.lower() == 'cuda' and torch.cuda.is_available():
        # reproducibility using cuda
        torch.cuda.manual_seed(cfg.random_seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:
        if cfg.device.lower() == 'cuda':
            print('device option was set to <cuda>, but no cuda device was found')
            utils.flush()
            cfg.device = 'cpu'
    
    trainer = Trainer(cfg)
    trainer.train()


if __name__ == '__main__':
    cfg, unparsed = get_config()
    main(cfg)
