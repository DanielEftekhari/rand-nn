import sys
import os
from argparse import Namespace
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

import config.utils as cutils
from config.config import get_config
import activations
import layers
import loss_fns
from models import FCNet, ConvNet
from hooks import Hook
import plotting
import db.utils as dutils
import metrics
import utils


class Trainer():
    def __init__(self, cfg):
        self.cfg = cfg
        
        self.db = dutils.init_db(self.cfg.db_path)
        self.init_post()
        
        self.device = torch.device(self.cfg.device)
        
        # dataset parameters
        if self.cfg.dataset.lower() == 'mnist':
            self.dataset = MNIST
            self.data_path = self.cfg.data_dir + 'mnist'
            self.img_size = [1, 28, 28]
            self.normalize = [(0.1307,), (0.3081,)]
        elif self.cfg.dataset.lower() == 'cifar10':
            self.dataset = CIFAR10
            self.data_path = self.cfg.data_dir + 'cifar10'
            self.img_size = [3, 32, 32]
            self.normalize = [(0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)]
        else:
            raise NotImplementedError()
        
        # datasets and dataloaders
        # base transforms
        self.train_transforms = [transforms.ToTensor()]
        if self.cfg.normalize_input:
            self.train_transforms.append(transforms.Normalize(self.normalize[0], self.normalize[1]))
        self.val_transforms = copy.deepcopy(self.train_transforms)
        
        # # (if applicable) additional training set transforms defined here
        # train_transforms.extend([
        #                          ])
        
        self.dataset_train = self.dataset(root=self.data_path, train=True, download=True,
                                          transform=transforms.Compose(self.train_transforms),
                                          target_transform=None)
        self.dataloader_train = DataLoader(dataset=self.dataset_train, batch_size=self.cfg.batch_size, shuffle=self.cfg.shuffle,
                                           num_workers=self.cfg.num_workers, pin_memory=True, drop_last=False)
        
        # number of output classes (based only on training data)
        self.c_dim = len(torch.unique(self.dataset_train.targets))
        
        self.dataset_val = self.dataset(root=self.data_path, train=False, download=True,
                                        transform=transforms.Compose(self.val_transforms),
                                        target_transform=None)
        self.dataloader_val = DataLoader(dataset=self.dataset_val, batch_size=self.cfg.batch_size, shuffle=False,
                                         num_workers=self.cfg.num_workers, pin_memory=True, drop_last=False)
        
        # maximum entropy threshold for training with random inputs
        self.max_entropy = metrics.max_entropy(self.c_dim)
        self.thresh_entropy = self.cfg.train_random * self.max_entropy
        
        # define model
        # parameters for each hidden layer is passed in as an argument
        self.params = utils.read_params(self.cfg.model_params[self.cfg.model_type])
        self.activation = getattr(activations, self.cfg.activation.lower())
        if self.cfg.model_type.lower()  == 'fc':
            if self.cfg.norm.lower() == 'batch':
                self.norm = nn.BatchNorm1d
            elif self.cfg.norm.lower() == 'layer':
                self.norm = layers.LayerNorm1d
            else:
                self.norm = None
            net = FCNet
        elif self.cfg.model_type.lower()  == 'conv':
            if self.cfg.norm.lower() == 'batch':
               self.norm = nn.BatchNorm2d
            elif self.cfg.norm.lower() == 'layer':
                self.norm = layers.LayerNorm2d
            else:
                self.norm = None
            net = ConvNet
        else:
            raise NotImplementedError()
        self.net = net(self.img_size, self.c_dim, self.params, self.activation, self.norm).to(self.device)
        self.post['params'] = self.params
        
        # TODO: add custom weight initialization scheme
        # # weight initialization - weights are initialized using Kaiming uniform (He) initialization by default
        
        # loss function <kl_y_to_p> generalizes the cross entropy loss to continuous label distributions
        # i.e. <kl_y_to_p> is equivalent to <cross_entropy_loss> for one-hot labels
        # but is also a sensible loss function for continuous label distributions
        self.criterion = loss_fns.kl_y_to_p
        
        if self.cfg.optim.lower() == 'sgd':
            self.optimizer = optim.SGD(params=self.net.parameters(), lr=self.cfg.lr, momentum=self.cfg.optim_params['sgd']['momentum'], nesterov=self.cfg.optim_params['sgd']['nesterov'])
            self.post['momentum'], self.post['nesterov'] = self.cfg.optim_params['sgd']['momentum'], self.cfg.optim_params['sgd']['nesterov']
        else:
            self.optimizer = optim.Adam(params=self.net.parameters(), lr=self.cfg.lr, betas=(self.cfg.optim_params['adam']['beta1'], self.cfg.optim_params['adam']['beta2']))
            self.post['beta1'], self.post['beta2'] = self.cfg.optim_params['adam']['beta1'], self.cfg.optim_params['adam']['beta2']
    
    def train(self):
        # tracking training and validation stats over epochs
        self.metrics = collections.defaultdict(list)
        self.metrics['epochs'].append(0)
        
        # best model is defined as model with best performing (lowest) validation loss
        self.best_loss = float('inf')
        
        # # fixed noise input -> can be used to benchmark output class entropy for random inputs
        # self.fixed_noise = torch.randn(size=(self.cfg.batch_size, *self.img_size)).to(self.device)
        
        # register hooks
        self.hook = Hook(self.cfg.num_log > 0)
        self.hook.init_hook(self.net.names, self.net.layers)
        
        # measure performance before any training is done
        with torch.no_grad():
            self.validate(self.dataloader_train, is_val_set=False, measure_entropy=True)
            self.validate(self.dataloader_val, is_val_set=True, measure_entropy=True)
        
        # save initial weights
        self.eval_best_model(epoch=0)
        self.save_model(epoch=0)
        
        for epoch in range(1, self.cfg.epochs+1):
            self.metrics['epochs'].append(epoch)
            
            self.hook.clear_hook()
            self.train_one_epoch(self.dataloader_train)
            
            self.hook.init_hook(self.net.names, self.net.layers)
            with torch.no_grad():
                self.validate(self.dataloader_train, is_val_set=False, measure_entropy=True)
                self.validate(self.dataloader_val, is_val_set=True, measure_entropy=True)
            
            if self.cfg.plot:
                plotting.plot_line(self.metrics['epochs'],
                                   [self.metrics['train_loss_avg'], self.metrics['val_loss_avg']],
                                   [self.metrics['train_loss_std'], self.metrics['val_loss_std']],
                                   ['Training', 'Validation'],
                                   'Epoch Number', 'Loss', self.cfg)
                plotting.plot_line(self.metrics['epochs'],
                                   [self.metrics['train_acc'], self.metrics['val_acc']],
                                   None,
                                   ['Training', 'Validation'],
                                   'Epoch Number', 'Accuracy', self.cfg)
                plotting.plot_line(self.metrics['epochs'],
                                   [self.metrics['train_entropy_avg'], self.metrics['val_entropy_avg'], self.metrics['entropy_rand_avg']],
                                   [self.metrics['train_entropy_std'], self.metrics['val_entropy_std'], self.metrics['entropy_rand_std']],
                                   ['Training', 'Validation', 'Random'],
                                   'Epoch Number', 'Entropy', self.cfg)
            
            self.eval_best_model(epoch)
            self.save_model(epoch)
            self.update_post()
        dutils.insert(self.db, self.post)
    
    def eval_best_model(self, epoch):
        if self.metrics['val_loss_avg'][-1] < self.best_loss:
            self.best_loss = self.metrics['val_loss_avg'][-1]
            print('New best model at epoch {:0=3d} with val_loss {:.4f}'.format(epoch, self.best_loss))
            utils.flush()
    
    def save_model(self, epoch):
        if self.cfg.save_model:
            save_name = '{}-net_{}_epoch{:0=3d}_val_loss{:.4f}'.format(self.cfg.model_type, self.cfg.model_name, epoch, self.metrics['val_loss_avg'][-1])
            torch.save(self.net.state_dict(), os.path.join(self.cfg.model_dir, self.cfg.model_type, self.cfg.model_name, '{}.pth'.format(save_name)))
            if self.best_loss == self.metrics['val_loss_avg'][-1]:
                with open(os.path.join(self.cfg.model_dir, self.cfg.model_type, self.cfg.model_name, '{}-net_{}.txt'.format(self.cfg.model_type, self.cfg.model_name)), 'w') as file:
                    file.write('{}.pth'.format(save_name))
    
    def train_one_epoch(self, dataloader):
        self.net.train()
        self.hook.flag_hook = False
        
        for mb, (x, y) in enumerate(dataloader):
            x, y = x.to(self.device), y.to(self.device)
            y_one_hot = utils.to_one_hot(y, self.c_dim)
            
            if self.cfg.train_random > 0 and (mb+1) % 10 == 0:
                with torch.no_grad():
                    x_rand = torch.randn(size=x.shape).to(self.device)
                    logits_rand = self.net(x_rand)
                    entropy_rand = metrics.entropy(utils.logits_to_probs(logits_rand))
                if torch.mean(entropy_rand).item() <= self.thresh_entropy:
                    print('training on random inputs & random labels for minibatch {}'.format(mb+1))
                    x = torch.randn(size=x.shape).to(self.device)
                    y_one_hot = torch.ones(size=(x.shape[0], self.c_dim)).to(self.device) / self.c_dim
            
            self.optimizer.zero_grad()
            logits = self.net(x)
            losses = self.criterion(logits, y_one_hot)
            torch.mean(losses).backward()
            self.optimizer.step()
    
    def validate(self, dataloader, is_val_set=True, measure_entropy=True):
        self.net.eval()
        self.hook.flag_hook = True
        
        prefix = self.get_prefix(is_val_set)
        self.metrics_epoch = collections.defaultdict(utils.Meter)
        matrix = np.zeros((self.c_dim, self.c_dim), dtype=np.uint32)
        for mb, (x, y) in enumerate(dataloader):
            x, y = x.to(self.device), y.to(self.device)
            y_one_hot = utils.to_one_hot(y, self.c_dim)
            
            logits = self.net(x)
            losses = self.criterion(logits, y_one_hot)
            
            matrix = matrix + metrics.confusion_matrix(utils.tensor2array(utils.get_class_outputs(logits)), utils.tensor2array(y), self.c_dim)
            self.metrics_epoch['{}_loss'.format(prefix)].update(utils.tensor2array(losses), x.shape[0])
            
            if self.cfg.num_log > 0 and self.cfg.plot and mb == 0:
                num_log = min(self.cfg.num_log, x.shape[0])
                name = '{}_{}_{}_epoch{:0=3d}_minibatch{}'
                filepath = '{}/{}'.format(os.path.join(self.cfg.plot_dir, self.cfg.model_type, self.cfg.model_name), name)
                x_ = x[0:num_log]
                x_np, y_np = utils.tensor2array(x[0:num_log]), utils.tensor2array(y[0:num_log])
                losses_np = utils.tensor2array(losses[0:num_log])
                
                plotting.make_grid(x_, filepath.format(prefix, 'data', 'x', self.metrics['epochs'][-1], mb+1))
                utils.save_array(x_np, filepath.format(prefix, 'data', 'x', self.metrics['epochs'][-1], mb+1))
                utils.save_array(y_np, filepath.format(prefix, 'data', 'y', self.metrics['epochs'][-1], mb+1))
                utils.save_array(losses_np, filepath.format(prefix, 'data', 'losses', self.metrics['epochs'][-1], mb+1))
                
                for (k, layer_name) in enumerate(self.hook.layers):
                    layer_np = utils.tensor2array(self.hook.layers[layer_name][0:num_log])
                    utils.save_array(layer_np, filepath.format(prefix, 'data', layer_name, self.metrics['epochs'][-1], mb+1))
            
            if measure_entropy:
                entropy = metrics.entropy(utils.logits_to_probs(logits))
                self.metrics_epoch['{}_entropy'.format(prefix)].update(utils.tensor2array(entropy), x.shape[0])
                
                if self.cfg.num_log > 0 and self.cfg.plot and mb == 0:
                    entropy_np = utils.tensor2array(entropy[0:num_log])
                    utils.save_array(entropy_np, filepath.format(prefix, 'data', 'entropy', self.metrics['epochs'][-1], mb+1))
                
                if is_val_set:
                    x_rand = torch.randn(size=x.shape).to(self.device)
                    logits_rand = self.net(x_rand)
                    entropy_rand = metrics.entropy(utils.logits_to_probs(logits_rand))
                    self.metrics_epoch['entropy_rand'].update(utils.tensor2array(entropy_rand), x.shape[0])
                    
                    if self.cfg.num_log > 0 and self.cfg.plot and mb == 0:
                        name = '{}_{}_{}_epoch{:0=3d}_minibatch{}'
                        filepath = '{}/{}'.format(os.path.join(self.cfg.plot_dir, self.cfg.model_type, self.cfg.model_name), name)
                        x_ = x_rand[0:num_log]
                        x_np = utils.tensor2array(x_rand[0:num_log])
                        entropy_np = utils.tensor2array(entropy_rand[0:num_log])
                        
                        plotting.make_grid(x_, filepath.format(prefix, 'noise', 'x', self.metrics['epochs'][-1], mb+1))
                        utils.save_array(x_np, filepath.format(prefix, 'noise', 'x', self.metrics['epochs'][-1], mb+1))
                        utils.save_array(entropy_np, filepath.format(prefix, 'noise', 'entropy', self.metrics['epochs'][-1], mb+1))
                        
                        for (k, layer_name) in enumerate(self.hook.layers):
                            layer_np = utils.tensor2array(self.hook.layers[layer_name][0:num_log])
                            utils.save_array(layer_np, filepath.format(prefix, 'noise', layer_name, self.metrics['epochs'][-1], mb+1))
            
            # disable hook after first minibatch by default - this is done for computational/speed purposes
            self.hook.flag_hook = False
        
        self.summarize_metrics(matrix, prefix)
    
    @staticmethod
    def get_prefix(is_val_set):
        if is_val_set: return 'val'
        else: return 'train'
    
    def summarize_metrics(self, matrix, prefix):
        for key in sorted(self.metrics_epoch.keys()):
            self.metrics['{}_{}'.format(key, 'avg')].append(self.metrics_epoch[key].avg)
            self.metrics['{}_{}'.format(key, 'std')].append(self.metrics_epoch[key].std)
            print('epoch{:0=3d}_{}{:.4f}'.format(self.metrics['epochs'][-1], '{}_{}'.format(key, 'avg'), self.metrics['{}_{}'.format(key, 'avg')][-1]))
            print('epoch{:0=3d}_{}{:.4f}'.format(self.metrics['epochs'][-1], '{}_{}'.format(key, 'std'), self.metrics['{}_{}'.format(key, 'std')][-1]))
        print(matrix)
        self.metrics['{}_acc'.format(prefix)].append(metrics.accuracy(matrix))
        print('epoch{:0=3d}_{}{:.4f}'.format(self.metrics['epochs'][-1], '{}_acc'.format(prefix), self.metrics['{}_acc'.format(prefix)][-1]))
        utils.flush()
    
    def init_post(self):
        last_run = dutils.get_last(self.db, 'run')
        if last_run:
            run = last_run + 1
        else:
            run = 1
        self.post = {'run': run}
        self.post['timestamp'] = self.cfg.time
        
        cfg_dict = vars(self.cfg)
        for key in cfg_dict:
            if type(cfg_dict[key]) == str:
                self.post[key] = cfg_dict[key].lower()
            elif type(cfg_dict[key]) != dict:
                self.post[key] = cfg_dict[key]
    
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
    start_time = utils.get_current_time()
    
    # override base-config parameters with arguments provided at run-time
    base_cfg_dict = utils.load_json(cfg.base_config)
    membership = cutils.get_membership(base_cfg_dict)
    
    cfg_dict = vars(cfg)
    cfg_dict = {key: cfg_dict[key] for key in cfg_dict if cfg_dict[key] is not None}
    
    updated_cfg_dict = cutils.update_params(base_cfg_dict, cfg_dict, membership)
    cfg = Namespace(**updated_cfg_dict)
    
    utils.make_dirs('./config/save/', replace=False)
    utils.save_json(updated_cfg_dict, './config/save/config_{}.json'.format(start_time))
    
    cfg.time = start_time
    cfg.model_name = '{}_{}'.format(cfg.model_name, start_time)
    
    # setting up output directories, and writing to stdout
    utils.make_dirs(os.path.join(cfg.stdout_dir, cfg.model_type), replace=False)
    sys.stdout = open(r'./{}/{}/stdout_{}_{}.txt'.format(cfg.stdout_dir, cfg.model_type, cfg.model_type, cfg.model_name), 'w')
    print(cfg)
    utils.flush()
    
    if cfg.plot:
        utils.make_dirs(os.path.join(cfg.plot_dir, cfg.model_type, cfg.model_name), replace=True)
    if cfg.save_model:
        utils.make_dirs(os.path.join(cfg.model_dir, cfg.model_type, cfg.model_name), replace=True)
    
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
