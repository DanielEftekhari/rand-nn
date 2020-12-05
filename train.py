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


def main(cfg):
    # setting up output directories, and writing to stdout
    utils.make_dirs(os.path.join(cfg.stdout_dir, cfg.nn_type), replace=False)
    sys.stdout = open('{}/{}/stdout_{}_{}.txt'.format(cfg.stdout_dir, cfg.nn_type, cfg.nn_type, cfg.model_name), 'w')
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
    
    if cfg.dataset.lower() == 'mnist':
        dataset = MNIST
        data_path = r'./data/mnist'
        img_size = [1, 28, 28]
        normalize = [(0.1307,), (0.3081,)]
    else:
        dataset = CIFAR10
        data_path = r'./data/cifar10'
        img_size = [3, 32, 32]
        normalize = [(0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)]
    
    # datasets and dataloaders
    # in training, <drop_last> minibatch in an epoch set to <True> for simplicity in tracking training performance
    
    # base transforms
    train_transforms = [transforms.ToTensor()]
    if cfg.normalize_input:
        train_transforms.append(transforms.Normalize(normalize[0], normalize[1]))
    val_transforms = copy.deepcopy(train_transforms)
    
    # (if applicable) additional training set transforms defined here
    # train_transforms.extend([])
    
    dataset_train = dataset(root=data_path, train=True, download=True,
                            transform=transforms.Compose(train_transforms),
                            target_transform=None)
    dataloader_train = DataLoader(dataset=dataset_train, batch_size=cfg.batch_size, shuffle=cfg.shuffle,
                                  num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    
    dataset_val = dataset(root=data_path, train=False, download=True,
                          transform=transforms.Compose(val_transforms),
                          target_transform=None)
    dataloader_val = DataLoader(dataset=dataset_val, batch_size=100, shuffle=False,
                                num_workers=cfg.num_workers, pin_memory=True, drop_last=False)
    
    # number of output classes
    targets = np.asarray(dataset_train.targets)
    c = np.unique(targets).shape[0]

    # define model
    # weights initialized using Kaiming uniform (He initialization)
    # parameters for each hidden layer is passed in as an argument
    activation = getattr(activations, cfg.activation.lower())
    if 'fc' in cfg.nn_type.lower():
        if cfg.norm.lower() == 'batch':
            norm = nn.BatchNorm1d
        elif cfg.norm.lower() == 'layer':
            norm = layers.LayerNorm1d
        else:
            norm = None
        net = FCNet(np.product(img_size), c, cfg.fc_params, activation, norm, cfg).to(device)
    else:
        if cfg.norm.lower() == 'batch':
            norm = nn.BatchNorm2d
        elif cfg.norm.lower() == 'layer':
            norm = layers.LayerNorm2d
        else:
            norm = None
        net = ConvNet(img_size, c, cfg.conv_params, activation, norm, cfg).to(device)
    
    criterion = nn.CrossEntropyLoss()
    
    if cfg.use_sgd:
        optimizer = optim.SGD(params=net.parameters(), lr=cfg.lr, momentum=cfg.momentum, nesterov=cfg.use_nesterov)
    else:
        optimizer = optim.Adam(params=net.parameters(), lr=cfg.lr, betas=(cfg.beta1, cfg.beta2))
    
    # tracking training and validation stats over epochs
    metrics = collections.defaultdict(list)
    
    # best model is defined as model with best performing validation loss
    best_loss = float('inf')
    for epoch in range(cfg.epochs):
        # tracking training and validation stats over a given epoch
        metrics_epoch = collections.defaultdict(list)
        matrix = np.zeros((c, c), dtype=np.uint32)
        metrics['epochs'].append(epoch+1)
        
        # training set
        net.train()
        
        for i, (x, y) in enumerate(dataloader_train):
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            logits = net(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            
            matrix = matrix + utils.confusion_matrix(utils.get_class_outputs(logits), y, c)
            utils.append((metrics_epoch['train_loss'], loss.item()),
                         )
        
        for key in sorted(metrics_epoch.keys()):
            if 'train_loss' in key:
                metrics[key].append(utils.get_average(metrics_epoch[key]))
                print('train_epoch{:0=3d}_{}{:.4f}'.format(epoch+1, key, metrics[key][-1]))
        print(matrix)
        metrics['train_acc'].append(utils.calculate_metrics(matrix))
        print('train_epoch{:0=3d}_{}{:.4f}'.format(epoch + 1, 'train_acc', metrics['train_acc'][-1]))
        utils.flush()
        
        # validation set
        net.eval()
        
        matrix = np.zeros((c, c), dtype=np.uint32)
        with torch.no_grad():
            for i, (x, y) in enumerate(dataloader_val):
                x, y = x.to(device), y.to(device)
                
                logits = net(x)
                loss = criterion(logits, y)
                
                matrix = matrix + utils.confusion_matrix(utils.get_class_outputs(logits), y, c)
                utils.append((metrics_epoch['val_loss'], loss.item()),
                             )
        
        for key in sorted(metrics_epoch.keys()):
            if 'val_loss' in key:
                metrics[key].append(utils.get_average(metrics_epoch[key]))
                print('val_epoch{:0=3d}_{}{:.4f}'.format(epoch + 1, key, metrics[key][-1]))
        print(matrix)
        metrics['val_acc'].append(utils.calculate_metrics(matrix))
        print('val_epoch{:0=3d}_{}{:.4f}'.format(epoch + 1, 'val_acc', metrics['val_acc'][-1]))
        utils.flush()
        
        if cfg.plot:
            plot_line(metrics['epochs'], metrics['train_loss'], metrics['val_loss'], 'Epoch Number', 'Loss', cfg)
            plot_line(metrics['epochs'], metrics['train_acc'], metrics['val_acc'], 'Epoch Number', 'Accuracy', cfg)
        
        if metrics['val_loss'][-1] < best_loss:
            best_loss = metrics['val_loss'][-1]
            print('New best model at epoch {:0=3d} with val_loss {:.4f}'.format(epoch+1, best_loss))
            utils.flush()
            
            if cfg.save_model:
                # save model when validation loss improves
                save_name = '{}-net_{}_epoch{:0=3d}_val_loss{:.4f}'.format(cfg.nn_type, cfg.model_name, epoch+1, best_loss)
                torch.save(net.state_dict(), os.path.join(cfg.model_dir, cfg.nn_type, cfg.model_name, '{}.pth'.format(save_name)))
                with open(os.path.join(cfg.model_dir, cfg.nn_type, cfg.model_name, '{}-net_{}.txt'.format(cfg.nn_type, cfg.model_name)), 'w') as file:
                    file.write('{}.pth'.format(save_name))


if __name__ == '__main__':
    cfg, unparsed = get_config()
    main(cfg)
