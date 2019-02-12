import pandas as pd
import numpy as np
import os
import sys

import torch
from torch.utils.data import Dataset, DataLoader

from torch import nn
import torch.nn.functional as F
import torch.optim

import click
import logging

#libpath = '/gpfs/projects/KozakovGroup/mhc_learning/mhc-adventures/ilovemhc'
#if libpath not in sys.path:
#    sys.path.append(libpath)

import ilovemhc
from ilovemhc.wrappers import *
from ilovemhc import utils
from ilovemhc import grids
from ilovemhc import dataset
from ilovemhc.engines import regression_trainer_with_tagwise_statistics, cuda_is_avail

@click.command()
@click.argument('model_dir')
@click.argument('model_name')
@click.option('-w', '--weight_decay', default=0.0001)
@click.option('-l', '--learning_rate', default=0.0001)
@click.option('-b', '--batch_size', default=128)
@click.option('-e', '--max_epochs', default=10)
@click.option('-c', '--ncores', default=1, help='N cores to use for data loading')
@click.option('-d', '--cuda_device', default="cuda:0", help='Cuda device id: e.g. "cuda:0"')
@click.option('--bin_size', default=1.0, help='Grid resolution')
def run(model_dir,
        model_name, 
        weight_decay, 
        learning_rate, 
        batch_size, 
        max_epochs,  
        ncores,
        cuda_device,
        bin_size):
    
    exec 'from ilovemhc.torch_models import %s as current_model_class' % model_name in globals(), locals()
    
    logging.info('model_dir     = {}'.format(model_dir))
    logging.info('model_name    = {}'.format(model_name))
    logging.info('weight_decay  = {}'.format(weight_decay))
    logging.info('learning_rate = {}'.format(learning_rate))
    logging.info('batch_size    = {}'.format(batch_size))
    logging.info('max_epochs    = {}'.format(max_epochs))
    logging.info('ncores        = {}'.format(ncores))
    logging.info('cuda_device   = {}'.format(cuda_device))
    logging.info('bin_size      = {}'.format(bin_size))
    
    train_csv = '../dataset/train_test/train-full-vsplit.csv'
    test_csv = '../dataset/train_test/test-full-vsplit.csv'
    root_dir = '../dataset'
    model_prefix = 'model'

    avail, device = cuda_is_avail(cuda_device)

    test_table = pd.read_csv(test_csv)
    train_table = pd.read_csv(train_csv)

    break_point = 1.2
    steepness = 3.
    buf = np.exp(steepness*break_point)
    c1 = 1./steepness*np.log(buf-2.)
    c2 = (1.-buf)/(2.-buf)
    target_scale = lambda x: c2 / (1. + np.exp((x - c1)*steepness))
    
    test_set = dataset.MolDataset(test_table, 
                                  root_dir, 
                                  bin_size=bin_size,
                                  target_transform=target_scale,
                                  add_index=True, 
                                  remove_grid=True)
    
    train_set = dataset.MolDataset(train_table, 
                                   root_dir, 
                                   bin_size=bin_size,
                                   target_transform=target_scale, 
                                   remove_grid=True)
    
    test_loader = DataLoader(dataset=test_set, 
                             batch_size=batch_size, 
                             num_workers=ncores, 
                             shuffle=False)
    
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, num_workers=ncores, shuffle=True)

    input_shape = torch.tensor(train_set[0][0].shape).numpy()
    logging.info(input_shape)
    
    model = current_model_class(input_shape)
    logging.info(model)
    
    #if avail:
    #    model = nn.DataParallel(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    logging.info(optimizer)

    loss = torch.nn.MSELoss()
    logging.info(loss)

    trainer = regression_trainer_with_tagwise_statistics(model, 
                                                         optimizer, 
                                                         loss, 
                                                         test_loader, 
                                                         test_table, 
                                                         device, 
                                                         model_dir, 
                                                         model_prefix,
                                                         every_n_iter=10)
    trainer.run(train_loader, max_epochs)
    logging.info("COMPLETED")
    
if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s', 
                    level=logging.INFO, 
                    datefmt='%m/%d/%Y %I:%M:%S %p')
    
    run()
