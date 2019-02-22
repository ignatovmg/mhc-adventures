import pandas as pd
import numpy as np
import os
import sys

import torch
from torch.utils.data import Dataset, DataLoader

from torch import nn
import torch.nn.functional as F
import torch.optim

torch.cuda.manual_seed_all(123456)
torch.manual_seed(123456)

import click
import logging

import ilovemhc
from ilovemhc.wrappers import *
from ilovemhc import utils
from ilovemhc import grids
from ilovemhc import dataset
from ilovemhc.grids import GridMaker
from ilovemhc.define import GRID_PRM_DIR
from ilovemhc.engines import make_trainer, make_evaluator, get_device
from ilovemhc.torch_models import load_model

@click.command()
@click.argument('model_dir')
@click.argument('model_name')
@click.argument('train_csv', type=click.Path(exists=True)) 
@click.argument('test_csv', type=click.Path(exists=True))
@click.argument('data_root', type=click.Path(exists=True))
@click.option('-w', '--weight_decay', default=0.0001)
@click.option('-l', '--learning_rate', default=0.0001)
@click.option('-b', '--batch_size', default=128)
@click.option('-e', '--max_epochs', default=10)
@click.option('-c', '--ncores', default=1, help='N cores to use for data loading')
@click.option('-d', '--device_name', default="cuda:0", help='Device id in torch format: e.g. "cuda:0" or "cpu"')
@click.option('--bin_size', default=1.0, help='Grid resolution')
@click.option('--ngpu', default=1, help='Number of GPUs to use')
@click.option('--use_saved', help='Path to saved model', type=click.Path(exists=True))
@click.option('--target_column', default='target', help='Name of the target column in csv files')
@click.option('--scaling', default=(3.0, 1.5), nargs=2, type=float, 
              help='Two values m and s white space separated (where m -> y(m) = 0.5 and s -> steepness)')
@click.option('--atom_property_csv', default=None, 
              help='CSV with atomic properties (root directory in ' + GRID_PRM_DIR)

def run(model_dir,
        model_name, 
        train_csv,
        test_csv, 
        data_root,
        weight_decay, 
        learning_rate, 
        batch_size, 
        max_epochs,  
        ncores,
        device_name,
        bin_size,
        ngpu,
        use_saved, 
        target_column, 
        scaling, 
        atom_property_csv):
              
    exec 'from ilovemhc.torch_models import %s as current_model_class' % model_name in globals(), locals()
    
    logging.info('model_dir         = {}'.format(model_dir))
    logging.info('model_name        = {}'.format(model_name))
    logging.info('train_csv         = {}'.format(train_csv))
    logging.info('test_csv          = {}'.format(test_csv))
    logging.info('data_root         = {}'.format(data_root))
    logging.info('weight_decay      = {}'.format(weight_decay))
    logging.info('learning_rate     = {}'.format(learning_rate))
    logging.info('batch_size        = {}'.format(batch_size))
    logging.info('max_epochs        = {}'.format(max_epochs))
    logging.info('ncores            = {}'.format(ncores))
    logging.info('device_name       = {}'.format(device_name))
    logging.info('bin_size          = {}'.format(bin_size))
    logging.info('ngpu              = {}'.format(ngpu))
    logging.info('use_saved         = {}'.format(use_saved))
    logging.info('target_column     = {}'.format(target_column))
    logging.info('scaling           = {}'.format(scaling))
    logging.info('atom_property_csv = {}'.format(atom_property_csv))
              
    model_prefix = 'model'

    logging.info('Getting device..')
    avail, device = get_device(device_name)

    if not avail:
        raise RuntimeError('CUDA is not available')

    logging.info('Reading tables..')
    test_table = pd.read_csv(test_csv)
    train_table = pd.read_csv(train_csv)

    test_table['target'] = test_table[target_column]
    train_table['target'] = train_table[target_column]
              
    target_scale = dataset.scale_func(scaling[0], scaling[1])
              
    grid_maker = None
    if atom_property_csv:
        logging.info('Getting custom GridMaker..')
        grid_maker = GridMaker(propspath=GRID_PRM_DIR + '/' + atom_property_csv)
    
    logging.info('Creating test dataset..')
    test_set = dataset.MolDataset(test_table, 
                                  data_root, 
                                  grid_maker=grid_maker,
                                  bin_size=bin_size,
                                  target_transform=target_scale,
                                  add_index=True)
    
    logging.info('Creating train dataset..')
    train_set = dataset.MolDataset(train_table, 
                                   data_root,
                                   grid_maker=grid_maker,
                                   bin_size=bin_size,
                                   target_transform=target_scale)
    
    logging.info('Creating test loader..')
    test_loader = DataLoader(dataset=test_set, 
                             batch_size=batch_size, 
                             num_workers=ncores, 
                             shuffle=False,
                             drop_last=False)
    
    logging.info('Creating train loader..')
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, num_workers=ncores, shuffle=False)

    logging.info('Getting input shape..')
    input_shape = torch.tensor(train_set[0][0].shape).numpy()
    logging.info(input_shape)
    
    if use_saved:
        logging.info('Loading model from %s..' % use_saved)
        model = torch.load(use_saved)
    else:
        logging.info('Initializing model..')
        model = current_model_class(input_shape)
    
    logging.info(model)
    
    using_data_parallel = False
    if avail and ngpu > 1:
        ngpu_total = torch.cuda.device_count()
        if ngpu_total < ngpu:
            logging.warning('Number of GPUs specified is too large: %i > %i. Using all GPUs' % (ngpu, ngpu_total))
            ngpu = ngpu_total
        if ngpu > 1:
            logging.info('Using DataParallel on %i GPUs' % ngpu)
            model = nn.DataParallel(model, device_ids=list(range(ngpu)))
            using_data_parallel = True
            
    model.to(device)
        
    logging.info('Creating optimizer..')
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    logging.info(optimizer)

    loss = torch.nn.MSELoss()
    logging.info(loss)

    logging.info('Creating evaluator..')
    evaluator = make_evaluator(model, loss, device, model_dir, model_prefix, every_niter=100)
       
    logging.info('Creating trainer..')
    trainer = make_trainer(model, optimizer, loss, device, evaluator, test_loader, 
                           test_table, model_dir, model_prefix, every_niter=100)
    
    logging.info('Starting trainer..')
    trainer(train_loader, max_epochs)
    
    logging.info("COMPLETED")
    
    #trainer = regression_trainer_with_tagwise_statistics(model, 
    #                                                     optimizer, 
    #                                                     loss, 
    #                                                     test_loader, 
    #                                                     test_table, 
    #                                                     device, 
    #                                                     model_dir, 
    #                                                     model_prefix,
    #                                                     every_n_iter=10)
    
    
    #trainer.run(train_loader, max_epochs)
    
if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s', 
                    level=logging.INFO, 
                    datefmt='%m/%d/%Y %I:%M:%S %p')
    
    run()
