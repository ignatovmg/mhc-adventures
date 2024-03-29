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
@click.argument('model_file', type=click.Path(exists=True)) 
@click.argument('data_csv', type=click.Path(exists=True)) 
@click.argument('data_root', type=click.Path(exists=True))
@click.option('-p', '--output_prefix', default='out', type=click.Path())
@click.option('-o', '--output_dir', default='.', type=click.Path())
@click.option('-b', '--batch_size', default=128)
@click.option('-c', '--ncores', default=0, help='N cores to use for data loading')
@click.option('-d', '--device_name', default="cpu", help='Device id in torch format: e.g. "cuda:0" or "cpu"')
@click.option('--bin_size', default=1.0, help='Grid resolution')
@click.option('--ngpu', default=1, help='Number of GPUs to use')
@click.option('--target_column', default='target', help='Name of the target column in csv file')
@click.option('--tag_column', default='tag', help='Name of the tag column in csv file')
@click.option('--scaling', default=(3.0, 1.5), nargs=2, type=float, 
              help='Two values m and s white space separated (where m -> y(m) = 0.5 and s -> steepness)')
@click.option('--atom_property_csv', default=None, 
              help='CSV with atomic properties (root directory in ' + GRID_PRM_DIR)
@click.option('--no_scale', is_flag=True, help='Don\'t scale the target value')
def run(model_file,
        data_csv, 
        data_root,
        output_dir,
        output_prefix,
        batch_size,
        ncores,
        device_name,
        bin_size,
        ngpu,
        target_column,
        tag_column,
        scaling,
        atom_property_csv,
        no_scale):

    for k, v in locals().iteritems():
        logging.info("{:20s} = {}".format(str(k), str(v)))
    
    # model_prefix = 'model'

    logging.info('Getting device..')
    avail, device = get_device(device_name)

    if not avail:
        logging.warning('CUDA is not available')
        #raise RuntimeError('CUDA is not available')

    logging.info('Reading tables..')
    
    data_table = pd.read_csv(data_csv)
    data_table['target'] = data_table[target_column]
    data_table['tag'] = data_table[tag_column]
   
    if no_scale:
        target_scale = None
    else:
        target_scale = dataset.scale_func_sigmoid(scaling[0], scaling[1])

    grid_maker = None
    if atom_property_csv:
        logging.info('Getting custom GridMaker..')
        grid_maker = GridMaker(propspath=GRID_PRM_DIR + '/' + atom_property_csv)
    
    logging.info('Creating data dataset..')
    data_set = dataset.MolDataset(data_table, 
                                  data_root, 
                                  grid_maker=grid_maker,
                                  bin_size=bin_size,
                                  target_transform=target_scale)
    
    logging.info('Creating data loader..')
    data_loader = DataLoader(dataset=data_set, 
                             batch_size=batch_size, 
                             num_workers=ncores, 
                             shuffle=False,
                             drop_last=False)
    
    logging.info('Getting input shape..')
    input_shape = torch.tensor(data_set[0][0].shape).numpy()
    logging.info(input_shape)
    
    logging.info('Loading model from %s..' % model_file)
    if str(device) == 'cpu':
        model = torch.load(model_file, map_location='cpu')
    else:
        model = torch.load(model_file)
    
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
        
    loss = torch.nn.MSELoss()
    logging.info(loss)

    logging.info('Creating evaluator..')
    model_prefix = os.path.basename(model_file)
    evaluator = make_evaluator(model, loss, device, model_dir=output_dir, model_prefix=output_prefix, every_niter=100)
       
    logging.info('Starting evaluator..')
    evaluator(data_loader, data_table, 1)
    
    logging.info("COMPLETED")


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s', 
                    level=logging.INFO, 
                    datefmt='%m/%d/%Y %I:%M:%S %p')
    
    run()
