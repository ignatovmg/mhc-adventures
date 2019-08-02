import pandas as pd
import numpy as np
import os
import traceback
import logging

import torch
from torch.utils.data import Dataset

from wrappers import *
import utils
import grids
import molgrid


def scale_func_sigmoid(break_point=1.2, steepness=3.):
    buf = np.exp(steepness*break_point)
    c1 = 1./steepness*np.log(buf-2.)
    c2 = (1.-buf)/(2.-buf)
    target_scale = lambda x: c2 / (1. + np.exp((x - c1)*steepness))
    return target_scale


class MolDataset(Dataset):
    """Mol dataset."""

    def __init__(self, csv_table,
                 root_dir='.', 
                 target_col='target',
                 path_col='path',
                 input_is_pdb=True,
                 grid_maker=None, 
                 bin_size=1.0,
                 grid_transform=None,
                 pdb_transform=None,
                 target_transform=None,
                 add_index=False, 
                 remove_grid=False):
        """
        Args:
            csv_table (pandas.DataFrame): Must contain columns specified in target_col and path_col
            root_dir (string, optional): Directory with all the samples (default: '.')
            input_is_pdb (bool, optional): Indicates that data  samples are in pdb format (default: True).
            grid_maker (class GridMaker, optional): Custom GridMaker (default: None)
            bin_size (float, optional): Grid bin size in Angstroms (default: 0.1)
            grid_transform (callable, optional): Optional transform to be applied on a grid. Returns numpy.array. (default: None). 
            pdb_transform (callable, optional): Optional transform to be applied on a sample. Returns path to new pdb. (default: None). 
            target_transform (callable, optional): Optional transform to be applied on a sample. Returns single float (default: None). 
        """
        if path_col not in csv_table.columns:
            raise ValueError('Table must contain column "%s"' % path_col)
            
        if target_col not in csv_table.columns:
            raise ValueError('Table must contain column "%s"' % target_col)
        
        self.csv = csv_table
        self.target_list = np.array(self.csv[target_col], dtype=float)
        self.path_list = np.array(self.csv[path_col])
        self.root_dir = root_dir
        self.grid_transform = grid_transform
        self.pdb_transform = pdb_transform
        self.target_transform = target_transform
        self.input_is_pdb = input_is_pdb
        self.add_index = add_index
        self.remove_grid = remove_grid
        
        if self.input_is_pdb:
            self.bin_size = bin_size
            
        if not grid_maker:
            self.grid_maker = grids.GridMaker()
        else:
            self.grid_maker = grid_maker
    '''        
    def make_grid(self, pdb_path, remove_tmp=True, hsd2his=True):
        if hsd2his:
            tmp_file = pdb_path + '.tmp'
            utils.hsd2his(pdb_path, tmp_file)
        else:
            tmp_file = pdb_path
            
        bin_size = self.bin_size
        props_file = self.grid_maker.properties_file
        types_file = self.grid_maker.types_file
        nchannels = self.grid_maker.nchannels
        grid = molgrid.make_grid(tmp_file, props_file, types_file, bin_size, nchannels)
        
        if hsd2his and remove_tmp:
            remove_files([tmp_file])
            
        return grid
    '''

    def __len__(self):
        return self.csv.shape[0]

    def __getitem__(self, idx):
        # get sample path
        local_path = self.path_list[idx]
        full_path = os.path.join(self.root_dir, local_path)
        file_absent_error(full_path)
        
        grid_path = None
        
        if self.input_is_pdb:
            # transform pdb if needed
            if self.pdb_transform:
                full_path = self.pdb_transform(full_path)

            try:
                grid = self.grid_maker.make_grid(full_path, self.bin_size)
            except Exception as e:
                logging.error('Failed creating grid for %s' % full_path)
                logging.exception(e)
                raise
        else:
            grid_path = full_path
            
        if grid_path:   
            dims, grid = self.grid_maker.read_grid(grid_path)
            grid = grid.reshape(dims)

            if self.remove_grid:
                remove_files([grid_path])

        # trasform grid
        if self.grid_transform:
            grid = self.grid_transform(grid)
        
        grid = torch.from_numpy(grid)
        
        # read and transform target
        target = self.target_list[idx]
        if self.target_transform:
            target = self.target_transform(target)
        
        #if self.add_index:
        sample = (grid.type(torch.FloatTensor), np.float32(target), np.long(idx))
        #else:
        #    sample = (grid.type(torch.FloatTensor), np.float32(target))

        return sample
