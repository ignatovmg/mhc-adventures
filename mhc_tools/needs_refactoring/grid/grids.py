import struct
import numpy as np
import pandas as pd
from itertools import product
from gridData import Grid
from path import Path

from . import define
from . import molgrid
from .wrappers import *


class GridMaker(object):
    """
    Creates 4D numpy array from a pdb file: (property_channels, x, y, z)
    """

    def __init__(self,
                 binpath=define.GRID_EXE,
                 namespath=define.ATOM_TYPES_DEFAULT, 
                 propspath=define.ATOM_PROPERTY_DEFAULT):
        
        self.grid_bin = binpath
        self.properties_file = propspath
        self.types_file = namespath
        self.prop_table = pd.read_csv(propspath, sep=' ')
        self.nchannels = self.prop_table.shape[1] - 2
        
    def read_grid(self, path):
        """
        (deprecated)

        Read grid from file created by self.make_grid_file
        """
        with open(path, 'r') as f:
            binary = f.read()

        head_fmt = '<qqqqq'
        head_len = struct.calcsize(head_fmt)
        unpk = struct.unpack(head_fmt, binary[:head_len])
        total = unpk[0]
        dimensions = unpk[1:]
        if total != reduce(lambda x,y: x*y, dimensions):
            #logging.error("Header mismatch after converting")
            raise RuntimeError("Header mismatch after converting")

        body_fmt = '<' + 'f'*total
        body_len = struct.calcsize(body_fmt)
        unpk = struct.unpack(body_fmt, binary[head_len:head_len + body_len])
        array = np.array(unpk, dtype=np.float32)

        return dimensions, array
        
    def grid_to_pdb(self, save_pdb, grid, bin_size, channel=None, use_channel_names=False):
        origin = np.array([7.473, 4.334, 7.701]) - 5.0;
        
        if grid.shape[0] % 2 != 0:
            raise ValueError('Number of channels must be even (%i)' % grid.shape[0])
            
        ch_per_mol = grid.shape[0] / 2
        
        def ele_map(val):
            if val > levels[0]:
                ele = ' F'
            elif val > levels[1]:
                ele = ' O'
            elif val > levels[2]:
                ele = ' N'
            elif val > levels[3]:
                ele = ' S'
            else:
                ele = ' C'
            return ele
        
        if not channel:
            clist = range(ch_per_mol)
        else:
            clist = [channel]
        
        with open(save_pdb, 'w') as f:
            for mdl, ch in enumerate(clist, 1):
                if (ch > ch_per_mol):
                    raise ValueError('Channel id must be less, than %i' % ch_per_mol)

                spacial_rec = grid[ch]
                spacial_lig = grid[ch+ch_per_mol]
                dims = spacial_rec.shape
                coords = product(*[range(x) for x in dims])

                max_lig = spacial_lig.max()
                levels = [max_lig / 2**i for i in range(1, 5)]
                logging.info(levels)
            
                if use_channel_names:
                    names = self.prop_table.columns[1:]
                    channel_name = names[mdl]
                else:
                    channel_name = str(mdl)
                    
                f.write('HEADER %s\n' % channel_name)
                for crd in coords:
                    xyz = (np.array(crd) * bin_size) + origin

                    if spacial_rec[crd] > 0.0:
                        ele = ele_map(spacial_rec[crd])

                        line = 'ATOM      1  C   *** A   1    %8.3f%8.3f%8.3f  1.00  1.00          %s\n' % \
                            (xyz[0], xyz[1], xyz[2], ele)
                        f.write(line)

                    if spacial_lig[crd] > 0.0:
                        ele = ele_map(spacial_lig[crd])
                        line = 'ATOM      1  C   *** B   1    %8.3f%8.3f%8.3f  1.00  1.00          %s\n' % \
                            (xyz[0], xyz[1], xyz[2], ele)
                        f.write(line)
                f.write('END\n')
                
    def grid_to_dx(self, save_dir, grid, bin_size, use_channel_names=False):
        #origin = np.array([7.473, 4.334, 7.701]) - 5.0
        
        if grid.shape[0] % 2 != 0:
            raise ValueError('Number of channels must be even (%i)' % grid.shape[0])
            
        ch_per_mol = grid.shape[0] / 2
        for shift, prefix in [(0, 'rec'), (ch_per_mol, 'lig')]:
            for ch in range(ch_per_mol):
                ch += shift
                if use_channel_names:
                    names = self.prop_table.columns[1:]
                    names *= 2
                    channel_name = names[ch]
                else:
                    channel_name = str(ch)
                    
                g = Grid(grid[ch], origin=np.array([7.473, 4.334, 7.701]) - 5.0, delta=bin_size) 
                g.export(Path(save_dir).joinpath(channel_name + '.dx'), 'DX')
    
    # this one uses direct binding of C++ code, so better to use this one
    def make_grid(self, inputpdb, binsize):
        """
        Make grid using C binding. Returns numpy.array
        """
        grid = molgrid.make_grid(inputpdb, self.properties_file, self.types_file, binsize, self.nchannels)
        return grid
    
    def make_grid_file(self, savepath, inputpdb, binsize):
        """
        (deprecated)

        Make grid (binary format) and write it to savepath
        """
        call = [self.grid_bin, inputpdb, self.properties_file, self.types_file, str(binsize), str(self.nchannels), savepath]
        output = shell_call(call)
        return output
