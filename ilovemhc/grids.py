import struct
import os

import numpy as np
import pandas as pd
import logging

from itertools import product

import utils
import define
import molgrid
from wrappers import *

class GridMaker():
    def __init__(self, binpath=define.GRID_EXE, 
                 namespath=define.ATOM_TYPES_DEFAULT, 
                 propspath=define.ATOM_PROPERTY_DEFAULT):
        
        self.grid_bin = binpath
        self.properties_file = propspath
        self.types_file = namespath
        self.prop_table = pd.read_csv(propspath, sep=' ')
        self.nchannels = self.prop_table.shape[1] - 2
        
    def read_grid(self, path):
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
    
    def create_gif(self, grid, channel=5, thre=0.1):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D 
        
        dim = grid.shape
        nch = dim[0]
        
        if (channel > nch / 2):
            throw_error(ValueError, 'Channel id must be less, than %i' % (nch / 2))

        counter = 1
        png_files = []
        
        fig = plt.figure()
        for angle in np.arange(0, 180, 10):
            ax = fig.gca(projection='3d')

            slic = grid[nch/2+channel, :, : ,:]
            x = range(slic.shape[0])
            y = range(slic.shape[1])
            z = range(slic.shape[2])
            coords = []
            density = []
            for i in x:
                for j in y: 
                    for k in z:
                        if slic[i,j,k] > thre:
                            coords.append([i,j,k])
                            density.append(slic[i,j,k])
            coords = np.array(coords)
            #print coords.shape

            ax.scatter(coords[:,0], coords[:,1], coords[:,2], c='r', depthshade=True, alpha=0.5)

            slic = grid[channel, :, : ,:]

            x = range(slic.shape[0])
            y = range(slic.shape[1])
            z = range(slic.shape[2])
            coords = []
            density = []
            for i in x:
                for j in y: 
                    for k in z:
                        if slic[i,j,k] > thre:
                            coords.append([i,j,k])
                            density.append(slic[i,j,k])
            coords = np.array(coords)

            #fig = plt.figure(figsize=(10,10))
            #ax = fig.add_subplot(111, projection='3d')
            ax.scatter(coords[:,0], coords[:,1], coords[:,2], c='b', depthshade=True, alpha=0.5)
            ax.view_init(60, angle)
            
            filename=("%03i" % counter) +'.png'
            plt.savefig(filename, dpi=96)
            png_files.append(filename)
            counter += 1
            
        call = 'convert -delay 30 ' + ' '.join(png_files) + ' a.gif && mv a.gif a.gif.png'
        shell_call(call, shell=True)
        
        call = 'rm -f ' + ' '.join(png_files)
        shell_call(call, shell=True)
    
    def create_gif_from_file(self, grid_file, channel=5, thre=0.1):
        dim, grid = self.read_grid(grid_file)
        grid = grid.reshape(dim)
        create_gif(self, grid, channel=5, thre=0.1)
        
    def grid_to_pdb(self, save_pdb, grid, bin_size, ch=0):
        origin = np.array([7.473, 4.334, 7.701]) - 5.0;
        
        ch_per_mol = grid.shape[0] / 2
        
        if (ch > ch_per_mol):
            throw_error(ValueError, 'Channel id must be less, than %i' % ch_per_mol)
            
        spacial_rec = grid[ch]
        spacial_lig = grid[ch+ch_per_mol]
        dims = spacial_rec.shape
        coords = product(*[range(x) for x in dims])
        
        max_lig = spacial_lig.max()
        levels = [max_lig / 2**i for i in range(1, 5)]
        logging.info(levels)
        
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

        with open(save_pdb, 'w') as f:
            for crd in coords:
                xyz = (np.array(crd) / bin_size) + origin
                
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
                
    def draw_grid_in_jupyter_notebook(self, grid, channel, thre):
        from IPython.display import Image
        
        if self.create_gif(grid, channel=channel, thre=thre) == 0:
            return Image(filename="a.gif.png")
        return 1
    
    # this one uses direct binding of C++ code, so better to use this one
    def make_grid(self, inputpdb, binsize, remove_tmp=True, hsd2his=False):
        if hsd2his:
            tmp_file = inputpdb + '.tmp.' + str(os.getpid())
            utils.hsd2his(inputpdb, tmp_file)
        else:
            tmp_file = inputpdb
            
        grid = molgrid.make_grid(tmp_file, self.properties_file, self.types_file, binsize, self.nchannels)
        
        if hsd2his and remove_tmp:
            remove_files([tmp_file])
            
        return grid
    
    def make_grid_file(self, savepath, inputpdb, binsize, remove_tmp=True, hsd2his=False):
        if hsd2his:
            tmp_file = inputpdb + '.tmp.' + str(os.getpid())
            utils.hsd2his(inputpdb, tmp_file)
        else:
            tmp_file = inputpdb
        
        call = [self.grid_bin, tmp_file, self.properties_file, self.types_file, str(binsize), str(self.nchannels), savepath]
        output = shell_call(call)
                
        if hsd2his and remove_tmp:
            remove_files([tmp_file])
            
        return output

    def make_grids(self, savedir, pdblist, binsize):
        table = []
        with open(pdblist, 'r') as f:
            #g.write(' grid\n')
            counter = 0
            
            for pdb in f:
                pdb = pdb.strip()
                if pdb[-4:] != '.pdb':
                    raise RuntimeError('Files must end with .pdb')
                
                outfile = savedir + '/' + os.path.basename(pdb)[:-3] + 'bin'
                self.make_grid_file(outfile, pdb, binsize)
                
                #g.write('%i %s\n' % (counter, outfile))
                table.append((counter, outfile))
                counter += 1