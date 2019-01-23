import subprocess
import struct
import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 

import utils
import define
from wrappers import *

class GridMaker():
    def __init__(self, binpath=define.GRID_EXE, 
                 namespath=define.ATOM_TYPES_DEFAULT, 
                 propspath=define.ATOM_PROPERTY_DEFAULT):
        
        self.grid_bin = binpath
        self.properties_file = propspath
        self.types_file = namespath
        self.prop_table = pd.read_csv(propspath, sep=' ')
        
    def read_grid(self, path):
        with open(path, 'r') as f:
            binary = f.read()

        head_fmt = '<qqqqq'
        head_len = struct.calcsize(head_fmt)
        unpk = struct.unpack(head_fmt, binary[:head_len])
        total = unpk[0]
        dimensions = unpk[1:]
        if total != reduce(lambda x,y: x*y, dimensions):
            print("Header mismatch after converting")
            return 1

        body_fmt = '<' + 'f'*total
        body_len = struct.calcsize(body_fmt)
        unpk = struct.unpack(body_fmt, binary[head_len:head_len + body_len])
        array = np.array(unpk, dtype=np.float32)

        return dimensions, array
    
    def create_gif(self, grid_file, channel=5, thre=0.1):
        
        dim, grid = self.read_grid(grid_file)
        grid = grid.reshape(dim)
        nch = dim[0]
        
        if (channel > nch / 2):
            print ('Channel id must be less, than %i' % (nch / 2))
            return 1

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

            ax.scatter(coords[:,0], coords[:,1], coords[:,2], c='r')

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
            ax.scatter(coords[:,0], coords[:,1], coords[:,2], c='b')
            ax.view_init(60, angle)
            
            filename=("%03i" % counter) +'.png'
            plt.savefig(filename, dpi=96)
            png_files.append(filename)
            counter += 1
            
        call = 'convert -delay 30 ' + ' '.join(png_files) + ' a.gif && mv a.gif a.gif.png'
        #print call
        status = subprocess.call(call, shell=True)
        
        call = 'rm -f ' + ' '.join(png_files)
        #print call
        subprocess.call(call, shell=True)
        
        return status

    def draw_grid_in_jupyter_notebook(self, grid_file, channel, thre):
        from IPython.display import Image
        
        if self.create_gif(grid_file, channel=channel, thre=thre) == 0:
            return Image(filename="a.gif.png")
        return 1

    def make_grid(self, savepath, inputpdb, binsize, remove_tmp=True):
        tmp_file = inputpdb + '.tmp'
        utils.hsd2his(inputpdb, tmp_file)
        
        call = [self.grid_bin, tmp_file, self.properties_file, self.types_file, str(binsize), savepath]
        output = shell_call(call)
        
        if remove_tmp:
            remove_files([tmp_file])

    def make_grids(self, savedir, pdblist, binsize):
        table = []
        with open(pdblist, 'r') as f:
            #g.write(' grid\n')
            counter = 0
            
            for pdb in f:
                pdb = pdb.strip()
                if pdb[-4:] != '.pdb':
                    throw_error('Files must end with .pdb')
                
                outfile = savedir + '/' + os.path.basename(pdb)[:-3] + 'bin'
                self.make_grid(outfile, pdb, binsize)
                
                #g.write('%i %s\n' % (counter, outfile))
                table.append((counter, outfile))
                counter += 1