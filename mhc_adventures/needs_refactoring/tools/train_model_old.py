import pandas as pd
import numpy as np
import os

import torch
from torch.utils.data import Dataset, DataLoader

from torch import nn
import torch.nn.functional as F
import torch.optim

import ilovemhc
from ilovemhc.wrappers import *
import ilovemhc.utils as utils
import ilovemhc.grids as grids
import ilovemhc.dataset as dataset
from ilovemhc.engines import regression_trainer_with_tagwise_statistics

def _cuda_avail():
    print("Checking CUDA availability ...")
    avail = torch.cuda.is_available()
    print("CUDA is available yay" if avail else "CUDA is not available :-(")

    if avail:
        ngpu = torch.cuda.device_count()
        print('Let\'s use %i GPUs' % ngpu)
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print('Using ' + str(device))
    
    return avail, device
    
class ModelClass_Probe(nn.Module):
    def __init__(self, input_shape):
        super(ModelClass_Probe, self).__init__()
        
        mean = 0.0
        std = 0.001
        init_fun = lambda x: nn.init.normal_(x.weight, mean, std)
        
        self.input_shape = input_shape
        self.batchnorm = nn.BatchNorm3d(self.input_shape[0])
        self.dropout = nn.Dropout3d()
        
        self.conv11 = nn.Conv3d(self.input_shape[0], 64, 3, padding=1)
        init_fun(self.conv11)
        
        self.conv21 = nn.Conv3d(64, 128, 3, padding=1)
        init_fun(self.conv21)
        
        self.conv31 = nn.Conv3d(128, 256, 3, padding=1)
        init_fun(self.conv31)
        
        self.fc1 = nn.Linear(256 * 5 * 3 * 3, 1024)
        init_fun(self.fc1)
        
        self.fc2 = nn.Linear(1024, 512)
        init_fun(self.fc2)

        self.fc3 = nn.Linear(512, 1)
        init_fun(self.fc3)
        
    def forward(self, x):
        x = self.batchnorm(x)
        x = F.relu(self.conv11(x))
        x = F.max_pool3d(x, 2, ceil_mode=True)
        x = self.dropout(x)
        
        x = F.relu(self.conv21(x))
        x = F.max_pool3d(x, 2, ceil_mode=True)
        x = self.dropout(x)

        x = F.relu(self.conv31(x))
        x = F.max_pool3d(x, 2, ceil_mode=True)
        x = self.dropout(x)
        
        x = x.view(-1, 256 * 5 * 3 * 3)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        x = F.relu(self.fc2(x))
        x = self.dropout(x)

        x = self.fc3(x)
        x = torch.sigmoid(x)
        return x
    
if __name__ == '__main__':
    ncores = 2
    batch_size = 2
    max_epochs = 2
    learning_rate = 0.001
    weight_decay = 0.001
    train_csv = '../data/data_samples/learning/train.csv'
    test_csv = '../data/data_samples/learning/test.csv'
    root_dir = '../data/data_samples/learning'
    model_dir = 'models'
    model_prefix = 'prefix'

    avail, device = _cuda_avail()

    test_table = pd.read_csv(test_csv)
    train_table = pd.read_csv(train_csv)

    target_scale = lambda x: 1.0 / (1.0 + np.exp((x-3.0) * 1.5)) * (1 + np.exp(-3*1.5))
    test_set = dataset.MolDataset(test_table, root_dir, target_transform=target_scale, add_index=True, remove_grid=True)
    train_set = dataset.MolDataset(train_table, root_dir, target_transform=target_scale, remove_grid=True)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, num_workers=ncores, shuffle=False)
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, num_workers=ncores, shuffle=True, drop_last=False)

    input_shape = torch.tensor(train_set[0][0].shape).numpy()
    print("Grid shape")
    print(input_shape)
    model = ModelClass_Probe(input_shape)

    if avail:
        #device_ids = [0, 1, 2, 3, 4, 5, 6, 7]
        model = nn.DataParallel(model) #, device_ids=device_ids)
        #device = torch.device("cuda:%i" % gpu)
        #model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    print(optimizer)

    loss = torch.nn.MSELoss()
    print(loss)

    trainer = regression_trainer_with_tagwise_statistics(model, 
                                                         optimizer, 
                                                         loss, 
                                                         test_loader, 
                                                         test_table, 
                                                         device, 
                                                         model_dir, 
                                                         model_prefix)
    trainer.run(train_loader, max_epochs)
    print("COMPLETED")
    
    
