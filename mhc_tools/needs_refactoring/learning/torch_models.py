import torch
from torch import nn
import torch.nn.functional as F
import torch.optim

from collections import OrderedDict
import logging
import numpy as np

# strip_keys is needed because if DataParallel was used during saving, the keys
# of model_state_dict are prepended with 'module.'
def load_model(model, saved_model_path, cpu=True, strip_keys=False):
    if cpu:
        model_state_dict = torch.load(saved_model_path, map_location='cpu')
    else:
        model_state_dict = torch.load(saved_model_path)
    print model_state_dict.__dict__
        
    if strip_keys:
        new_dict = []
        for key, val in model_state_dict.iteritems():
            if key.startswith('module.'):
                new_dict.append(key[7:], val)
            else:
                new_dict.append(key, val)
        model_state_dict = OrderedDict(new_dict)
        
    model.load_state_dict(model_state_dict)
    return model


class ModelTemplate(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        
    def _infer_shape(self, layers, shape):
        x = torch.randn(tuple([1] + shape))
        logging.info('Inferring shape for convolutional part...')
        logging.info(x.shape)
        
        for l in layers:
            x = l(x);
            logging.info("==== " + str(l) + " ====")
            logging.info(x.shape)
        
        return torch.tensor(x.shape)
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        for l in self.conv:
            x = l(x)
        x = x.view(x.size()[0], -1)
        for l in self.fc:
            x = l(x)
        x = torch.sigmoid(x)
        return x  
    
# ========================= Best Model So Far ===========================
#
#                               Model3
#
class Model1(nn.Module):
    def __init__(self, input_shape):
        nn.Module.__init__(self)
        
        self.input_shape = input_shape
        
        self.conv = nn.ModuleList(
            [nn.BatchNorm3d(self.input_shape[0]),
             nn.Conv3d(self.input_shape[0], 64, 3, padding=1), 
             nn.ReLU(), 
             nn.MaxPool3d(2),
             nn.Dropout3d(),
                
             nn.Conv3d(64, 128, 3, padding=1), 
             nn.ReLU(), 
             nn.MaxPool3d(2), 
             nn.Dropout3d(),
                
             nn.Conv3d(128, 256, 3, padding=1), 
             nn.ReLU(), 
             nn.MaxPool3d(2),
             nn.Dropout3d()])
        
        #conv_shape = _output_shape(self.input_shape[1:], self.conv)
        conv_shape = self._infer_shape()
        
        self.fc = nn.ModuleList(
            [nn.Linear(int(conv_shape.prod()), 1024), 
             nn.ReLU(),
             nn.Dropout3d(),
             
             nn.Linear(1024, 1)])
        
        map(_init_fun, self.conv)
        map(_init_fun, self.fc)
        
    def _infer_shape(self):
        x = torch.randn(tuple([1] + map(int, self.input_shape)))
        logging.info('Inferring shape for convolutional part...')
        logging.info(x.shape)
        
        for l in self.conv:
            x = l(x);
            logging.info("==== " + str(l) + " ====")
            logging.info(x.shape)
        
        return torch.tensor(x.shape)
    
    def forward(self, x):
        for l in self.conv:
            x = l(x)
        x = x.view(x.size()[0], -1)
        for l in self.fc:
            x = l(x)
        x = torch.sigmoid(x)
        return x    
    
# changing dropout to batchnorm
class Model2(nn.Module):
    def __init__(self, input_shape):
        nn.Module.__init__(self)
        
        self.input_shape = input_shape
        
        self.conv = nn.ModuleList(
            [nn.BatchNorm3d(self.input_shape[0]),
             nn.Conv3d(self.input_shape[0], 64, 3, padding=1), 
             nn.ReLU(), 
             nn.MaxPool3d(2),
             nn.BatchNorm3d(64),
                
             nn.Conv3d(64, 128, 3, padding=1), 
             nn.ReLU(), 
             nn.MaxPool3d(2), 
             nn.BatchNorm3d(128),
                
             nn.Conv3d(128, 256, 3, padding=1), 
             nn.ReLU(), 
             nn.MaxPool3d(2),
             nn.BatchNorm3d(256)])
        
        conv_shape = _output_shape(self.input_shape[1:], self.conv)
        
        self.fc = nn.ModuleList(
            [nn.Linear(256 * int(conv_shape.prod()), 1024), 
             nn.ReLU(),
             nn.BatchNorm1d(1024),

             nn.Linear(1024, 1)])
        
        map(_init_fun, self.conv)
        map(_init_fun, self.fc)
    
    def forward(self, x):
        for l in self.conv:
            x = l(x)
        x = x.view(x.size()[0], -1)
        for l in self.fc:
            x = l(x)
        x = torch.sigmoid(x)
        return x

# doubling number of filters in each layer
class Model3(nn.Module):
    def __init__(self, input_shape):
        nn.Module.__init__(self)
        
        self.input_shape = input_shape
        
        self.conv = nn.ModuleList(
            [nn.BatchNorm3d(self.input_shape[0]),
             nn.Conv3d(self.input_shape[0], 128, 3, padding=1), 
             nn.ReLU(), 
             nn.MaxPool3d(2),
             nn.Dropout3d(),
                
             nn.Conv3d(128, 256, 3, padding=1), 
             nn.ReLU(), 
             nn.MaxPool3d(2), 
             nn.Dropout3d(),
                
             nn.Conv3d(256, 512, 3, padding=1), 
             nn.ReLU(), 
             nn.MaxPool3d(2),
             nn.Dropout3d()])
        
        conv_shape = _output_shape(self.input_shape[1:], self.conv)
        
        self.fc = nn.ModuleList(
            [nn.Linear(512 * int(conv_shape.prod()), 1024), 
             nn.ReLU(),
             nn.Dropout3d(),

             nn.Linear(1024, 1)])
        
        map(_init_fun, self.conv)
        map(_init_fun, self.fc)
    
    def forward(self, x):
        for l in self.conv:
            x = l(x)
        x = x.view(x.size()[0], -1)
        for l in self.fc:
            x = l(x)
        x = torch.sigmoid(x)
        return x

# duplicating each conv layer
class Model4(nn.Module):
    def __init__(self, input_shape):
        nn.Module.__init__(self)
        
        self.input_shape = input_shape
        
        self.conv = nn.ModuleList(
            [nn.BatchNorm3d(self.input_shape[0]),
             nn.Conv3d(self.input_shape[0], 64, 3, padding=1), 
             nn.ReLU(), 
             nn.Dropout3d(),
             
             nn.Conv3d(64, 64, 3, padding=1), 
             nn.ReLU(), 
             nn.MaxPool3d(2),
             nn.Dropout3d(),
                
             nn.Conv3d(64, 128, 3, padding=1), 
             nn.ReLU(), 
             nn.Dropout3d(),
             
             nn.Conv3d(128, 128, 3, padding=1), 
             nn.ReLU(),
             nn.MaxPool3d(2), 
             nn.Dropout3d(),
                
             nn.Conv3d(128, 256, 3, padding=1), 
             nn.ReLU(), 
             nn.Dropout3d(),
             
             nn.Conv3d(256, 256, 3, padding=1), 
             nn.ReLU(),
             nn.MaxPool3d(2),
             nn.Dropout3d()])
        
        conv_shape = _output_shape(self.input_shape[1:], self.conv)
        
        self.fc = nn.ModuleList(
            [nn.Linear(256 * int(conv_shape.prod()), 1024), 
             nn.ReLU(),
             nn.Dropout3d(),

             nn.Linear(1024, 1)])
        
        map(_init_fun, self.conv)
        map(_init_fun, self.fc)
    
    def forward(self, x):
        for l in self.conv:
            x = l(x)
        x = x.view(x.size()[0], -1)
        for l in self.fc:
            x = l(x)
        x = torch.sigmoid(x)
        return x

# increased the number of fully connected layers
class Model5(nn.Module):
    def __init__(self, input_shape):
        nn.Module.__init__(self)
        
        self.input_shape = input_shape
        
        self.conv = nn.ModuleList(
            [nn.BatchNorm3d(self.input_shape[0]),
             nn.Conv3d(self.input_shape[0], 64, 3, padding=1), 
             nn.ReLU(), 
             nn.MaxPool3d(2, ceil_mode=True),
             nn.Dropout3d(),
                
             nn.Conv3d(64, 128, 3, padding=1), 
             nn.ReLU(), 
             nn.MaxPool3d(2, ceil_mode=True), 
             nn.Dropout3d(),
                
             nn.Conv3d(128, 256, 3, padding=1), 
             nn.ReLU(), 
             nn.MaxPool3d(2, ceil_mode=True),
             nn.Dropout3d()])
        
        conv_shape = _output_shape(self.input_shape[1:], self.conv)
        
        self.fc = nn.ModuleList(
            [nn.Linear(256 * int(conv_shape.prod()), 1024), 
             nn.ReLU(),
             nn.Dropout3d(),
             
             nn.Linear(1024, 1024), 
             nn.ReLU(),
             nn.Dropout3d(),

             nn.Linear(1024, 2)])
        
        map(_init_fun, self.conv)
        map(_init_fun, self.fc)
    
    def forward(self, x):
        for l in self.conv:
            x = l(x)
        x = x.view(x.size()[0], -1)
        for l in self.fc:
            x = l(x)
        x = torch.sigmoid(x)
        return x

# 4 conv layers
class Model6(nn.Module):
    def __init__(self, input_shape):
        nn.Module.__init__(self)
        
        self.input_shape = input_shape
        
        self.conv = nn.ModuleList(
            [nn.BatchNorm3d(self.input_shape[0]),
             nn.Conv3d(self.input_shape[0], 64, 3, padding=1), 
             nn.ReLU(), 
             nn.MaxPool3d(2),
             nn.Dropout3d(),
                
             nn.Conv3d(64, 128, 3, padding=1), 
             nn.ReLU(), 
             nn.MaxPool3d(2), 
             nn.Dropout3d(),
                
             nn.Conv3d(128, 256, 3, padding=1), 
             nn.ReLU(), 
             nn.MaxPool3d(2),
             nn.Dropout3d(), 
             
             nn.Conv3d(256, 512, 3, padding=1), 
             nn.ReLU(), 
             nn.Dropout3d()])
        
        conv_shape = _output_shape(self.input_shape[1:], self.conv)
        
        self.fc = nn.ModuleList(
            [nn.Linear(512 * int(conv_shape.prod()), 1024), 
             nn.ReLU(),
             nn.Dropout3d(),

             nn.Linear(1024, 1)])
        
        map(_init_fun, self.conv)
        map(_init_fun, self.fc)
    
    def forward(self, x):
        for l in self.conv:
            x = l(x)
        x = x.view(x.size()[0], -1)
        for l in self.fc:
            x = l(x)
        x = torch.sigmoid(x)
        return x

# changed 1st kernel to 5*5
class Model7(nn.Module):
    def __init__(self, input_shape):
        nn.Module.__init__(self)
        
        self.input_shape = input_shape
        
        self.conv = nn.ModuleList(
            [nn.BatchNorm3d(self.input_shape[0]),
             nn.Conv3d(self.input_shape[0], 64, 5, padding=2), 
             nn.ReLU(), 
             nn.MaxPool3d(2),
             nn.Dropout3d(),
                
             nn.Conv3d(64, 128, 3, padding=1), 
             nn.ReLU(), 
             nn.MaxPool3d(2), 
             nn.Dropout3d(),
                
             nn.Conv3d(128, 256, 3, padding=1), 
             nn.ReLU(), 
             nn.MaxPool3d(2),
             nn.Dropout3d()])
        
        conv_shape = _output_shape(self.input_shape[1:], self.conv)
        
        self.fc = nn.ModuleList(
            [nn.Linear(256 * int(conv_shape.prod()), 1024), 
             nn.ReLU(),
             nn.Dropout3d(),

             nn.Linear(1024, 1)])
        
        map(_init_fun, self.conv)
        map(_init_fun, self.fc)
    
    def forward(self, x):
        for l in self.conv:
            x = l(x)
        x = x.view(x.size()[0], -1)
        for l in self.fc:
            x = l(x)
        x = torch.sigmoid(x)
        return x
    

# The following is the resnet from 
# https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
# reshaped for 3D

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, in_shape, block=BasicBlock, layers=[3, 4, 6, 3], num_classes=1, zero_init_residual=False):
        super(ResNet, self).__init__()
        
        self.in_shape = in_shape
        self.inplanes = 64
        self.conv1 = nn.Conv3d(in_shape[0], 64, kernel_size=5, stride=2, padding=2, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        
        fc_input_size = self._infer_shape().prod()
        self.fc = nn.Linear(fc_input_size * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm3d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    def _infer_shape(self):
        probe = torch.randn(tuple([1] + map(int, self.in_shape)))
        logging.info('Inferring shape for convolutional part...')
        logging.info(probe.shape)
        
        x = self.conv1(probe); logging.info(x.shape)
        x = self.bn1(x); logging.info(x.shape)
        x = self.relu(x); logging.info(x.shape)
        x = self.maxpool(x); logging.info(x.shape)
        
        x = self.layer1(x); logging.info(x.shape)
        x = self.layer2(x); logging.info(x.shape)
        x = self.layer3(x); logging.info(x.shape)
        x = self.layer4(x); logging.info(x.shape)
        x = self.avgpool(x); logging.info(x.shape)
        
        return torch.tensor(x.shape)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        x = torch.sigmoid(x)
        return x


def ResNet34(in_shape, **kwargs):
    return ResNet(in_shape, layers=[3, 4, 6, 3], **kwargs)


def ResNet101(in_shape, **kwargs):
    return ResNet(in_shape, layers=[3, 4, 23, 3], **kwargs)


def ResNet152(in_shape, **kwargs):
    return ResNet(in_shape, layers=[3, 8, 36, 3], **kwargs)


# old model    
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

# ======================= Regularization Testing =======================   
class RegModel1(ModelTemplate):
    def __init__(self, input_shape):
        ModelTemplate.__init__(self)
        
        self.input_shape = list(input_shape)
        
        self.conv = nn.ModuleList(
            [nn.BatchNorm3d(self.input_shape[0]),
             nn.Conv3d(self.input_shape[0], 128, 3, padding=1), 
             nn.BatchNorm3d(128),
             nn.ReLU(), 
             nn.MaxPool3d(2),

             nn.Conv3d(128, 256, 3, padding=1),
             nn.BatchNorm3d(256),
             nn.ReLU(), 
             nn.MaxPool3d(2), 

             nn.Conv3d(256, 512, 3, padding=1), 
             nn.BatchNorm3d(512),
             nn.ReLU(), 
             nn.MaxPool3d(2)])
        
        self.conv_shape = self._infer_shape(self.conv, self.input_shape)
        fc_input = int(self.conv_shape.prod())
        
        self.fc = nn.ModuleList(
            [nn.Linear(fc_input, 1024),
             nn.BatchNorm1d(1024),
             nn.ReLU(),
             
             nn.Linear(1024, 1)])
        
        self._init_weights()
        #map(_init_fun, self.conv)
        #map(_init_fun, self.fc)


# removing leading batchnorm
class RegModel2(ModelTemplate):
    def __init__(self, input_shape):
        ModelTemplate.__init__(self)
        
        self.input_shape = list(input_shape)
        
        self.conv = nn.ModuleList(
            [nn.Conv3d(self.input_shape[0], 128, 3, padding=1),
             nn.BatchNorm3d(128),
             nn.ReLU(),
             nn.MaxPool3d(2),

             nn.Conv3d(128, 256, 3, padding=1),
             nn.BatchNorm3d(256),
             nn.ReLU(), 
             nn.MaxPool3d(2), 

             nn.Conv3d(256, 512, 3, padding=1), 
             nn.BatchNorm3d(512),
             nn.ReLU(), 
             nn.MaxPool3d(2)])
        
        self.conv_shape = self._infer_shape(self.conv, self.input_shape)
        fc_input = int(self.conv_shape.prod())
        
        self.fc = nn.ModuleList(
            [nn.Linear(fc_input, 1024),
             nn.BatchNorm1d(1024),
             nn.ReLU(),
             
             nn.Linear(1024, 1)])
        
        self._init_weights()

# removing all batchnorm
class RegModel3(ModelTemplate):
    def __init__(self, input_shape):
        ModelTemplate.__init__(self)
        
        self.input_shape = list(input_shape)
        
        self.conv = nn.ModuleList(
            [nn.Conv3d(self.input_shape[0], 128, 3, padding=1), 
             nn.ReLU(),
             nn.MaxPool3d(2),

             nn.Conv3d(128, 256, 3, padding=1),
             nn.ReLU(), 
             nn.MaxPool3d(2), 

             nn.Conv3d(256, 512, 3, padding=1), 
             nn.ReLU(), 
             nn.MaxPool3d(2)])
        
        self.conv_shape = self._infer_shape(self.conv, self.input_shape)
        fc_input = int(self.conv_shape.prod())
        
        self.fc = nn.ModuleList(
            [nn.Linear(fc_input, 1024),
             nn.ReLU(),
             
             nn.Linear(1024, 1)])
        
        self._init_weights()
        
# adding dropout
class RegModel4(ModelTemplate):
    def __init__(self, input_shape):
        ModelTemplate.__init__(self)
        
        self.input_shape = list(input_shape)
        
        self.conv = nn.ModuleList(
            [nn.Conv3d(self.input_shape[0], 128, 3, padding=1), 
             nn.ReLU(),
             nn.MaxPool3d(2),

             nn.Conv3d(128, 256, 3, padding=1),
             nn.ReLU(), 
             nn.MaxPool3d(2), 

             nn.Conv3d(256, 512, 3, padding=1), 
             nn.ReLU(), 
             nn.MaxPool3d(2)])
        
        self.conv_shape = self._infer_shape(self.conv, self.input_shape)
        fc_input = int(self.conv_shape.prod())
        
        self.fc = nn.ModuleList(
            [nn.Linear(fc_input, 1024),
             nn.ReLU(),
             nn.Dropout3d(),
             
             nn.Linear(1024, 1)])
        
        self._init_weights()

# deprecated functions
def conv3d_shape(h_w_d, kernel_size=1, stride=1, padding=0, dilation=1, ceil_flag=False):
    """
    Utility function for computing output of convolutions
    takes a tuple of (h,w,d) and returns a tuple of (h,w,d)
    """

    def transform(inp):
        if type(inp) is int:
            res = np.array([inp, inp, inp])
        elif type(inp) in [tuple, list]:
            res = np.array(inp)
        else:
            raise TypeError('Must be int, list or tuple')
        return res
    
    _kernel_size = transform(kernel_size)
    _stride = transform(stride)
    _padding = transform(padding)
    _dilation = transform(dilation)
        
    rnd = np.ceil if ceil_flag else np.floor
    res = rnd(((np.array(h_w_d) + (2 * _padding) - (_dilation * (_kernel_size - 1) ) - 1) / _stride) + 1)
    
    return res

def _init_fun(x):
    mean = 0.0
    std = 0.001
    nn.init.normal_(x.weight, mean, std) if hasattr(x, 'weight') else None

def _output_shape(in_shape, layers):
    x = in_shape.copy()
    print x
    
    for l in layers:
        if not hasattr(l, 'kernel_size'):
            continue
        cflag = l.ceil_mode if hasattr(l, 'ceil_mode') else False
        x = conv3d_shape(x, l.kernel_size, l.stride, l.padding, l.dilation, cflag)
        print x
    return x