# -*- coding: utf-8 -*-

import sys
import os
import torch
import numpy as np
import random
import csv
import os
import random
from random import shuffle
from os import listdir
from os.path import join
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms,utils
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
logging.getLogger().setLevel(logging.INFO)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class basic_blocks(nn.Module):
    def __init__(self,input_channel,mid_channel,output_channel):
        super(basic_blocks,self).__init__()
        self.conv_block1 = nn.Sequential(nn.Conv2d(input_channel,mid_channel,kernel_size = 3,padding = 1,padding_mode = 'replicate'),nn.BatchNorm2d(mid_channel),nn.ReLU())
        self.conv_block2 = nn.Sequential(nn.Conv2d(mid_channel, output_channel, kernel_size = 3,padding = 1,padding_mode = 'replicate'),nn.BatchNorm2d(output_channel),nn.ReLU())
        
    def forward(self,x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        return x

class encoder(nn.Module):
    def __init__(self,input_channel,output_channel):
        super(encoder,self).__init__()
        self.encode = nn.Sequential(nn.MaxPool2d(2),basic_blocks(input_channel,output_channel,output_channel))
        
    def forward(self,x):
        return self.encode(x)

class decoder(nn.Module):
    def __init__(self,input_channel,output_channel,bilinear = True):
        super(decoder,self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor = 2, mode = 'bilinear', align_corners = True)
            self.conv_block = basic_blocks(input_channel,input_channel//2,output_channel)
    def forward(self,x1,x2):
        x1 = self.up(x1)
        x = torch.cat([x2,x1],dim=1)
        return self.conv_block(x)
    
    
class Unet_defocus(nn.Module):
    def __init__(self,input_channel = 2,output_channel = 13,bilinear = True):
        super(Unet_defocus,self).__init__()
        
        factor =  2 if bilinear else 1
        self.en0 = basic_blocks(input_channel,64,64) 
        self.en1 = encoder(64,128)
        self.en2 = encoder(128,256)
        self.en3 = encoder(256,512)
        self.en4 = encoder(512,1024 // factor)
        
        self.de1 = decoder(1024,512//factor)
        self.de2 = decoder(512,256//factor)
        self.de3 = decoder(256,128//factor)
        self.de4 = decoder(128,64)
        self.output = nn.Conv2d(64,output_channel,kernel_size=1)
    def forward(self,x):
        x0 = self.en0(x)
        x1 = self.en1(x0)
        x2 = self.en2(x1)
        x3 = self.en3(x2)
        x4 = self.en4(x3)
        x = self.de1(x4,x3)
        x = self.de2(x,x2)
        x = self.de3(x,x1)
        x = self.de4(x,x0)
        x  = self.output(x)
        return x

class paper1_net(nn.Module):
    def __init__(self):
        super(paper1_net,self).__init__()
        self.conv1 = nn.Conv2d(2, 32, 3, padding = 1,padding_mode = 'reflect')
        self.bn1 = nn.BatchNorm2d(32)
        self.maxpool = nn.MaxPool2d(2,stride=2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding = 1,padding_mode = 'reflect')
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding = 1,padding_mode = 'reflect')
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 3, padding = 1,padding_mode = 'reflect')
        self.bn4 = nn.BatchNorm2d(256)
        self.fc1 = nn.Linear(256*8*8,1024)
        self.dropout = nn.Dropout()
        self.fc2 = nn.Linear(1024,13)
        
    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = self.maxpool(x)
        
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.maxpool(x)
        
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.maxpool(x)
        x = x.view(-1,256*8*8)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
if __name__ == '__main__':
    x = torch.randn(1,2,512,512)
    model = Unet_defocus()
    out = model(x)
    print(out.shape)
    
    
    