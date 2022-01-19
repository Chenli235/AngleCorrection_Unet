# -*- coding: utf-8 -*-

import os
from os import listdir
from os.path import join
import matplotlib.pyplot as plt
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms,utils,models
import torchvision.utils as vutils
from torchvision import utils,models
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
#from pytorchcv.model_provider import get_model as ptcv_get_model
from scipy.interpolate import CubicSpline
import numpy as np
import timeit
import time
import random
from random import shuffle
import logging
import argparse
from datetime import datetime
import skimage.io
import skimage.transform
import skimage.color
import skimage

from model import Unet_defocus
from datasets.datasetandload import Defocus_dataset_2, Normalizer,Augmenter
logging.getLogger().setLevel(logging.DEBUG)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# Configuration
parser = argparse.ArgumentParser(description='Light-sheet estimation')
parser.add_argument('--dataset', default = 'defocus_dataset')
parser.add_argument('--data_path', default = '../data')
parser.add_argument('---workers', default=1,type=int)
parser.add_argument('--epochs', default=3,type=int)
parser.add_argument('--batch_size', default = 4, type=int)
parser.add_argument('--model_name', default='ModelUnet',type=str)

def main():
    args = parser.parse_args()
    print("PYTORCH VERSION", torch.__version__)
    args.data_dir = args.data_path
    args.start_epoch = 0
    args.model_name = '{}_{}'.format(args.model_name, datetime.now().strftime("%Y%m%d-%H%M%S"))
    makedirs('./results')
    args.res_dir = os.path.join('./results', args.model_name)
    makedirs(args.res_dir)
    logging.info('New train: loading data')
    train_dir = os.path.join(args.data_path,args.dataset)
    
    train_dataset = Defocus_dataset_2(train_dir,True,True,transforms.Compose([Normalizer(),Augmenter()]))
    train_dataloader = DataLoader(train_dataset,batch_size = args.batch_size,shuffle=True,num_workers = args.workers,pin_memory=True)
    
    
    
    classes_num = 13
    model = Unet_defocus(2,classes_num).to(device)
    criterion = nn.BCEWithLogitsLoss().to(device)
    #optimizer = optim.RMSprop(model.parameters(), lr=0.0001, weight_decay=1e-8, momentum=0.9)
    optimizer = optim.Adam(model.parameters(),lr=0.00001,betas=(0.5,0.999))
    train_losses = []
    for epoch in range(1,2):
        training_loss = 0
        running_loss = 0
        model.train()
        optimizer.zero_grad()
        if epoch % 10 == 0:
            logging.info('\n Training Epoch:{}\n'.format(epoch))
        for index, data in enumerate(train_dataloader):
            optimizer.zero_grad()
            
            imgs = data['image']
            true_masks = data['mask']
            #mask_type = torch.double
            true_masks = true_masks.to(device)
            masks_pred = model(imgs.to(device))
            
            loss = criterion(masks_pred,true_masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            training_loss += loss.item()
            if index % 20 == 19:
                logging.info('\n Epoch: {}/{}  Loss: {}\n'.format(epoch,index+1,running_loss))
                running_loss = 0
        if epoch % 50 == 0:
            logging.info('Epoch: {} is saved.'.format(epoch))
            torch.save(model.state_dict(),args.res_dir + '/' + 'net_{}.pt'.format(epoch))
        train_losses.append(training_loss)
        np.savetxt(args.res_dir + '/' + 'train_losses.txt',train_losses,delimiter=',')
            

def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)
        


if __name__ == '__main__':
    main()

