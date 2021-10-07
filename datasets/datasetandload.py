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
from random import choice
from os import listdir
from os.path import join
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms,utils
import matplotlib.pyplot as plt

import skimage.io
import skimage.transform
import skimage.color
import skimage
from PIL import Image

image_size = 512

def is_tiff(filename):
    return any(filename.endswith(extension) for extension in ['.tif','.tiff'])

class Defocus_dataset_2(Dataset):
    def __init__(self,image_dir,labeled,train_mode = True,transform=None,random_shuffle=True):
        self.images_dir = image_dir + '/train'
        self.infocus_dir = self.images_dir + '/26'
        self.mask_dir = image_dir + '/mask'
        self.labeled = labeled
        self.transform = transform
        self.distance = 6
        image_txt = 'img_512.txt'
        
        if train_mode == True:
            image_list = []
            imagefile_512 = os.path.join(image_dir, image_txt)
            with open(imagefile_512,'r') as f:
                while True:
                    line = f.readline()
                    if not line:
                        break
                    image_list.append(line)
            self.image_files = np.array(image_list)
        
        if random_shuffle:
            shuffle(self.image_files)
        if self.distance == 6:
            self.levels = [8,11,14,17,20,23,26,29,32,35,38,41,44] # 6um
            
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self,index):
        img,label,mask = self.read_image_mask(index)
        sample = {'image':img,'mask':mask,'label':label}
        if self.transform:
            sample = self.transform(sample)
        return sample
        
    def read_label(self,filename):
        if self.distance == 6:
            if filename.endswith('_08.tiff'):
                return 0
            elif filename.endswith('_11.tiff'):
                return 1
            elif filename.endswith('_14.tiff'):
                return 2
            elif filename.endswith('_17.tiff'):
                return 3
            elif filename.endswith('_20.tiff'):
                return 4
            elif filename.endswith('_23.tiff'):
                return 5
            elif filename.endswith('_26.tiff'):
                return 6
            elif filename.endswith('_29.tiff'):
                return 7
            elif filename.endswith('_32.tiff'):
                return 8
            elif filename.endswith('_35.tiff'):
                return 9
            elif filename.endswith('_38.tiff'):
                return 10
            elif filename.endswith('_41.tiff'):
                return 11
            elif filename.endswith('_44.tiff'):
                return 12
    
    def read_image_mask(self,index):
        match_word = self.image_files[index][:20]
        level_1 = choice(self.levels)
        if self.distance == 6:
            level_0 = level_1 - 3
            level_2 = level_1 + 3
        image_path_1 = self.images_dir + '/' + str(level_1)
        image_path_2 = self.images_dir + '/' + str(level_2)
        mask_path = self.mask_dir + '/' + str(level_1)
        img_name = image_path_1 + '/' + match_word + '_' +str(level_1).zfill(2) + '.tiff'
        # read image and process
        image_1 = skimage.io.imread(image_path_1 + '/' + match_word + '_' +str(level_1).zfill(2) + '.tiff')
        image_2 = skimage.io.imread(image_path_2 + '/' + match_word + '_' +str(level_2).zfill(2) + '.tiff')
        image_1[image_1>10000] = 10000
        image_2[image_2>10000] = 10000
        
        x = random.randint(0, image_1.shape[1]-image_size)
        y = random.randint(0, image_1.shape[0]-image_size)
        img_1 = image_1[y:y+image_size,x:x+image_size]
        img_2 = image_2[y:y+image_size,x:x+image_size]
        img = np.dstack((img_1,img_2))
        if np.random.rand() > 0.8:
            scale = np.random.uniform(0.95,1.1)
            img = img*scale
        # read mask
        mask = skimage.io.imread(mask_path + '/' + match_word + '_' + str(level_1).zfill(2) + '.png')
        mask = mask[y:y+image_size,x:x+image_size]
        mask = mask/255.0
        label = self.read_label(img_name)
        # mask_n_classes = torch.zeros(13,512,512)
        # mask_n_classes[label,:,:] = mask
        return img,label,mask
    
class Normalizer(object):
    def __init__(self,mean=None,std=None):
        if mean == None:
            self.mean = np.array([[[0]]])
        else:
            self.mean = mean
        if std == None:
            self.std = np.array([[[10000-0]]])
        else:
            self.std = std
    def __call__(self,sample):
        image,mask = sample['image'].astype(np.float32),sample['mask']
        sample = {'image':((image-self.mean)/self.std),'mask':mask,'label':sample['label']}
        return sample

class Augmenter(object):
    def __call__(self,sample):
        image = sample['image'].astype(np.float32)
        mask = sample['mask'].astype(np.float32)
        label = sample['label']
        if np.random.rand() < 0.2:
            image = image[:,::-1,:]
            mask = mask[:,::-1]
        if np.random.rand() < 0.2:
            image = image[::-1,:,:]
            mask = mask[::-1,:]
        mask_n_classes = np.zeros((13,512,512))
        mask_n_classes[label,:,:] = mask
        return {'image':torch.from_numpy(image.copy()).type(torch.FloatTensor).permute(2,0,1),'mask':torch.from_numpy(mask_n_classes.copy()).type(torch.FloatTensor)}



