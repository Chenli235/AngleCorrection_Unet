# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 13:13:58 2021

@author: ische
"""

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
import scipy as sp
import scipy.ndimage

def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

def is_tiff(filename):
    return any(filename.endswith(extension) for extension in ['.tif','.tiff'])

def generate_onemask():
    input_image_path = "test_images/693_26_1.tiff"
    img = skimage.io.imread(input_image_path)
    img[img>10000] = 10000
    img = sp.ndimage.filters.gaussian_filter(img,[1,1],mode = 'reflect')
    mask  = np.zeros(img.shape)
    mask_  = np.zeros(img.shape)
    mask[img<=300] = 0.0
    mask[img>300] = 255.0
    im = Image.fromarray(mask)
    im_black = Image.fromarray(mask_)
    im.convert('L').save('mask.png')
    im_black.convert('L').save('mask_black.png')
    
if __name__ == '__main__':
    input_image_path1 = "../data/defocus_dataset/train"
    input_image_path2 = "../data/defocus_dataset/test"
    output_mask_path = "../data/defocus_dataset/mask"
    for i in range(1,52):
        makedirs(output_mask_path + '/' + str(i))
    # convert image into mask one by one
    for i in range(1,52):
        print(i)
        source_image_path_train = input_image_path1 + '/' + str(i)
        source_image_path_test = input_image_path2 + '/' + str(i)
        image_files_train = np.array([join(source_image_path_train,file_name) for file_name in sorted(listdir(source_image_path_train)) if is_tiff(file_name)])
        image_files_test = np.array([join(source_image_path_test,file_name) for file_name in sorted(listdir(source_image_path_test)) if is_tiff(file_name)])
        # read images one by one len(image_files_train)
        for index in range(len(image_files_train)):
            #print(image_files_train[index])
            img = skimage.io.imread(image_files_train[index])
            img_name = image_files_train[index][-28:-5]
            img[img>10000] = 10000
            img = sp.ndimage.filters.gaussian_filter(img,[1,1],mode = 'reflect')
            mask  = np.zeros(img.shape)
            mask[img<=300] = 0.0
            mask[img>300] = 255.0
            im = Image.fromarray(mask)
            im.convert('L').save(output_mask_path + '/' + str(i) + '/' + img_name + '.png')
        
        for index in range(len(image_files_test)):
            #print(image_files_train[index])
            img = skimage.io.imread(image_files_test[index])
            img_name = image_files_test[index][-28:-5]
            img[img>10000] = 10000
            img = sp.ndimage.filters.gaussian_filter(img,[1,1],mode = 'reflect')
            mask  = np.zeros(img.shape)
            mask[img<=300] = 0
            mask[img>300] = 255
            im = Image.fromarray(mask)
            im.convert('L').save(output_mask_path + '/' + str(i) + '/' + img_name + '.png')
            