# -*- coding: utf-8 -*-
"""
Created on Sun Jul 25 09:40:42 2021

@author: chen li
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model import Unet_defocus
import skimage.io
import skimage.transform
import skimage.color
import skimage
import logging
import scipy.misc
import scipy.stats
import matplotlib.pyplot as plt

import scipy as sp
import scipy.ndimage
import heapq

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
import scipy.io as io
from PIL import Image
from utils import *
import time

logging.getLogger().setLevel(logging.DEBUG)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
image_size = 512

def draw_trainloss():
    file = open('results/GAN_20210728-204833/train_losses.txt')
    value = []
    while 1:
        lines = file.readlines(10000)
        if not lines:
            break
        for line in lines:
            value.append(float(line))
            pass
    file.close()
    plt.plot(np.array(value))
    

def cal_certainty(prob):
    sum_prob = np.sum(prob)
    num_classes = prob.shape[0]
    if sum_prob > 0:
        normalized_prob = prob/sum_prob
        certain_pro = 1.0 - scipy.stats.entropy(normalized_prob.flatten())/np.log(num_classes)
    else:
        certain_pro = 0
    return certain_pro

def test_one_image_512():
    model_path = 'results/GAN_20210728-204833/net_500.pt'
    classes_num = 13
    model = Unet_defocus(2,classes_num).to(device)
    model.load_state_dict(torch.load(model_path))
    # image size need to be 512
    img1_path = 'test_images/906_32_1.tiff'
    img2_path = 'test_images/906_32_2.tiff'
    image_1 = skimage.io.imread(img1_path)
    image_2 = skimage.io.imread(img2_path)
    image_1[image_1>10000] = 10000
    image_2[image_2>10000] = 10000
    img = np.dstack((image_1,image_2)).astype(np.float32)
    img = np.expand_dims(img,axis = 0)
    img = img/10000.0
    img = torch.from_numpy(img.copy()).type(torch.FloatTensor).permute(0,3,1,2).contiguous().to(device)
    with torch.no_grad():
        model.eval()
        output = model(img)
        #probs = F.softmax(output,dim=1)
        probs = F.softmax(output,dim=1).cpu().detach().numpy()
        #pred = torch.argmax(probs,dim=1).cpu().detach().numpy()
        # cert = np.zeros((512,512))
        # for i in range(output.shape[2]):
        #     for j in range(output.shape[3]):
        #         cert[i,j] = cal_certainty(prob[0,:,i,j])
    pred = output.argmax(dim=1,keepdim=True).cpu().detach().numpy()
    print(output.shape)
    prediction_img = np.zeros((512,512))
    source_img = sp.ndimage.filters.gaussian_filter(image_1,[1,1],mode = 'reflect')
    for i in range(output.shape[2]):
            for j in range(output.shape[3]):
                # background value set to 100
                if source_img[i,j]<=300 or cal_certainty(probs[0,:,i,j])<0.3:
                    prediction_img[i,j] = 100 # meaning the pixel belongs to background
                else:
                    #prob = probs[0,:,i,j]
                    #max_index = heapq.nlargest(3, range(len(prob)), prob.take)
                    #pred_defocus_level = sum(prob[max_index]/sum(prob[max_index])*max_index)
                    prediction_img[i,j] = pred[0,0,i,j]
    return prediction_img
    
def pred_convert_img(pred_img):
    img = np.zeros((pred_img.shape[0],pred_img.shape[1],3))
    for i in range(pred_img.shape[0]):
        for j in range(pred_img.shape[1]):
            if pred_img[i,j] == 100:
                continue
            else:
                img[i,j,:] = get_class_rgb(pred_img[i,j])
    return img
    
    
    
    
def generate_colorbar():
    fig, ax = plt.subplots(figsize=(1, 6))
    fig.subplots_adjust(bottom=0.5)
    
    cmap = mpl.cm.rainbow
    norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)
    
    cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                    norm=norm,
                                    orientation='horizontal',ticks=[])
    #cb1.set_label('Some Units')
    fig.show()
    return cb1

def generate_defocus_color(predict_img):
    img = np.zeros((predict_img.shape[1],predict_img.shape[1],3))
    rainbow = plt.get_cmap('rainbow')
    #draw pixel value one by one
    for i in range(predict_img.shape[0]):
        for j in range(predict_img.shape[1]):
            if predict_img[i,j] == 100:
                pass
            else:
                rgba = rainbow(predict_img[i,j])
                for channel in range(3):
                    img[i,j,channel] = rgba[channel]
                
    return img
def test_one_image_1024():
    model_path = 'results/GAN_20210728-204833/net_500.pt'
    classes_num = 13
    model = Unet_defocus(2,classes_num).to(device)
    model.load_state_dict(torch.load(model_path))
    
    img1_path = "test_images/961898_anglecorrection_GalvoX_0.14977_GalvoY_0.047227_GalvoRoll_-0.019153_detectionlens_-0.0043_1.tiff"
    img2_path = "test_images/961898_anglecorrection_GalvoX_0.14977_GalvoY_0.047227_GalvoRoll_-0.019153_detectionlens_-0.0043_2.tiff"
    start = time.time()
    image_1 = skimage.io.imread(img1_path)
    image_2 = skimage.io.imread(img2_path)
    image_1[image_1>10000] = 10000
    image_2[image_2>10000] = 10000
    image_1 = image_1[1324-512:1324+512,1324-512:1324+512]
    image_2 = image_2[1324-512:1324+512,1324-512:1324+512]
    #image_1 = image_1[:,::-1]
    #image_2 = image_2[:,::-1]
    img = np.dstack((image_1,image_2)).astype(np.float32)
    img = np.expand_dims(img,axis = 0)
    img = img/10000.0
    img = torch.from_numpy(img.copy()).type(torch.FloatTensor).permute(0,3,1,2)
    output = torch.zeros(1,13,1024,1024)
    pred = np.zeros((1,1,10242,1024))
    with torch.no_grad():
        model.eval()
        for i in range(1024//512):
            for j in range(1024//512):
                img_ = img[:,:,i*512:i*512+512,j*512:j*512+512].contiguous().to(device)
                output[:,:,i*512:i*512+512,j*512:j*512+512] = model(img_)
                pred[:,:,i*512:i*512+512,j*512:j*512+512] = output[:,:,i*512:i*512+512,j*512:j*512+512].argmax(dim=1,keepdim=True).cpu().detach().numpy()
                
    probs = F.softmax(output,dim=1).cpu().detach().numpy()
    prediction_img = np.zeros((1024,1024))
    source_img = sp.ndimage.filters.gaussian_filter(image_1,[1,1],mode = 'reflect')
    for i in range(output.shape[2]):
            for j in range(output.shape[3]):
                # background value set to 100
                if source_img[i,j]<=300 or cal_certainty(probs[0,:,i,j])<0.3:
                    prediction_img[i,j] = 100 # meaning the pixel belongs to background
                else:
                    #prob = probs[0,:,i,j]
                    prediction_img[i,j] = pred[0,0,i,j]
                    
    end = time.time()
    print(end-start)
    return prediction_img
def cal_certainty_fast(probs):
    sum_probs = np.sum(probs,axis = 1)
    normalized_probs = probs/sum_probs
    cert_img = 1 - scipy.stats.entropy(normalized_probs,axis = 1)/np.log(13)
    return cert_img.squeeze()
    
    
def test_one_image_1024_fast():
    model_path = 'results/GAN_20210728-204833/net_500.pt'
    classes_num = 13
    model = Unet_defocus(2,classes_num).to(device)
    model.load_state_dict(torch.load(model_path))
    
    img1_path = "test_images_new/157613_defocuscorrection_GalvoX_0.054839_GalvoY_0.3165_GalvoRoll_0.0073971_detectionlens_0.01979_1.tiff"
    img2_path = "test_images_new/157613_defocuscorrection_GalvoX_0.054839_GalvoY_0.3165_GalvoRoll_0.0073971_detectionlens_0.01979_2.tiff"
    start = time.time()
    image_1 = skimage.io.imread(img1_path)
    image_2 = skimage.io.imread(img2_path)
    image_1[image_1>10000] = 10000
    image_2[image_2>10000] = 10000
    image_1 = image_1[1024-512:1024+512,1024-512:1024+512]
    image_2 = image_2[1024-512:1024+512,1024-512:1024+512]
    #image_1 = image_1[:,::-1]
    #image_2 = image_2[:,::-1]
    img = np.dstack((image_1,image_2)).astype(np.float32)
    img = np.expand_dims(img,axis = 0)
    img = img/10000.0
    img = torch.from_numpy(img.copy()).type(torch.FloatTensor).permute(0,3,1,2)
    output = torch.zeros(1,13,1024,1024)
    pred = np.zeros((1,1,1024,1024))
    with torch.no_grad():
        model.eval()
        for i in range(1024//512):
            for j in range(1024//512):
                img_ = img[:,:,i*512:i*512+512,j*512:j*512+512].contiguous().to(device)
                output[:,:,i*512:i*512+512,j*512:j*512+512] = model(img_)
                pred[:,:,i*512:i*512+512,j*512:j*512+512] = output[:,:,i*512:i*512+512,j*512:j*512+512].argmax(dim=1,keepdim=True).cpu().detach().numpy()
    
    probs = F.softmax(output,dim=1).cpu().detach().numpy()
    prediction_img = np.zeros((1024,1024))
    prediction_img = pred[0,0,:,:]
    source_img = sp.ndimage.filters.gaussian_filter(image_1,[1,1],mode = 'reflect')
    cert_img = cal_certainty_fast(probs)
    print(cert_img.squeeze().shape)
    print(source_img.shape)
    print(cert_img.shape)
    prediction_img[source_img<=300] = 100
    prediction_img[cert_img<=0.3] = 100
    # for i in range(output.shape[2]):
    #         for j in range(output.shape[3]):
    #             # background value set to 100
    #             if source_img[i,j]<=300 or cal_certainty(probs[0,:,i,j])<0.3:
    #                 prediction_img[i,j] = 100 # meaning the pixel belongs to background
    #             else:
    #                 #prob = probs[0,:,i,j]
    #                 prediction_img[i,j] = pred[0,0,i,j]
                    
    end = time.time()
    print(end-start)
    return prediction_img

def test_one_image_512_fast():
    model_path = 'results/GAN_20210728-204833/net_500.pt'
    classes_num = 13
    model = Unet_defocus(2,classes_num).to(device)
    model.load_state_dict(torch.load(model_path))
    
    img1_path = "test_images/961898_original_GalvoX_0.18516_GalvoY_-0.053165_GalvoRoll_-0.034161_detectionlens_-0.0044_1.tiff"
    img2_path = "test_images/961898_original_GalvoX_0.18516_GalvoY_-0.053165_GalvoRoll_-0.034161_detectionlens_-0.0044_2.tiff"
    start = time.time()
    image_1 = skimage.io.imread(img1_path)
    image_2 = skimage.io.imread(img2_path)
    image_1[image_1>10000] = 10000
    image_2[image_2>10000] = 10000
    image_1 = image_1[1260-256:1260+256,1260-256:1260+256]
    image_2 = image_2[1260-256:1260+256,1260-256:1260+256]
    img = np.dstack((image_1,image_2)).astype(np.float32)
    img = np.expand_dims(img,axis = 0)
    img = img/10000.0
    img = torch.from_numpy(img.copy()).type(torch.FloatTensor).permute(0,3,1,2)
    output = torch.zeros(1,13,1024,1024)
    pred = np.zeros((1,1,1024,1024))
    with torch.no_grad():
        model.eval()
        img_ = img.contiguous().to(device)
        output = model(img_)
        pred = output.argmax(dim=1,keepdim=True).cpu().detach().numpy()
        
    
    probs = F.softmax(output,dim=1).cpu().detach().numpy()
    prediction_img = np.zeros((512,512))
    prediction_img = pred[0,0,:,:]
    source_img = sp.ndimage.filters.gaussian_filter(image_1,[1,1],mode = 'reflect')
    cert_img = cal_certainty_fast(probs)
    print(cert_img.squeeze().shape)
    print(source_img.shape)
    print(cert_img.shape)
    prediction_img[source_img<=300] = 100
    prediction_img[cert_img<=0.3] = 100
    # for i in range(output.shape[2]):
    #         for j in range(output.shape[3]):
    #             # background value set to 100
    #             if source_img[i,j]<=300 or cal_certainty(probs[0,:,i,j])<0.3:
    #                 prediction_img[i,j] = 100 # meaning the pixel belongs to background
    #             else:
    #                 #prob = probs[0,:,i,j]
    #                 prediction_img[i,j] = pred[0,0,i,j]
                    
    end = time.time()
    print(end-start)
    return prediction_img



def test_one_image_2048():
    
    model_path = 'results/GAN_20210728-204833/net_500.pt'
    classes_num = 13
    model = Unet_defocus(2,classes_num).to(device)
    model.load_state_dict(torch.load(model_path))
    
    img1_path = "test_images/157613_defocuscorrection_GalvoX_0.054839_GalvoY_0.3165_GalvoRoll_0.0073971_detectionlens_0.01979_1.tiff"
    img2_path = "test_images/157613_defocuscorrection_GalvoX_0.054839_GalvoY_0.3165_GalvoRoll_0.0073971_detectionlens_0.01979_2.tiff"
    image_1 = skimage.io.imread(img1_path)
    image_2 = skimage.io.imread(img2_path)
    image_1[image_1>10000] = 10000
    image_2[image_2>10000] = 10000
    img = np.dstack((image_1,image_2)).astype(np.float32)
    img = np.expand_dims(img,axis = 0)
    img = img/10000.0
    img = torch.from_numpy(img.copy()).type(torch.FloatTensor).permute(0,3,1,2)
    
    output = torch.zeros(1,13,2048,2048)
    pred = np.zeros((1,1,2048,2048))
    with torch.no_grad():
        model.eval()
        for i in range(2048//512):
            for j in range(2048//512):
                img_ = img[:,:,i*512:i*512+512,j*512:j*512+512].contiguous().to(device)
                output[:,:,i*512:i*512+512,j*512:j*512+512] = model(img_)
                pred[:,:,i*512:i*512+512,j*512:j*512+512] = output[:,:,i*512:i*512+512,j*512:j*512+512].argmax(dim=1,keepdim=True).cpu().detach().numpy()
            
    probs = F.softmax(output,dim=1).cpu().detach().numpy()
    prediction_img = np.zeros((2048,2048))
    source_img = sp.ndimage.filters.gaussian_filter(image_1,[1,1],mode = 'reflect')
    for i in range(output.shape[2]):
            for j in range(output.shape[3]):
                # background value set to 100
                if source_img[i,j]<=300 or cal_certainty(probs[0,:,i,j])<0.3:
                    prediction_img[i,j] = 100 # meaning the pixel belongs to background
                else:
                    #prob = probs[0,:,i,j]
                    prediction_img[i,j] = pred[0,0,i,j]
                    #max_index = heapq.nlargest(3, range(len(prob)), prob.take)
                    #pred_defocus_level = sum(prob[max_index]/sum(prob[max_index])*max_index)
                    #prediction_img[i,j] = (pred_defocus_level)/12
    return prediction_img
    
        
        
if __name__ == "__main__":
    
    # pred_img = test_one_image_512()
    # img_defocus = pred_convert_img(pred_img)
    # plt.imsave('defocus_img.tiff',img_defocus)
    # img_defocus = generate_defocus_color(pred_img)
    # plt.imsave('defocus_img.tiff',img_defocus)
    # pred_img = test_one_image_2048()
    # img_defocus = generate_defocus_color(pred_img)
    # plt.imsave('defocus_img.tiff',img_defocus)
    # Image.fromarray(pred_img).save('pred_img.tif')
    pred_img = test_one_image_1024_fast()
    img_defocus = pred_convert_img(pred_img)
    plt.imsave('defocus_img.tiff',img_defocus)
    
    
    
    