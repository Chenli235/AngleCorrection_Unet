# -*- coding: utf-8 -*-
"""
Created on Sun Jul 25 09:40:42 2021

@author: chen li
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model import Unet_defocus, paper1_net
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

def test_one_image_512(image_1,image_2):
    model_path = 'net_500_Unet.pt'
    classes_num = 13
    model = Unet_defocus(2,classes_num).to(device)
    model.load_state_dict(torch.load(model_path))
    # image size need to be 512
    #img1_path = 'test_images/906_32_1.tiff'
    #img2_path = 'test_images/906_32_2.tiff'
    #image_1 = skimage.io.imread(img1_path)
    #image_2 = skimage.io.imread(img2_path)
    image_1 = np.array(image_1)
    image_2 = np.array(image_2)
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
    
    prediction_img = np.zeros((512,512))
    pred = output.argmax(dim=1,keepdim=True).cpu().detach().numpy()
    source_img = sp.ndimage.filters.gaussian_filter(image_1,[1,1],mode = 'reflect')
    for i in range(output.shape[2]):
            for j in range(output.shape[3]):
                # background value set to 100
                if source_img[i,j]<=300 or cal_certainty(probs[0,:,i,j])<0.3:
                #if cal_certainty(probs[0,:,i,j])<0.3:
                    prediction_img[i,j] = 100 # meaning the pixel belongs to background
                else:
                    #prob = probs[0,:,i,j]
                    #max_index = heapq.nlargest(3, range(len(prob)), prob.take)
                    #pred_defocus_level = sum(prob[max_index]/sum(prob[max_index])*max_index)
                    #prediction_img[i,j] = (pred_defocus_level)/12
                    prediction_img[i,j] = pred[0,0,i,j];
    return prediction_img
    
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

def test_one_image_2048(image_1,image_2):
    
    model_path = 'net_500_Unet.pt'
    classes_num = 13
    model = Unet_defocus(2,classes_num).to(device)
    model.load_state_dict(torch.load(model_path))
    
    #img1_path = "test_images/Galvoy_0.09_GalvoRoll_0_655477_-0.0088_1.tiff"
    #img2_path = "test_images/Galvoy_0.09_GalvoRoll_0_655477_-0.0028_2.tiff"
    #image_1 = skimage.io.imread(img1_path)
    #image_2 = skimage.io.imread(img2_path)
    image_1 = np.array(image_1)
    image_2 = np.array(image_2)
    image_1[image_1>10000] = 10000
    image_2[image_2>10000] = 10000
    img = np.dstack((image_1,image_2)).astype(np.float32)
    img = np.expand_dims(img,axis = 0)
    img = img/10000.0
    img = torch.from_numpy(img.copy()).type(torch.FloatTensor).permute(0,3,1,2)
    
    output = torch.zeros(1,13,img.shape[2],img.shape[3])
    with torch.no_grad():
        model.eval()
        for i in range(img.shape[2]//512):
            for j in range(img.shape[2]//512):
                img_ = img[:,:,i*512:i*512+512,j*512:j*512+512].contiguous().to(device)
                output[:,:,i*512:i*512+512,j*512:j*512+512] = model(img_)
                
            
    probs = F.softmax(output,dim=1).cpu().detach().numpy()
    #pred = output.argmax(dim=1,keepdim=True).cpu().detach().numpy()
    prediction_img = np.zeros((output.shape[2],output.shape[3]))
    source_img = sp.ndimage.filters.gaussian_filter(image_1,[1,1],mode = 'reflect')
    for i in range(output.shape[2]):
            for j in range(output.shape[3]):
                # background value set to 100
                if source_img[i,j]<=300 or cal_certainty(probs[0,:,i,j])<0.35:
                #if cal_certainty(probs[0,:,i,j])<0.35:
                    prediction_img[i,j] = 100 # meaning the pixel belongs to background
                else:
                    prob = probs[0,:,i,j]
                    max_index = heapq.nlargest(3, range(len(prob)), prob.take)
                    pred_defocus_level = sum(prob[max_index]/sum(prob[max_index])*max_index)
                    pred_defocus_level = max_index[0]
                    prediction_img[i,j] = (pred_defocus_level)/12
    Image.fromarray(prediction_img).save('prediction_img.tif')
    return prediction_img

def img_to_batches(img):
    height = img.shape[2]
    width = img.shape[3]
    image_size = 128
    num = (height/128)*(width/128)
    image_batches = np.zeros((int(num),2,image_size,image_size))
    height_num = int(height/image_size)
    width_num = int(width/image_size)
    for i in range(height_num):
        for j in range(width_num):
            image_batches[i*width_num+j,:,:,:] = img[0,:,i*image_size:i*image_size+image_size,j*image_size:j*image_size+image_size]
    return image_batches


def cal_certainty(prob):
    sum_prob = np.sum(prob)
    num_classes = prob.shape[0]
    
    if sum_prob > 0:
        normalized_prob = prob/sum_prob
        certain_proxy = 1.0 - scipy.stats.entropy(normalized_prob)/np.log(num_classes)
    else:
        certain_proxy = 0.0
    certain_proxy = np.clip(certain_proxy,0.0,1.0)
    #print(certain_proxy)
    return certain_proxy

def get_certainty(prob):
    num_batches = prob.shape[0]
    #num_classes = prob.shape[1]
    cert = np.zeros(num_batches)
    for i in range(num_batches):
        cert[i] = cal_certainty(prob[i])
        
    return cert

def test_one_image_2048_paper1(image_1,image_2):
    model_path = 'selfmade_net_6nm_20000_paper1.pt'
    classifier = paper1_net().to(device)
    classifier.load_state_dict(torch.load(model_path))
    image_1 = np.array(image_1)
    image_2 = np.array(image_2)
    image_1[image_1>10000] = 10000
    image_2[image_2>10000] = 10000
    img = np.dstack((image_1,image_2)).astype(np.float32)
    img = np.expand_dims(img,axis = 0)
    img = img/10000.0
    img = torch.from_numpy(img.copy()).type(torch.FloatTensor).permute(0,3,1,2)
    img_batches = img_to_batches(img)
    
    with torch.no_grad():
        classifier.eval()
        output = classifier(torch.from_numpy(img_batches.copy()).type(torch.FloatTensor).to(device))
    
    pred = output.argmax(dim=1,keepdim=True).cpu().detach().numpy()
    
    prob = F.softmax(output,dim=1).cpu().detach().numpy()
    cert = get_certainty(prob)
    return pred,cert,prob
def test_one_image_512_paper1(image_1,image_2):
    model_path = 'selfmade_net_6nm_20000_paper1.pt'
    classifier = paper1_net().to(device)
    classifier.load_state_dict(torch.load(model_path))
    image_1 = np.array(image_1)
    image_2 = np.array(image_2)
    image_1[image_1>10000] = 10000
    image_2[image_2>10000] = 10000
    img = np.dstack((image_1,image_2)).astype(np.float32)
    img = np.expand_dims(img,axis = 0)
    img = img/10000.0
    img = torch.from_numpy(img.copy()).type(torch.FloatTensor).permute(0,3,1,2)
    img_batches = img_to_batches(img)
    with torch.no_grad():
        classifier.eval()
        output = classifier(torch.from_numpy(img_batches.copy()).type(torch.FloatTensor).to(device))
    
    pred = output.argmax(dim=1,keepdim=True).cpu().detach().numpy()
    prob = F.softmax(output,dim=1).cpu().detach().numpy()
    cert = get_certainty(prob)
    
    return pred,cert,prob

def test_one_image_128_paper1(image_1,image_2):
    model_path = 'selfmade_net_6nm_20000_paper1.pt'
    classifier = paper1_net().to(device)
    classifier.load_state_dict(torch.load(model_path))
    image_1 = np.array(image_1)
    image_2 = np.array(image_2)
    image_1[image_1>10000] = 10000
    image_2[image_2>10000] = 10000
    img = np.dstack((image_1,image_2)).astype(np.float32)
    img = np.expand_dims(img,axis = 0)
    img = img/10000.0
    img = torch.from_numpy(img.copy()).type(torch.FloatTensor).permute(0,3,1,2)
    with torch.no_grad():
        classifier.eval()
        output = classifier(img.type(torch.FloatTensor).to(device))
    pred = output.argmax(dim=1,keepdim=True).cpu().detach().numpy()
    prob = F.softmax(output,dim=1).cpu().detach().numpy()
    cert = get_certainty(prob)
    return pred,cert


if __name__ == "__main__":
    
    # pred_img = test_one_image_512()
    # img_defocus = generate_defocus_color(pred_img)
    # plt.imsave('defocus_img.tiff',img_defocus)
    pred_img = test_one_image_2048()
    img_defocus = generate_defocus_color(pred_img)
    plt.imsave('defocus_img.tiff',img_defocus)
    Image.fromarray(pred_img).save('pred_img.tif')
    