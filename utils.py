# -*- coding: utf-8 -*-

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

def generate_colorbar():
    fig, ax = plt.subplots(figsize=(6, 1))
    fig.subplots_adjust(bottom=0.5)
    #cmap = mpl.colors.ListedColormap(['#e31d16','#e38416','#e3d816','#84e316','#16e36a','#16e3d1','#1697e3','#163de3','#5b16e3','#a816e3','#e3169c'])
    # RGB value from -40um to 40um
    cmap = mpl.colors.ListedColormap(['#FF0101','#FF5E01','#FF9D01','#FFDD01','#E2FF01','#AAFF01','#6BFF01','#01FF77','#01FFDF','#01C5FF','#018DFF','#013FFF','#6701FF'])
    cmap.set_over('0.25')
    cmap.set_under('0.75')
    bounds = [0,1,2,3,4,5,6,7,8,9,10,11,12,13]
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    fig.colorbar(
    mpl.cm.ScalarMappable(cmap=cmap, norm=norm),
    cax=ax,
    boundaries=bounds,
    extend='neither',
    ticks=[],
    spacing='proportional',
    orientation='horizontal',
    )
    #cb2.set_label('')
    fig.show()
    
    return 0

def get_class_rgb(pred):
    if pred == 0:
        return (255/255,1/255,1/255)
    elif pred == 1:
        return (255/255,94/255,1/255)
    elif pred == 2:
        return (255/255,157/255,1/255)
    elif pred == 3:
        return (255/255,221/255,1/255)
    elif pred == 4:
        return (226/255,255/255,1/255)
    elif pred == 5:
        return (170/255,255/255,1/255)
    elif pred == 6:
        return (107/255,255/255,1/255)
    elif pred == 7:
        return (1/255,255/255,119/255)
    elif pred == 8:
        return (1/255,255/255,223/255)
    elif pred == 9:
        return (1/255,197/255,255/255)
    elif pred == 10:
        return (1/255,141/255,255/255)
    elif pred == 11:
        return (1/255,63/255,255/255)
    elif pred == 12:
        return (103/255,1/255,255/255)

    
    
    
if __name__ == "__main__":
    generate_colorbar()
    
    
    
    