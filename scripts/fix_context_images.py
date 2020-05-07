# -*- coding: utf-8 -*-
"""
Created on Thu May  7 11:32:31 2020

@author: Bill
"""

import numpy as np
import matplotlib.pyplot as plt 
import imageio
import skimage.transform
import glob

best_shape = (550,550)
non_iconic_path = '../data/non_iconic/'



imgfiles = glob.glob(non_iconic_path+'*.jpg')

for imgfile in imgfiles:
    if 'FIXED' in imgfile:
        continue
    img = np.asarray(imageio.imread(imgfile))
    
    imgshape = img.shape
    tall = imgshape[0] > imgshape[1]
    if tall:
        img = img[0:imgshape[1], :]
    else:
        img = img[:, 0:imgshape[0]]
            
    img = skimage.transform.resize(img, best_shape)
    
    imlist = imgfile.split('.')
    imlist[-2] += '_FIXED'
    imgfilefix = str.join('.', imlist)
    imageio.imsave(imgfilefix, img)