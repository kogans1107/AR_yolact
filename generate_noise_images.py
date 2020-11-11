# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 14:12:19 2020

@author: Bill
"""
import numpy as np
import matplotlib.pyplot as plt

def make_some_noise(H, W, lam):
    
    nsize = (H, W, 3)   
    noise = np.random.poisson(lam=lam, size=nsize).astype(np.uint8)
    return noise

def scl(x):
    return (x-np.min(x))/(np.max(x)-np.min(x))

if __name__ == "__main__":
    H = [480, 550, 768]
    W = [640, 550, 1024]
    nimgs = 100
    
    picksize = np.random.randint(0, high=3, size=(nimgs))
    lam = 20
    
    for i,p in enumerate(picksize):
        img = make_some_noise(H[p], W[p], lam)
        plt.imsave('./data/noise/noise_' + str(i) + '.jpg', img)
    
    
    