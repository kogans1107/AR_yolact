# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 09:06:04 2020

This loads the results that are written to the ./results folder by eval.py. These 
results consist of two dictionaries: one for boxes, one for masks. 

There are 10 bins of either confidence or recall in ap_data_dump. So the length of lists
ap['mask'] and ap['box'] is 10. I think these are the bins that get printed by 
eval.py when it finishes, which is averages over types of objects. When I use COCO, 
the things in each of these 10 bins are lists of length 80, for the 80 categories
of objects in COCO.  Each of the 80 elements of these lists are APDataObjects. 
The class APDataObject has a method get_ap(), which returns a number that is 
probably the per-object-type figure-of-merit that we want. 
 
@author: Bill
"""
import tkinter as tk
from tkinter import filedialog, simpledialog
import numpy as np
import platform as platf
import matplotlib.pyplot as plt
import pickle

import data as D  

#            precision = num_true / (num_true + num_false)
#            recall    = num_true / self.num_gt_positives

def argsort(seq):
    # http://stackoverflow.com/questions/3071415/efficient-method-to-calculate-the-rank-vector-of-a-list-in-python
    return sorted(range(len(seq)), key=seq.__getitem__)


def build_AP_display(apd):
    nobj = len(apd[0])
    try:
        label_map = D.KAR_LABEL_MAP
        classes = D.KAR_CLASSES
    except AttributeError:
        label_map = D.COCO_LABEL_MAP
        classes = D.COCO_CLASSES
    
    classes = list(classes)
    
    try:
        assert(len(label_map)==nobj)
    except AssertionError:
        print('len(label map) is not equal to the number of objects')

#    iou_thresholds = [x / 100 for x in range(50, 100, 5)]

    nbins = len(apd)
    img_display = np.zeros((nobj, nbins))


    
    for ic, cbin in enumerate(apd):
        for iobj in range(nobj):
            img_display[iobj, ic] = apd[ic][iobj].get_ap()
            
    
    obj_score = np.mean(img_display,axis=1)
    iord = np.argsort(obj_score)
    
    worst_first_objs = [classes[i] for i in iord]
    worst_first_image = img_display[iord,:]

    
    return worst_first_image, worst_first_objs

if __name__=='__main__':
    from eval import APDataObject  # need this defined so ap_data can be loaded. 

    root = tk.Tk() 
    top = tk.Toplevel(root)
    top.withdraw()  

    ap_file =  \
            filedialog.askopenfilename(parent=top, \
                                        title='Choose AP results file')

    try:        
        with open(ap_file,'rb') as f:
            ap = pickle.load(f)
    except:
        print('Trouble opening weights file.')
        
    
    