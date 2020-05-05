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
#
#    Is IoU a way to get precision? Precision answers "what fraction of the positives
#   are true positives? 
#
#   Recall answers "what fraction of the ground truth positives are recorded as 
#     positives?" 
#
#   Precision answers a question about all the pixels for which YOLACT said "I see 
#     something here."  It is the fraction of those that are correct. 
# 
#   Recall answers a question about the ground truth positive pixels. It is 
#    the fraction of those that YOLACT called posiitve. 
# 
#   Intersection over union is true positives over all positives. So it is 
#      exactly precision. 




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
        
#    THIS IS MY CLASS LIST FOR DEBUGGING WHEN I HAD THE WRONG CONFIG FILE 
#    classes = \
#    ['Cooling Block', 'Ice Bucket', 'N3 (Dark Green Bee)', 'RP (Orange Smile)', \
#     'Sterile 1.5ml Tube', 'Taqpath (Yellow Star)', 'Trash', '1.5-2ml Rack',\
#     '15ml Capped Tube', '15ml Tube Rack', 'N1 (Red Good)', 'N2 (Dark Blue Bee)',\
#     'Nuclease Free Water', 'PCR Plate', 'Spin Column', 'Bunsen Burner', 'Thumb', \
#     'Fingers', 'GeneDrive', 'Hand', 'Laptop', 'Micropipette', 'Tip Box', 'PCR Machine',\
#     'RT-PCR (Red Ladybug)', 'Vortex', '1.5ml Tube (Pink Flower)', '2ml Tube (Blue Smile)',\
#     'RPE Buffer (Blue)', 'RWI Buffer (Green)', 'RNase Water (Orange)', 'RLT Buffer (Pink)',\
#     'Opaque Rack Cover', 'nCOVPC (Green Flower)', 'Gloves',\
#     'Micropipette (Nucleic Acid Yellow)', 'Micropipette (Reagant Orange)',\
#     'Tip Box (Reagant Orange)', 'Tip Box (Nucleic Acid Yellow)', 'Striker Flint',\
#     'Micropipette Tip', 'Microcentrifuge', 'Serological Pipette', 'Electronic Pipet-Aid',\
#     'VWR Marker', 'Micropipette Tippy End', 'Nucleic Acid Decontaminant']
    
    classes = list(classes)
    
    try:
        assert(len(classes)==nobj)
    except AssertionError:
        print('len(classes) is not equal to the number of objects')
        print(len(classes),nobj)
        if len(classes) < nobj:
            delt = nobj - len(classes)
            for i in range(delt):
                classes.append('crap')

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
    
    # KTO For today just F9 one or both of these. They will give you a per-object display
    #   of scores, and a corresponding (by rows) list of classes. They are both sorted from 
    #   worst to best. The important thing is to know which objects are the worst. 
    #
#    img, names = build_AP_display(ap['boxes'])
    img, names = build_AP_display(ap['mask'])
    
    for i,n in enumerate(names):
#        print(n, '\t\t\t\t\t', np.mean(img[i,:]))
        tabslist = ['\t']*((48-len(n))//8)
        tabs=''
        for t in tabslist:
            tabs = tabs + t
            
        print((n + tabs + '{:5.3f}').format( np.mean(img[i,:])))
        
    