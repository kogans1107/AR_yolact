# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 13:39:00 2020

@author: Bill
"""
import glob
import pickle
import matplotlib.pyplot as plt

import tkinter as tk
from tkinter.filedialog import FileDialog
import time
import platform as platf
import numpy as np

from eval import APDataObject

from sklearn.metrics import auc

def uichoosedir(title = None, initialdir = None):
    root = tk.Tk()
    root.focus_force()
    root.withdraw() # we don't want a full GUI, so keep the root window 
                    #  from appearing
    pathname = tk.filedialog.askdirectory(title=title, initialdir = initialdir)
    return pathname


dir_to_process = uichoosedir()
ap_data_files = sorted(glob.glob(dir_to_process + '/'+ 'ap_data*'))

with open(ap_data_files[0],'rb') as f:
    ap_data = pickle.load(f)


classes = [c.classname for c in ap_data['mask'][0] ]
#
#  pretend I have a list of strings for classes
obj_dict = {'classes',[]}
for c in classes:
    obj_dict['classes'].append({'name': c, 'mask':[]},{'box':[] ,'true_positives': []})

   

#epoch_number = ???? # np.linspace(0,number of files?)
epoch_number = []

for num, fname in enumerate(ap_data_files):
    epoch_number.append(num)
    
    with open(fname,'rb') as f:
        apdata = pickle.load(f)
        stuff = apdata['mask'][0][0]
        classes = stuff.classnames
        # how to get class names, but if we have classnames...
        for c in classnames:
            # if I knew how to get the ap dump for object type c
            ap_dump_obj = []
            obj_dict[c].append(ap_dump_obj.get_ap()) # not sure how to fit this in with recall bins
            # what we really want is one number for each object type for each dump file. 
            

plt.subplots(2,2,1)
plt.plot(epoch_num, obj_dict[classname[23]])
plt.ylim((0,1))

plist=[]
for i in range(len(classes)):
    plot_num = (i % 4)+1
    plt.subplot(2,2,plot_num)
    