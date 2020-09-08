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

def get_tp(df):
       for i in range(class_list):
            mask_dict[i]['true_positives'].append(df['mask'][0][i].num_gt_positives)
            box_dict[i]['true_positives'].append(df['box'][0][i].num_gt_positives)
    


if __name__ == "__main__":
    
dir_to_process = uichoosedir()
ap_data_files = sorted(glob.glob(dir_to_process + '/'+ 'ap_data*'))

with open(ap_data_files[0],'rb') as f:
    ap_data = pickle.load(f)


classes = [c.classname for c in ap_data['mask'][0]]

# number = []

# for num,classlist in enumerate(classes):
#     number.append(num)
#
#  pretend I have a list of strings for classes
obj_dict = {'classes' : {'mask': [] , 'box' : []}}
for num,c in enumerate(classes):
    obj_dict['classes']['mask'].append({'name': c,'id': num, 'percision': [] ,'true_positives': []}) 
    obj_dict['classes']['box'].append({'name': c,'id': num, 'percision': [] ,'true_positives': []}) 
                                        

mask_dict = obj_dict['classes']['mask']
box_dict = obj_dict['classes']['box']
class_list = obj_dict['classes']

#epoch_number = ???? # np.linspace(0,number of files?)
# epoch_number = []

# for num, fname in enumerate(ap_data_files):
#     epoch_number.append(num)
    
for i in range(len(ap_data_files)):
    try:
        with open(ap_data_files[i],'rb') as f:
            ap_data = pickle.load(f)
            get_tp(ap_data)
    except Exception as e:
        print(ap_data_files[i])
        continue
        
        
    
    
#     with open(fname,'rb') as f:
#         apdata = pickle.load(f)
#         stuff = apdata['mask'][0][0]
#         classes = stuff.classnames
#         # how to get class names, but if we have classnames...
#         for c in classnames:
#             # if I knew how to get the ap dump for object type c
#             ap_dump_obj = []
#             obj_dict[c].append(ap_dump_obj.get_ap()) # not sure how to fit this in with recall bins
#             # what we really want is one number for each object type for each dump file. 
            

# plt.subplots(2,2,1)
# plt.plot(epoch_num, obj_dict[classname[23]])
# plt.ylim((0,1))

# plist=[]
# for i in range(len(classes)):
#     plot_num = (i % 4)+1
#     plt.subplot(2,2,plot_num)
    