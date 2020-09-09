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

import matplotlib.backends.backend_pdf as pdf

from eval import APDataObject

from sklearn.metrics import auc

def uichoosedir(title = None, initialdir = None):
    root = tk.Tk()
    root.focus_force()
    root.withdraw() # we don't want a full GUI, so keep the root window 
                    #  from appearing
    pathname = tk.filedialog.askdirectory(title=title, initialdir = initialdir)
    return pathname

# def get_all_auc(get_auc):
#     precision = []
#     recall_bins = np.linspace(0.0,0.99,10)
#     for i in range(len(classes)):
#          precision.append(auc(recall_bins,[get_auc[j].get_ap() for j in range(10)]))  
#     return  precision
     
def get_auc(data):
    recall_bins = np.linspace(0.0,0.99,10)
    precision = auc(recall_bins,[data[i].get_ap() for i in range(10)])
    return precision
        

def get_tp(df):
    # mask_auc = get_all_auc(np.transpose(df['mask'],(1,0)))
    # box_auc = get_all_auc(np.transpose(df['box'],(1,0)))

    for i in range(len(classes)):
        mask_list = []
        box_list = []
        mask_dict[i]['true_positives'].append(df['mask'][0][i].num_gt_positives)
        box_dict[i]['true_positives'].append(df['box'][0][i].num_gt_positives)
        for j in range(10):
            mask_list.append(df['mask'][j][i])
            box_list.append(df['box'][j][i])
        mask_dict[i]['percision'].append(get_auc(mask_list))
        box_dict[i]['percision'].append(get_auc(box_list))
        # box_dict[i]['percision'].append(get_auc(df['box'][:][i]))
        
        
        
        # mask_dict[j]['percision'].append(mask_auc[i])
        # box_dict[j]['percision'].append(box_auc[i]
        
        # # print(len(mask_auc))
        # # print(len(box_auc))
        # for j in range(len(classes)):
             


if __name__ == "__main__":
    
    
    dir_to_process = uichoosedir()
    print('Loading files')
    ap_data_files = sorted(glob.glob(dir_to_process + '/'+ 'ap_data*'))
    print('loaded ', len(ap_data_files), 'files')
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
    
    pdf_files = pdf.PdfPages('results/data_analysis/output.pdf')
    

    
    # fig,axs = plt.subplots(2,2)
    # axs[0,0].set_title('Mask Precision')
    # axs[0,1].set_title('Box Precision')
    # axs[1,0].set_title('Mask True Positivies')
    # axs[1,1].set_title('Box True Positivies')
    
    # axs[0,0].set_ylabel('Precision Value')
    # axs[0,1].set_ylabel('Precision Value')
    # axs[1,0].set_ylabel('Number of GT Positives')
    # axs[1,1].set_ylabel('Number of GT Positives')
    
    # axs[0,0].set_xlabel('pkl file number')
    # axs[0,1].set_xlabel('pkl file number')
    # axs[1,0].set_xlabel('pkl file number')
    # axs[1,1].set_xlabel('pkl file number')
        
    
    
    for i in range(len(classes)):
        fig,axs = plt.subplots(2,2)
        fig.suptitle(mask_dict[i]['name'],y=0.05)
        
        axs[0,0].set_title('Mask Precision')
        axs[0,1].set_title('Box Precision')
        axs[1,0].set_title('Mask True Positivies')
        axs[1,1].set_title('Box True Positivies')
        
        axs[0,0].set_ylabel('Precision Value')
        axs[0,1].set_ylabel('Precision Value')
        axs[1,0].set_ylabel('Number of GT Positives')
        axs[1,1].set_ylabel('Number of GT Positives')
        
        axs[0,0].set_xlabel('pkl file number')
        axs[0,1].set_xlabel('pkl file number')
        axs[1,0].set_xlabel('pkl file number')
        axs[1,1].set_xlabel('pkl file number')
        
    
        axs[0][0].plot(mask_dict[i]['percision'])
        axs[0][1].plot(box_dict[i]['percision'])
        axs[1][0].plot(mask_dict[i]['true_positives'])
        axs[1][1].plot(box_dict[i]['true_positives'])
        
        fig.tight_layout()
        pdf_files.savefig(fig)
        plt.close()
        
        
    # fig_all,axs_all = plt.subplots(2,2)
    
    fig.suptitle('All Classes')
    # axs_all[0,0].set_title('Mask Precision')
    # axs_all[0,1].set_title('Box Precision')
    # axs_all[1,0].set_title('Mask True Positivies')
    # axs_all[1,1].set_title('Box True Positivies')
    
    # axs_all[0,0].set_ylabel('Precision Value')
    # axs_all[0,1].set_ylabel('Precision Value')
    # axs_all[1,0].set_ylabel('Number of GT Positives')
    # axs_all[1,1].set_ylabel('Number of GT Positives')
    
    # axs_all[0,0].set_xlabel('pkl file number')
    # axs_all[0,1].set_xlabel('pkl file number')
    # axs_all[1,0].set_xlabel('pkl file number')
    # axs_all[1,1].set_xlabel('pkl file number')
    
    for i in range(len(classes)):
        axs[0][0].plot(mask_dict[i]['percision'])
        axs[0][1].plot(box_dict[i]['percision'])
        axs[1][0].plot(mask_dict[i]['true_positives'])
        axs[1][1].plot(box_dict[i]['true_positives'])
        fig.tight_layout()
    
    pdf_files.savefig(fig)
    pdf_files.close()
    # plist=[]
    # for i in range(len(classes)):
    #     plot_num = (i % 4)+1
    #     plt.subplot(2,2,plot_num)
        