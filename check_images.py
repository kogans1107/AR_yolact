# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 14:09:41 2020
@author: Bill
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 14:04:23 2020
@author: Bill
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 05:22:29 2020
This takes a COCO annotation JSON file and displays the images and their 
segmentation and labels, just to double check that everything is ok with them, prior 
to training. 
@author: Bill and Karrington
"""
#
#  The following import gets contents of your config.py file into D. 
import data as D
  
from utils.augmentations import SSDAugmentation #, FastBaseTransform, BaseTransform

import os
import time
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, simpledialog
import numpy as np
import platform as platf
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib
import torch
import traceback

class_list = []
label_map = {}
class_label = 'CLASSES:'
map_annotation =  'LABEL MAP:'
num = []
inst_dict = {}


def missing_images(js): #this function has to be in the images directory and return the name of missing image files
    file_names = []
    file_DNE = []
    
    for jsi in js['images']:
        file_names.append(jsi['file_name'])
        
    file_exist = [f for f in file_names if os.path.isfile(f)]
    file_DNE = list(set(file_exist) ^ set(file_names))
    
    print ('Missing Files: ', file_DNE)
    
def count_instances(id, js): #this function counts all of the instances
    count = 0
    for jsa in js['annotations']:
        if jsa['category_id'] == id:
            count += 1
    return(count)


def instance_collection(js): #this function makes a diction of all the instances in the json file of all the classes
    num = []
    inst_dict = {}
    for jsc in js['categories']:
        inst = count_instances(jsc['id'], js)
        num.append(inst)
        dic_name = jsc['name']
        inst_dict.update({dic_name:inst})
    return(inst_dict)

def change_cat_id(old_id, new_id, js): #this function updates the categorie id numbers
    for jsa in js['annotations']:
        if jsa['category_id'] == old_id:
            jsa['category_id'] = new_id
    for jsc in js['categories']:
        if jsc['id'] == old_id:
            jsc['id'] = new_id

def change_name(old_name, new_name, js): # this function changes the desired name of files
    for jsc in js['categories']:
        if jsc['name'] == old_name:
            jsc['name'] = new_name
        

class ImageChecker(tk.Frame):
    def __init__(self):
        from pycocotools.coco import COCO

        # Hocus pocus...bring Tk to life...
        self.root = tk.Tk() 
        top = tk.Toplevel(self.root)
        top.withdraw()  # ...in secret....

        infile = \
            filedialog.askopenfilename(parent=top, \
                                        title='Choose JSON file')

#        infile = 'C:/Users/peria/Desktop/work/Brent Lab/git-repo/yolact/' + \
#        'data/coco/annotations/milliCOCO.json'
#        
        self.dataset = D.COCODetection(image_path=D.cfg.dataset.train_images,
                            info_file=infile,
                            transform=SSDAugmentation(D.MEANS))

        self.coco = COCO(infile)

#        print('Type of COC(infile) is',type(self.coco))

        matplotlib.use('Qt5Agg')

        try:
            self.label_map = D.KAR_LABEL_MAP
            self.classes = D.KAR_CLASSES
        except AttributeError:
            self.label_map = D.COCO_LABEL_MAP
            self.classes = D.COCO_CLASSES


        self.images_dir = '/'.join(infile.split('/')[0:-2]) + '/images/'

        n_img = len(self.dataset.ids)
        
        self.ids = list(self.coco.imgToAnns.keys())
        self.random_image_iter = iter(np.argsort(np.random.uniform(size=(n_img))))

        pad=3 # Why? 
        geom=("{0}x{1}+0+0".format(
            (self.root.winfo_screenwidth()-pad)//2, \
                (self.root.winfo_screenheight()-pad)*7//8))
        self.root.geometry(geom)
        
        tk.Frame.__init__(self, self.root)
        
        
        self.create_widgets()

    def create_widgets(self):   
        self.fig = Figure(figsize=(8,8), dpi=100)
        self.ax = self.fig.add_subplot(111)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)  # A tk.DrawingArea.

        self.enough = tk.Button(self, text="Enough, already!")
        self.enough.bind('<Button-1>', self.byebye)
 
        self.next = tk.Button(self, text="Show Next Image")
        self.next.bind('<Button-1>', self.show_next_image)
    
        self.grid()
        self.canvas.get_tk_widget().grid(row=1,columnspan=3, rowspan=18)
        self.enough.grid(row=20,column=0)
        self.next.grid(row=20,column=2)
        
        self.show_next_image(None)


#-----------------------------


    def show_next_image(self, event):

        while True:
            try:   
                i_img = next(self.random_image_iter)
                file_name = self.coco.loadImgs(self.ids[i_img])[0]['file_name']

                try:
                    (image, target, masks, height, width, crowd) = \
                    self.dataset.pull_item(i_img)
                    
                    img = image.cpu().detach().numpy().transpose((1,2,0))
                    img = (img - np.min(img))/(np.max(img) - np.min(img))
                    
                    self.ax.clear()
                    self.ax.imshow(img)

                    anno = self.dataset.pull_anno(i_img)
                    for i,a in enumerate(anno):
                        mask = masks[i]
                        igtz = np.where(mask > 0)
                        try:
                            x0, y0, x1, y1 = \
                            np.min(igtz[1]), np.min(igtz[0]), \
                            np.max(igtz[1]), np.max(igtz[0])
                            w = x1 - x0
                            h = y1 - y0
                            
                            # Seems crazy, but I had to subtract 1 from the label_map value...
                            labeltext = self.classes[self.label_map[a['category_id']]-1]
                            
                            self.ax.plot([x0, x0+w, x0+w, x0,   x0],\
                                         [y0, y0,   y0+h, y0+h, y0])
                            self.ax.set_title(file_name)
                            self.ax.text(x0, y0, labeltext, \
                             horizontalalignment='left', verticalalignment='bottom',\
                             color='black', bbox=dict(facecolor='yellow', alpha=0.5))
                        except Exception as exc:
                            print(traceback.format_exc())
                            print(exc)
                            
                        
                    self.canvas.draw()
                    break

                except (FileNotFoundError, AssertionError):
                    print(file_name)
                else:
                    print('new error!')
                    
            except StopIteration:
                print('No more files! You have seen them all...')
                self.byebye(None)
                break
    

    def byebye(self, event):
        self.root.destroy()
    
    def start(self):
        self.root.mainloop()
 
    def get_slash(self): # OMG I hate computers sometimes. 
        if platf.system() == 'Windows':
            slash = '\\' 
        else:
            slash = '/'
        return slash
    

if __name__ == "__main__":
    
    ic = ImageChecker().start()
        
