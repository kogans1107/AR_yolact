# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 13:15:52 2020

@author: Bill and Karrington
"""
import os
import matplotlib.pyplot as plt
import numpy as np

print(os.getcwd())

# fname = '/hdd/Shared/Kar_Things/git.workspace/AR_yolact/logs/yolact_base.log'

fname = 'logs/yolact_base.json'
with open(fname,'r') as f:
    data = f.read()

losses = []
i0 = 0
i1 = len(data)
while True:
    iT = data.find('T":', i0, i1)+3
    iend = data.find('}', iT, i1)-1
    try: 
        T = float(data[iT:iend])
    except ValueError:
        T = np.nan
    losses.append(T)
    i0 = iend

Slosses = []
i0 = 0
i1 = len(data)
while True:
    iS = data.find('S":', i0, i1)+3
    iend = data.find(',', iS, i1)-1
    try: 
        S = float(data[iT:iend])
    except ValueError:
        S = np.nan
    S = float(data[iS:iend])
    Slosses.append(S)
    i0 = iend