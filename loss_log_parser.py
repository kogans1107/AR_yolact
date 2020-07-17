# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 13:15:52 2020

@author: Bill and Karrington
"""

fname = 'C:\\Users\\peria\\Desktop\\work\\Brent Lab\\git-repo\\yolact\\logs\\yolact_base.log'
with open(fname,'r') as f:
    data = f.read()

losses = []
i0 = 0
i1 = len(data)
while True:
    iT = data.find('T":', i0, i1)+3
    iend = data.find('}', iT, i1)-1
    T = float(data[iT:iend])
    losses.append(T)
    i0 = iend

Slosses = []
i0 = 0
i1 = len(data)
while True:
    iS = data.find('S":', i0, i1)+3
    iend = data.find(',', iS, i1)-1
    S = float(data[iS:iend])
    Slosses.append(S)
    i0 = iend