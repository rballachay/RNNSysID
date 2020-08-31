#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 16:00:08 2020

@author: RileyBallachay
"""
import numpy as np
import re

textsource = '/Users/RileyBallachay/Desktop/Tau3x3.txt'
textsource2 = '/Users/RileyBallachay/Desktop/kp3x3.txt'

def find_all_indexes(input_str, search_str):
    l1 = []
    length = len(input_str)
    index = 0
    while index < length:
        i = input_str.find(search_str, index)
        if i == -1:
            return l1
        l1.append(i)
        index = i + 1
    return l1

findwords = ["/step - loss: "," - val_loss: "]

with open(textsource, 'r') as file:
    data = file.read().replace('\n', '')
    
lossID = find_all_indexes(data,findwords[0])
valID = find_all_indexes(data,findwords[1])

loss = np.zeros(len(lossID))
val = np.zeros(len(lossID))
    
for (i,ID) in enumerate(lossID):
    ID = ID+len(findwords[0])
    loss[i] = data[ID:ID+7]
    VALID = valID[i]+len(findwords[1])
    val[i] = data[VALID:VALID+7]

name1='/Users/RileyBallachay/Documents/Fifth Year/RNNSystemIdentification/Model Validation/MIMO 3x3/tau_loss.txt'
np.savetxt(name1,loss)

name2='/Users/RileyBallachay/Documents/Fifth Year/RNNSystemIdentification/Model Validation/MIMO 3x3/tau_val_loss.txt'
np.savetxt(name2,val)

with open(textsource2, 'r') as file:
    data = file.read().replace('\n', '')
    
lossID = find_all_indexes(data,findwords[0])
valID = find_all_indexes(data,findwords[1])

loss = np.zeros(len(lossID))
val = np.zeros(len(lossID))
    
for (i,ID) in enumerate(lossID):
    ID = ID+len(findwords[0])
    loss[i] = data[ID:ID+7]
    VALID = valID[i]+len(findwords[1])
    val[i] = data[VALID:VALID+7]

name1='/Users/RileyBallachay/Documents/Fifth Year/RNNSystemIdentification/Model Validation/MIMO 3x3/kp_loss.txt'
np.savetxt(name1,loss)

name2='/Users/RileyBallachay/Documents/Fifth Year/RNNSystemIdentification/Model Validation/MIMO 3x3/kp_val_loss.txt'
np.savetxt(name2,val)