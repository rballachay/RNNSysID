#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 13:36:34 2021

@author: RileyBallachay
"""
import matplotlib.pyplot as plt
import numpy as np
import os
"""
LSTM_vLoss = '/Users/RileyBallachay/Documents/Fifth Year/RNNSystemIdentification/Models/2020-08-13 12:29/tau_val_loss.txt'
LSTM_loss = '/Users/RileyBallachay/Documents/Fifth Year/RNNSystemIdentification/Models/2020-08-13 12:29/tau_loss.txt'

GRU_vLoss = '/Users/RileyBallachay/Documents/Fifth Year/RNNSystemIdentification/Models/2020-08-13 17:12/tau_val_loss.txt'
GRU_loss = '/Users/RileyBallachay/Documents/Fifth Year/RNNSystemIdentification/Models/2020-08-13 17:12/tau_loss.txt'

LSTM_val_loss = np.loadtxt(LSTM_vLoss)
LSTM_loss = np.loadtxt(LSTM_loss)

GRU_val_loss = np.loadtxt(GRU_vLoss)
GRU_loss = np.loadtxt(GRU_loss)

plt.figure(dpi=200)
plt.plot(range(1,101),LSTM_loss,'darkred',label='LSTM Validation')
plt.plot(range(1,101),LSTM_val_loss,'indianred',linestyle='dashed',label='LSTM Training')
plt.plot(range(1,101),GRU_loss,'darkslategray',label='GRU Validation')
plt.plot(range(1,101),GRU_val_loss,'darkcyan',linestyle='dashed',label='GRU Training')
plt.ylabel("Loss (Negative Log Likelihood)")
plt.xlabel("Epochs")
plt.legend()

n = np.full((100000,2500,10),0,dtype=float)

print("%f bytes" % (n.size * n.itemsize*1e-9))
"""

for (idx,losstype) in enumerate(['MIMO 1x1']):
    direc = '/Users/RileyBallachay/Documents/Fifth Year/RNNSystemIdentification/Model Validation/'+losstype
    files = [f for f in os.listdir(direc) if f.endswith('.txt')]
    files.sort()
    names = ['Kp Loss','Kp Val Loss','τ Loss', 'τ Val Loss','θ Loss','θ Val Loss']
    colors = ['darkred','indianred','midnightblue','steelblue','darkgreen','lightgreen']
    for (it,file) in enumerate(files):
        plotData = np.zeros(250)
        data = np.loadtxt(direc+ '/'+file)
        ind = np.argmin(data)
        data[ind:] = min(data)
        plotData[0:len(data)] = data
        plotData[len(data):] = data[-1]
        if 'Val' in names[it]:
            linestyle='solid'
        else:
            linestyle='dashed'
        plt.figure(1)
        plt.plot(range(1,251),plotData,colors[it],label=names[it],linestyle=linestyle)

        
plt.ylabel("Loss (Negative Log Likelihood)")
plt.xlabel("Number of Epochs")