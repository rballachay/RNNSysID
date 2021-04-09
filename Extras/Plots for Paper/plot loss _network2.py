#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 18:27:52 2020

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
fig, axes = plt.subplots(1, 1,figsize=(16,5),dpi=400) 

for (idx,losstype) in enumerate(['MIMO 1x1','MIMO 2x2','MIMO 3x3']):
    ax=axes[idx]
    direc = '/Users/RileyBallachay/Documents/Fifth Year/RNNSystemIdentification/Model Validation/'+losstype
    files = [f for f in os.listdir(direc) if f.endswith('.txt')]
    files.sort()
    names = ['Kp Loss','Kp Val Loss','τ Loss', 'τ Val Loss','θ Loss','θ Val Loss']
    colors = ['darkred','indianred','midnightblue','steelblue','darkgreen','lightgreen']
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
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
        
        ax.plot(range(1,251),plotData,colors[it],label=names[it],linestyle=linestyle)
        
    #ax.set_ylim([0,3])
    ax.legend(title=losstype)
   
    for ax in axes.flat:
        ax.label_outer()
        
plt.ylabel("Loss (Negative Log Likelihood)")
plt.xlabel("Number of Epochs")
