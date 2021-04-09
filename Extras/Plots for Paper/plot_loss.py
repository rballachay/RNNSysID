#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 13:36:34 2021

@author: RileyBallachay
"""
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
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
sns.set()
for (idx,losstype) in enumerate(['MIMO 1x1']):
    direc = '/Users/RileyBallachay/Documents/Fifth Year/RNNSystemIdentification/Model Validation/'+losstype
    files = [f for f in os.listdir(direc) if f.endswith('.txt')]
    files.sort()
    names = ['a Loss','a Val Loss','b Loss', 'b Val Loss','k Loss','k Val Loss']
erplot based on magnitut    colors = ['darkred','indianred','midnightblue','steelblue','darkgreen','lightgreen']
    names2 = ['Training Loss','Validation Loss']
    for (it,file) in enumerate(files):
        plotData = np.zeros(500)
        data = np.loadtxt(direc+ '/'+file)
        ind = np.argmin(data)
        data[ind:] = min(data)
        plotData[0:500] = data[:500]
        plotData[len(data):] = data[-1]
        if 'Val' in names[it]:
            linestyle='solid'
        else:
            linestyle='dashed'
        plt.figure(1,dpi=300,figsize=(10,5))
        plt.plot(range(0,500),log(plotData[0:500]),colors[it],label=names2[it],linestyle=linestyle)
        if it==1:
            break
plt.legend()      
plt.ylabel("Loss (Negative Log Likelihood)")
plt.xlabel("Number of Epochs")