#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 18:27:52 2020

@author: RileyBallachay
"""

import matplotlib.pyplot as plt
import numpy as np

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