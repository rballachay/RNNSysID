#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 18:14:32 2020

@author: RileyBallachay
"""
import numpy as np
import matplotlib.pyplot as plt
Input = np.concatenate((np.zeros(10),np.ones(90)))
fftIn = np.fft.fft(Input)

def TF(s):
    return 100/(10*s+1)

Output = np.zeros_like(Input)
indices=[]

for i,_ in enumerate(Output):
    Output[i] = TF(i)*Input[i]
    indices.append(i)
   
Output = np.fft.fft(Output)
plt.plot(indices,Output)
plt.plot(indices,fftIn)    
