#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 12:22:28 2020

@author: RileyBallachay
"""
import numpy as np
from pylfsr import LFSR
import matplotlib.pyplot as plt
from random import randrange, uniform

L = LFSR(fpoly=[9,5],initstate ='random',verbose=False)
L.info()
max_sample = 25
L.runKCycle(1000)
L.info()
seq = L.seq
print(seq)
G = np.zeros(max_sample*1000)
for i in range(0,1000):
    if seq[i]==0:
        seq[i]=-1
    G[25*i:25*i+25]=seq[i]

Y = np.zeros_like(G)

irand = randrange(0, 10)

a = np.linspace(0,2,100)
b=np.linspace(0,.1,100)

irand = randrange(0,100)

for i in range(1,1000*max_sample):
    Y[i]= .5*Y[i-1]+.1*G[i-1]
    

plt.plot(G[0:1000])
plt.plot(Y[0:1000])
