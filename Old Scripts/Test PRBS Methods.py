#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 10:36:21 2020

@author: RileyBallachay
"""
from scipy.signal import max_len_seq
import matplotlib.pyplot as plt
import numpy as np
import random

sig,_ = max_len_seq(12,length=1000)

plt.plot(sig[:100])

def PRBS(emptyArg,nstep=100,prob_switch=0.3, Range=[-1.0, 1.0]):  
    """Returns a pseudo-random binary sequence 
    which ranges between -1 and +1"""
    gbn = np.ones(nstep)
    gbn = gbn*random.choice([-1,1])
    probability = np.random.random(nstep)
    for i in range(0,(nstep-1)):
        prob = probability[i]
        gbn[i+1] = gbn[i]
        if prob < prob_switch:
            gbn[i+1] = -gbn[i+1]
    gbn=gbn.reshape((len(gbn),1))
    return gbn
    
rang=np.zeros(100)
uArray =list(map(PRBS,rang))