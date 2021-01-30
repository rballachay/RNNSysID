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
import time
import control

start = time.time()

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
    
rang=np.zeros(1000)
uArray =np.array(list(map(PRBS,rang)))[...,-1]

def fun(iterator):
    sys = control.tf([np.random.randint(10),],[np.random.randint(10),1.])
    _,yEnd,_ = control.forced_response(sys,U=uArray[iterator,:],T=np.linspace(0,100,100))
    return yEnd

yArray = np.array(list(map(fun,range(1000))))
    
print(str(time.time()-start))