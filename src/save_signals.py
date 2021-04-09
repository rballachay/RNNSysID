#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 09:46:59 2020

@author: RileyBallachay
"""

from Signal import Signal
from Model import Model
import pandas as pd
import numpy as np
import time

# These constants are also defined in the Signal module 
# Don't change here unless you also change them there
numTrials = 1000
plots = 10

inDims = range(1,2)
outDims = range(1,2)

for (inDimension,outDimension) in zip(inDims,outDims):   
    start_time = time.time()
    
    sig = Signal(inDimension,outDimension,numTrials,numPlots=plots,stdev=5)
    
    uArray,yArray,tauArray,KpArray,thetaArray,train,test = sig.sys_simulation(b_possible_values=[.01,.99],a_possible_values=[.01,.99],
                                        k_possible_values=[0,1],order=False,disturbance=False)
    
    uArray = uArray[:,:,0]
    yArray = yArray[:,:,0]
    

    np.savetxt('/Users/RileyBallachay/Documents/Fifth Year/RNNSystemIdentification/MATLAB code/u_1x1.csv',uArray)
    np.savetxt('/Users/RileyBallachay/Documents/Fifth Year/RNNSystemIdentification/MATLAB code/y_1x1.csv',yArray)
    np.savetxt('/Users/RileyBallachay/Documents/Fifth Year/RNNSystemIdentification/MATLAB code/taus_1x1.csv',tauArray)
    np.savetxt('/Users/RileyBallachay/Documents/Fifth Year/RNNSystemIdentification/MATLAB code/Kps_1x1.csv',KpArray)
    np.savetxt('/Users/RileyBallachay/Documents/Fifth Year/RNNSystemIdentification/MATLAB code/thetas_1x1.csv',thetaArray)