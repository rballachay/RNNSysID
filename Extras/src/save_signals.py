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
batchSize = 64
plots = 0

inDims = range(1,6)
outDims = range(1,6)

for (inDimension,outDimension) in zip(inDims,outDims):   
    start_time = time.time()
    
    sig = Signal(inDimension,outDimension,numTrials,numPlots=plots)
    
    uArray,yArray,tauArray,KpArray,thetaArray,train,test = sig.sys_simulation()
    
    uArray = uArray[:,:,0]
    yArray = yArray[:,:,0]
    
    if inDimension==1:
        np.savetxt('/Users/RileyBallachay/Documents/Fifth Year/RNNSystemIdentification/MATLAB code/u_1x1.csv',uArray, delimiter=',')
        np.savetxt('/Users/RileyBallachay/Documents/Fifth Year/RNNSystemIdentification/MATLAB code/y_1x1.csv',yArray, delimiter=',')
        np.savetxt('/Users/RileyBallachay/Documents/Fifth Year/RNNSystemIdentification/MATLAB code/tau_1x1.csv',tauArray, delimiter=',')
        np.savetxt('/Users/RileyBallachay/Documents/Fifth Year/RNNSystemIdentification/MATLAB code/kp_1x1.csv',KpArray, delimiter=',')
        np.savetxt('/Users/RileyBallachay/Documents/Fifth Year/RNNSystemIdentification/MATLAB code/theta_1x1.csv',thetaArray, delimiter=',')
        
    elif inDimension==2:
        np.savetxt('/Users/RileyBallachay/Documents/Fifth Year/RNNSystemIdentification/MATLAB code/u_2x2.csv',uArray)
        np.savetxt('/Users/RileyBallachay/Documents/Fifth Year/RNNSystemIdentification/MATLAB code/y_2x2.csv',yArray)