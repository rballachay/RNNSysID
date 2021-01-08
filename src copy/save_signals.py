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
    
    sig = Signal(inDimension,outDimension,numTrials,numPlots=plots)
    
    uArray,yArray,tauArray,KpArray,thetaArray,train,test = sig.sys_simulation(KpRange=[.5,.99],tauRange=[.5,.99],thetaRange=[1,10],order=2)
    
    uArray = uArray[:,:,0]
    yArray = yArray[:,:,0]
    

    np.savetxt('/Users/RileyBallachay/Documents/Fifth Year/RNNSystemIdentification/MATLAB code/u_1x1_order2_oscillatory.csv',uArray)
    np.savetxt('/Users/RileyBallachay/Documents/Fifth Year/RNNSystemIdentification/MATLAB code/y_1x1_order2_oscillatory.csv',yArray)
    np.savetxt('/Users/RileyBallachay/Documents/Fifth Year/RNNSystemIdentification/MATLAB code/taus_1x1_order2_oscillatory.csv',tauArray)
    np.savetxt('/Users/RileyBallachay/Documents/Fifth Year/RNNSystemIdentification/MATLAB code/Kps_1x1_order2_oscillatory.csv',KpArray)
    np.savetxt('/Users/RileyBallachay/Documents/Fifth Year/RNNSystemIdentification/MATLAB code/thetas_1x1_order2_oscillatory.csv',thetaArray)