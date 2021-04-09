#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 17:18:50 2020

@author: RileyBallachay
"""
from Signal import Signal
from Model import Model
import time
import scipy
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np
import seaborn as sns
from scipy.ndimage import gaussian_filter1d

# These constants are also defined in the Signal module 
# Don't change here unless you also change them there
numTrials = 10
batchSize = 256
plots = 10

inDims = range(1,2)
outDims = range(1,2)
 
for (inDimension,outDimension) in zip(inDims,outDims):   
    start_time = time.time()
    
    sig = Signal(inDimension,outDimension,numTrials,numPlots=plots)
    
    uArray2,yArray,tauArray,KpArray,thetaArray,train,test = sig.sys_simulation(disturbance=False,b_possible_values=[.01,.99],a_possible_values=[.01,.99],
                                                                              k_possible_values=[0,1],not_prbs=True)

    uArray1,yArray,tauArray,KpArray,thetaArray,train,test = sig.sys_simulation(disturbance=False,b_possible_values=[.01,.99],a_possible_values=[.01,.99],
                    
                                                                              k_possible_values=[0,1],not_prbs=False)
    sns.set()
    plt.figure(figsize=(15,5),dpi=300)
    
    plt.subplot(1, 2, 1)
    plt.plot(uArray1[0,:,0],'firebrick')
    plt.plot(uArray2[0,:,0],'forestgreen')
    plt.xlabel('Time (s)')
    
    plt.subplot(1, 2, 2)
    plt.plot(gaussian_filter1d(abs(scipy.fft.fft(uArray1[0,:,0]))[:100],2),'firebrick',label='PRBS')
    plt.plot(gaussian_filter1d(abs(scipy.fft.fft(uArray2[0,:,0]))[:100],2),'forestgreen',label='Random Process')
    plt.xlabel('Frequency (Hz)')
    plt.legend()
    

    print("--- %s seconds ---" % (time.time() - start_time))

    # These two lines are for training the model based on nstep and the sig data
    # Only uncomment if you want to train and not predict
    trainModel = Model()
    trainModel.load_and_train(sig,epochs=100,batchSize=batchSize,saveModel=False,plotLoss=bool(plots!=0),plotVal=bool(plots!=0))
    print("--- %s seconds ---" % (time.time() - start_time))
