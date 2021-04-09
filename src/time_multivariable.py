#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 15:07:39 2020

@author: RileyBallachay
"""
from Signal import Signal
from Model import Model
import time
import os
import numpy as np
import threading
import matplotlib.pyplot as plt

# These constants are also defined in the Signal module 
# Don't change here unless you also change them there
NUMTRIALS = 10
batchSize = 16
plots = 5

uData = np.loadtxt('/Users/RileyBallachay/Documents/Fifth Year/RNNSystemIdentification/MATLAB code/u_1x1.csv',delimiter=' ')
yData = np.loadtxt('/Users/RileyBallachay/Documents/Fifth Year/RNNSystemIdentification/MATLAB code/y_1x1.csv',delimiter=' ')

valPath = '/Users/RileyBallachay/Documents/Fifth Year/RNNSystemIdentification/Model Validation/'
model_paths = [f.path for f in os.scandir(valPath) if f.is_dir()]

inDims = range(1,2)
outDims = range(1,2)

for (inDimension,outDimension) in zip(inDims,outDims): 
    name ='MIMO ' + str(inDimension) + 'x' + str(outDimension)
    path = valPath + name + '/Checkpoints/'
    
    start_time = time.time()
    numTrials=int(NUMTRIALS/(inDimension*outDimension))
    sig = Signal(inDimension,outDimension,numTrials,numPlots=plots)

    # In this case, since we are only loading the model, not trying to train it,
    # we can use function simulate and preprocess 
    sig.system_validation_multi()
    print("--- %s seconds ---" % (time.time() - start_time))
    
    # Initialize the models that are saved using the parameters declared above
    predictor = Model()
    predictor.load_model(sig,path)
    #start_time = time.time()
    
    kps = np.zeros(1000)
    taus = np.zeros(1000)

    # Function to make predictions based off the simulation   
    a,b = yData.shape
    inlet = yData - uData
    inlet = inlet.reshape((a,b,1))
    
    start_time = time.time()
    data = predictor.modelDict['a'](inlet)
    kps = np.array(data.mean()[:,0])
    taus = np.array(data.mean()[:,1])
    
    kps_unc = np.array(data.stddev()[:,0])
    taus_unc = np.array(data.stddev()[:,1])
    
    w = predictor.modelDict['a'].get_weights()
    weights = []
    for li in w:
        weights = weights + list(li.flatten()[:])
    
    plt.figure(dpi=200)
    plt.hist(weights,bins=100,range=[-2,2])   
    
    print("--- %s seconds ---" % (time.time() - start_time))
    
    