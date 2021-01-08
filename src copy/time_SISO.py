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

# These constants are also defined in the Signal module 
# Don't change here unless you also change them there
NUMTRIALS = 1000
batchSize = 16
plots = 5

uData = np.loadtxt('/Users/RileyBallachay/Documents/Fifth Year/RNNSystemIdentification/MATLAB code/u_1x1_order2.csv',delimiter=' ')
yData = np.loadtxt('/Users/RileyBallachay/Documents/Fifth Year/RNNSystemIdentification/MATLAB code/y_1x1_order2.csv',delimiter=' ')

valPath = '/Users/RileyBallachay/Documents/Fifth Year/RNNSystemIdentification/Model Validation/'
model_paths = [f.path for f in os.scandir(valPath) if f.is_dir()]

inDims = range(1,2)
outDims = range(1,2)

def threaded_function(inlet):  
    kps = predictor.modelDict['kp'].predict(inlet)[0]
    taus = predictor.modelDict['tau'].predict(inlet)[0]
    return kps,taus
        




for (inDimension,outDimension) in zip(inDims,outDims): 
    name ='MIMO ' + str(inDimension) + 'x' + str(outDimension)
    path = valPath + name + '/Checkpoints/'
    
    start_time = time.time()
    numTrials=int(NUMTRIALS/(inDimension*outDimension))
    sig = Signal(inDimension,outDimension,numTrials,numPlots=plots)

    # In this case, since we are only loading the model, not trying to train it,
    # we can use function simulate and preprocess 
    sig.system_validation(KpRange=[1,10],tauRange=[1,10],thetaRange=[1,10])
    print("--- %s seconds ---" % (time.time() - start_time))
    
    # Initialize the models that are saved using the parameters declared above
    predictor = Model()
    predictor.load_model(sig,path)
    #start_time = time.time()
    
    kps = np.zeros(1000)
    taus = np.zeros(1000)
    thetas = np.zeros(1000)

    # Function to make predictions based off the simulation   
    a,b = yData.shape
    inlet = yData * uData
    inlet = inlet.reshape((a,b,1))
    
    start_time = time.time()
    kps = np.array(predictor.modelDict['kp'](inlet).mean()[:])
    taus = np.array(predictor.modelDict['tau'](inlet).mean()[:])
    thetas = np.array(predictor.modelDict['theta'](inlet).mean()[:])
    
    kps_unc = np.array(2*predictor.modelDict['kp'](inlet).stddev()[:])
    taus_unc = np.array(2*predictor.modelDict['kp'](inlet).stddev()[:])
    thetas_unc = np.array(2*predictor.modelDict['theta'](inlet).stddev()[:])
    
    print("--- %s seconds ---" % (time.time() - start_time))
    
    