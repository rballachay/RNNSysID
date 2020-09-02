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

# These constants are also defined in the Signal module 
# Don't change here unless you also change them there
NUMTRIALS = 100
batchSize = 16
plots = 5

uData = np.loadtxt('/Users/RileyBallachay/Documents/Fifth Year/RNNSystemIdentification/MATLAB code/u_1x1.csv',delimiter=',')
yData = np.loadtxt('/Users/RileyBallachay/Documents/Fifth Year/RNNSystemIdentification/MATLAB code/y_1x1.csv',delimiter=',')

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
    sig.system_validation(KpRange=[1,10],tauRange=[1,10],thetaRange=[1,10])
    print("--- %s seconds ---" % (time.time() - start_time))
    
    # Initialize the models that are saved using the parameters declared above
    predictor = Model()
    predictor.load_model(sig,path)
    start_time = time.time()
    
    kps = np.zeros(1000)
    taus = np.zeros(1000)
    thetas = np.zeros(1000)
    
    # Function to make predictions based off the simulation 
    for i,(u,y) in enumerate(zip(uData,yData)):
        in1 =  (u*y).reshape(1,100,1)
        in2 = (y-u).reshape(1,100,1)
        
        kps[i] = predictor.modelDict['kp'].predict(in1)[0]
        taus[i] = predictor.modelDict['tau'].predict(in2)[0]
        thetas[i] = predictor.modelDict['theta'].predict(in2)[0]
    
    print("--- %s seconds ---" % (time.time() - start_time))
    
    