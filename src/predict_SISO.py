#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 15:07:39 2020

@author: RileyBallachay
"""
from Signal import Signal
from Model import Model
import scipy
import numpy as np
import time
import os

# These constants are also defined in the Signal module 
# Don't change here unless you also change them there
NUMTRIALS = 1000
batchSize = 16
plots = 5

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
    xData,yData = sig.system_validation_multi(disturbance=False,a_possible_values=[0.01,0.99],b_possible_values=[0.01,0.99],k_possible_values=[0,1],not_prbs=False)
    print("--- %s seconds ---" % (time.time() - start_time))
    
    xData=sig.PRBS()
    xData = sig.random_process()
    # Initialize the models that are saved using the parameters declared above
    predictor = Model()
    predictor.load_model(sig,path)
    
    #xData=np.sin(np.linspace(0,300))
    x=(xData) + 1
    huh = (scipy.stats.entropy(x))
    
    
    # Function to make predictions based off the simulation 
    kp_yhat = predictor.predict_multinomial(sig,stepResponse=False)
    #tau_yhat = self.modelDict['tau'](sig.xData['tau'])
    #theta_yhat = self.modelDict['theta'](sig.xData['theta'])
    
    print("--- %s seconds ---" % (time.time() - start_time))
    
    