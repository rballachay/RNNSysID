#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 11:10:44 2021

@author: RileyBallachay
"""
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from Signal import Signal
from Model import Model
import time
import os
import numpy as np
import seaborn as sns
import scipy.stats as stats
import matplotlib.pylab as pylab

# These constants are also defined in the Signal module 
# Don't change here unless you also change them there
NUMTRIALS = 100000
batchSize = 32
plots = 5

valPath = '/Users/RileyBallachay/Documents/Fifth Year/RNNSystemIdentification/Model Validation/Closed Loop/'
model_paths = [f.path for f in os.scandir(valPath) if f.is_dir()]

inDims = range(1,2)
outDims = range(1,2)

for (inDimension,outDimension) in zip(inDims,outDims): 
    name ='MIMO ' + '1' + 'x' + '1'
    path = valPath + name + '/Checkpoints/'
    
    start_time = time.time()
    numTrials=int(NUMTRIALS/(inDimension*outDimension))
    sig = Signal(inDimension,outDimension,numTrials,numPlots=plots,stdev='variable')

    # In this case, since we are only loading the model, not trying to train it,
    # we can use function simulate and preprocess 
    xData,yData = sig.closed_loop_validation(b_possible_values=[.01,.99],a_possible_values=[.01,.99],
                                        k_possible_values=[1,10],order=False)
    print("--- %s seconds ---" % (time.time() - start_time))
    
    # Initialize the models that are saved using the parameters declared above
    predictor = Model()
    predictor.load_model(sig,path)
    
    # Function to make predictions based off the simulation 
    predDict,errDict = predictor.predict_system(sig,savePredict=True,stepResponse=False)