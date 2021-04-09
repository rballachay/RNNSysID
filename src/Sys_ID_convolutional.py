#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 10:10:50 2021

@author: RileyBallachay
"""
from Signal import Signal
from Model import Model
import time

# These constants are also defined in the Signal module 
# Don't change here unless you also change them there
numTrials = 100000
batchSize = 100
plots = 5

inDims = range(1,10,2)
outDims = range(1,10,2)

for (inDimension,outDimension) in zip(inDims,outDims):   
    start_time = time.time()
    
    sig = Signal(inDimension,outDimension,numTrials,numPlots=plots)
    
    uArray,yArray,tauArray,KpArray,thetaArray,train,test = sig.sys_simulation(disturbance=False,k_possible_values=[0,1])
    print("--- %s seconds ---" % (time.time() - start_time))

    # These two lines are for training the model based on nstep and the sig data
    # Only uncomment if you want to train and not predict
    trainModel = Model()
    trainModel.train_model(sig,epochs=1000,batchSize=batchSize,saveModel=False,plotLoss=bool(plots!=0),plotVal=bool(plots!=0),trainConvolutional=False)
    print("--- %s seconds ---" % (time.time() - start_time))
   
'''
# In this case, since we are only loading the model, not trying to train it,
# we can use function simulate and preprocess 
xData,yData = sig.MIMO_validation()

# Initialize the models that are saved using the parameters declared above
predictor = Model(nstep)
predictor.load_MIMO() 

# Function to make predictions based off the simulation 
i = predictor.predict_MIMO(sig,savePredict=True)

print("--- %s seconds ---" % (time.time() - start_time))
'''