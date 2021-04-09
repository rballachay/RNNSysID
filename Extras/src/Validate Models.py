 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 10:16:14 2020

@author: RileyBallachay
"""

from Signal import Signal
from Model import Model
import time
import os

# These constants are also defined in the Signal module 
# Don't change here unless you also change them there
NUMTRIALS = 100
batchSize = 16
plots = 5

valPath = '/Users/RileyBallachay/Documents/Fifth Year/RNNSystemIdentification/Model Validation/'
model_paths = [f.path for f in os.scandir(valPath) if f.is_dir()]

inDims = range(3,6)
outDims = range(3,6)

for (inDimension,outDimension) in zip(inDims,outDims): 
    name ='MIMO ' + str(inDimension) + 'x' + str(outDimension)
    path = valPath + name + '/Checkpoints/'
    
    start_time = time.time()
    numTrials=int(NUMTRIALS/(inDimension*outDimension))
    sig = Signal(inDimension,outDimension,numTrials,numPlots=plots,stdev=5)

    # In this case, since we are only loading the model, not trying to train it,
    # we can use function simulate and preprocess 
    xData,yData = sig.system_validation(KpRange=[1,10],tauRange=[1,10],thetaRange=[1,10])
    print("--- %s seconds ---" % (time.time() - start_time))
    
    # Initialize the models that are saved using the parameters declared above
    predictor = Model()
    predictor.load_model(sig,path)
    
    # Function to make predictions based off the simulation 
    predictor.predict_system(sig,savePredict=True)
    
    print("--- %s seconds ---" % (time.time() - start_time))
    
    
    print("--- %s seconds ---" % (time.time() - start_time))