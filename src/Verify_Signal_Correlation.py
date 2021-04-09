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

# These constants are also defined in the Signal module 
# Don't change here unless you also change them there
numTrials = 1000
batchSize = 256
plots = 10

inDims = range(1,2)
outDims = range(1,2)


for (inDimension,outDimension) in zip(inDims,outDims):   
    start_time = time.time()
    
    sig = Signal(inDimension,outDimension,numTrials,numPlots=plots)
    
    uArray,yArray,tauArray,KpArray,thetaArray,train,test = sig.sys_simulation(stdev=10,disturbance=False,b_possible_values=[.49,.51],a_possible_values=[.49,.51],
                                                                              k_possible_values=[0,1])
    
    array=np.zeros(1000000)
    for (i,(u1,y1)) in enumerate(zip(uArray,yArray)):
        z1 = y1-u1
        for (j,(u2,y2)) in enumerate(zip(uArray,yArray)):
            z2 = y2-u2
            if i==j:
                array[i*100+j] = 1000
            else:
                array[i*100+j] = sum(abs(scipy.signal.correlate(z1,z2,mode='full')))
    
    index = np.argmin(array)
    plt.plot(yArray[int(index/1000)]-uArray[int(index/1000)])
    plt.plot(yArray[int(index%1000)]-uArray[int(index%1000)],'k--')

    print("--- %s seconds ---" % (time.time() - start_time))

    # These two lines are for training the model based on nstep and the sig data
    # Only uncomment if you want to train and not predict
    trainModel = Model()
    trainModel.load_and_train(sig,epochs=100,batchSize=batchSize,saveModel=False,plotLoss=bool(plots!=0),plotVal=bool(plots!=0))
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