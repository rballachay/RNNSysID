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
numTrials = 100
batchSize = 256
plots = 10

inDims = range(1,2)
outDims = range(1,2)


def lag_finder(y1, y2, sr,title):
    delays = range(1,10)
    correlations = np.zeros_like(delays)
    backup_correlation = np.zeros_like(delays)

    for d in delays:
        y2Temp = y2[d:,0]/np.max(abs(y2[d:,0]))
        delay = np.correlate(y1[:-d,0],y2Temp)
        plt.figure(d*np.random.randint(1))
        plt.title('Lag: ' + str(d) + ' s  ' + 'Real =' + str(title))
        plt.plot(y1[:,0])
        plt.plot(y2Temp)
        su1 = np.cumsum(y1[:,0])
        plt.plot(su1/np.max(abs(su1)))
        correlations[d-1]=delay
        peaks, _ = signal.find_peaks(y2Temp)
        print(str(peaks)+'\n')
        peaks, _ = signal.find_peaks(su1)
        print(str(peaks)+'\n')
        plt.show()  
    
    af = scipy.fft(y1[:,0])
    bf = scipy.fft(y2[:,0])
    plt.plot(af)
    plt.plot(bf)
    
    c = scipy.ifft(af * scipy.conj(bf))
    
    delay = np.argmax(abs(c))

    #delay = np.argmax(correlations) 
    plt.figure()
    #plt.plot(delay_arr, corr)
    plt.title('Lag: ' + str(np.round(delay, 3)) + ' s  ' + 'Real =' + str(title))
    plt.xlabel('Lag')
    plt.ylabel('Correlation coeff')
    plt.show()



for (inDimension,outDimension) in zip(inDims,outDims):   
    start_time = time.time()
    
    sig = Signal(inDimension,outDimension,numTrials,numPlots=plots)
    
    uArray,yArray,tauArray,KpArray,thetaArray,train,test = sig.sys_simulation(disturbance=False,b_possible_values=[.01,.99],a_possible_values=[.01,.99],
                                                                              k_possible_values=[0,1],not_prbs=True)
    
    
    '''
    array=np.zeros(10000)
    i=0
    for (i,(u1,y1)) in enumerate(zip(uArray,yArray)):
        z1 = y1-u1
        for (j,(u2,y2)) in enumerate(zip(uArray,yArray)):
            z2 = y2-u2
            array[i*100+j] = sum(scipy.signal.correlate(z1,z2,mode='full'))
            
    '''
    print("--- %s seconds ---" % (time.time() - start_time))

    # These two lines are for training the model based on nstep and the sig data
    # Only uncomment if you want to train and not predict
    trainModel = Model()
    trainModel.load_and_train(sig,epochs=1000,batchSize=batchSize,saveModel=False,plotLoss=bool(plots!=0),plotVal=bool(plots!=0))
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