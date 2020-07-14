#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 17:18:50 2020

@author: RileyBallachay
"""
from Signal import Signal
from Model import Model
import time

# These constants are also defined in the Signal module 
# Don't change here unless you also change them there
numTrials = 50000
nstep = 100
timelength = 100
trainFrac = .7
  
start_time = time.time()
sig = Signal(numTrials,nstep,timelength,trainFrac)
sig.MIMO_simulation(stdev=5)
a = sig.kps
o = sig.uArray
print("--- %s seconds ---" % (time.time() - start_time))


# These two lines are for training the model based on nstep and the sig data
# Only uncomment if you want to train and not predict
trainModel = Model(nstep)
trainModel.train_MIMO(sig,epochs=100)
print("--- %s seconds ---" % (time.time() - start_time))
