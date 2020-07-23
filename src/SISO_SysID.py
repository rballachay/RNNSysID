#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 14:50:26 2020

@author: RileyBallachay
"""
from Signal import Signal
from Model import Model
import numpy as np

# These constants are also defined in the Signal module 
# Don't change here unless you also change them there
numTrials = 100000
nstep = 100
timelength = 100
trainFrac = .7


usplit = np.stack(np.split(np.array([1,2,3,4,5,6]),2),axis=0)

# Calls the module Signal with the initialization parameters
# then simulates using the initialized model
sig = Signal(numTrials,nstep,timelength,trainFrac,stdev=5)
sig.SISO_simulation(KpRange=[0.75,10.5],tauRange=[0.75,10.5])


# These two lines are for training the model based on nstep and the sig data
# Only uncomment if you want to train and not predict
trainModel = Model(nstep)

trainModel.train_SISO(sig,epochs=100,probabilistic=False)

"""
# In this case, since we are only loading the model, not trying to train it,
# we can use function simulate and preprocess
xData,yData = sig.SISO_validation()

# Initialize the models that are saved using the parameters declared above
predictor = Model(nstep,Modeltype='regular')
predictor.load_SISO()

# Function to make predictions based off the simulation 
i = predictor.predict_SISO(sig,savePredict=True)
"""