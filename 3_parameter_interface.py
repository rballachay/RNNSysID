#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 14:50:26 2020

@author: RileyBallachay
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from Signal import Signal
from Model import Model

# These constants are also defined in the Signal module 
# Don't change here unless you also change them there
numTrials = 1000
nstep = 100
timelength = 100
trainFrac = .7


# Calls the module Signal with the initialization parameters
# then simulates using the initialized model
sig = Signal(numTrials,nstep,timelength,trainFrac)
sig.training_simulation(KpRange=[0.5,10],tauRange=[0.5,10],thetaRange=[0.5,10])

# These two lines are for training the model based on nstep and the sig data
# Only uncomment if you want to train and not predict
trainModel = Model(nstep)
trainModel.train_FOPTD(sig,epochs=50)

# In this case, since we are only loading the model, not trying to train it,
# we can use function simulate and preprocess
xData,yData = sig.simulate_and_preprocess()

# Initialize the models that are saved using the parameters declared above
predictor = Model(nstep)
predictor.load_FOPTD()

# Function to make predictions based off the simulation 
i = predictor.predict(sig,savePredict=True)
