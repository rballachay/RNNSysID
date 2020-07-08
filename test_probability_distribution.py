#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 14:50:26 2020

@author: RileyBallachay
"""
import numpy as np
import matplotlib.pyplot as plt
from Signal import Signal
from Model import Model

# These constants are also defined in the Signal module 
# Don't change here unless you also change them there
numTrials = 1000
nstep = 100
timelength = 100
trainFrac = .7

deviations =np.arange(0,15)

error=np.array()
kp_pred = np.array()
theta_pred = np.array()
tau_pred = np.array()

kp_true = np.array()
theta_true = np.array()
tau_true = np.array()

# then simulates using the initialized model
sig = Signal(numTrials,nstep,timelength,trainFrac)
sig.training_simulation(KpRange=[0.5,10],tauRange=[0.5,10],thetaRange=[0.5,10])

# In this case, since we are only loading the model, not trying to train it,
# we can use function simulate and preprocess
xData,yData = sig.simulate_and_preprocess(stdev=5)

# Initialize the models that are saved using the parameters declared above
predictor = Model(nstep)
predictor.load_FOPTD()

# Function to make predictions based off the simulation 
predictor.predict(sig,savePredict=False,plotPredict=False)

error = predictor.errors
kp_pred = predictor.kpPredictions
theta_pred = predictor.thetaPredictions
tau_pred = predictor.tauPredictions

kp_true = sig.kps
theta_true = sig.thetas
tau_true = sig.taus

plt.figure()
plt.plot(kp_true,error,'.')