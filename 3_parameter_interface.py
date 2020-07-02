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
numTrials = 100
nstep = 100
timelength = 100
trainFrac = .7

# Calls the module Signal with the initialization parameters
# then simulates using the initialized model
sig = Signal(numTrials,nstep,timelength,trainFrac)

# In this case, since we are only loading the model, not trying to train it,
# we can use function simulate and preprocess
xData,yData = sig.simulate_and_preprocess()

# Initialize the models that are saved using the parameters declared above
predictor = Model(nstep)
predictor.load_FOPTD()

kpPredictions = predictor.modelDict['kp'].predict(xData['kp'])
tauPredictions = predictor.modelDict['tau'].predict(xData['tau'])
thetaPredictions = predictor.modelDict['theta'].predict(xData['theta'])

uArrays = sig.uArray[sig.train,:]
yArrays = sig.yArray[sig.train,:]

for (i,index) in enumerate(sig.train):
    taup = tauPredictions[i]
    Kp = kpPredictions[i]
    theta = thetaPredictions[i]
    t = np.linspace(0,timelength,nstep)
    u = uArrays[i,:]
    yPred = (odeint(sig.FOmodel,0,t,args=(t,u,Kp,taup,theta),hmax=1.).ravel())
    yTrue = yArrays[i,:]
    
    plt.figure(dpi=100)
    plt.plot(t,u,label='Input Signal')
    plt.plot(t,yTrue, label='FOPTD Response')
    plt.plot(t,yPred,'--', label='Predicted Response')
    plt.xlabel("Time (s)")
    plt.ylabel("Change from set point")
    plt.legend()
    savePath = "/Users/RileyBallachay/Documents/Fifth Year/RNNSystemIdentification/Predictions/" + str(i) + ".png"
    
    #plt.savefig(savePath)