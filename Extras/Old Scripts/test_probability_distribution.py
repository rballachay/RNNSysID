#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 14:50:26 2020

@author: RileyBallachay
"""
import numpy as np
import scipy
import matplotlib.pyplot as plt
from Signal import Signal
from Model import Model
import pandas as pd
from os import path
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

errorCSV = "/Users/RileyBallachay/Documents/Fifth Year/RNNSystemIdentification/Uncertainty/rawDataNew.csv"
# These constants are also defined in the Signal module 
# Don't change here unless you also change them there
numTrials = 1000
nstep = 100
timelength = 100
trainFrac = .7

if not(path.exists(errorCSV)):
    # Initialize the models that are saved using the parameters declared above
    predictor = Model(nstep)
    predictor.load_FOPTD()
        
    deviations = np.arange(0,5)
    
    stdev = np.array([0])
    error=np.array([0])
    kp_pred = np.array([0])
    theta_pred = np.array([0])
    tau_pred = np.array([0])
    
    kp_true = np.array([0])
    theta_true = np.array([0])
    tau_true = np.array([0])
    
    for deviation in deviations:
        # then simulates using the initialized model
        sig = Signal(numTrials,nstep,timelength,trainFrac)
        sig.training_simulation(KpRange=[0.5,10],tauRange=[0.5,10],thetaRange=[0.5,10])
        
        # In this case, since we are only loading the model, not trying to train it,
        # we can use function simulate and preprocess
        xData,yData = sig.simulate_and_preprocess(stdev=deviation)
    
        # Function to make predictions based off the simulation 
        predictor.predict(sig,savePredict=False,plotPredict=False)
    
        error = np.concatenate((predictor.errors,error))
        kp_pred = np.concatenate((predictor.kpPredictions[:,0],kp_pred))
        theta_pred = np.concatenate((predictor.thetaPredictions[:,0],theta_pred))
        tau_pred = np.concatenate((predictor.tauPredictions[:,0],tau_pred))
        
        kp_true = np.concatenate((sig.kps,kp_true))
        theta_true = np.concatenate((sig.thetas,theta_true))
        tau_true = np.concatenate((sig.taus,tau_true))
        stdev = np.concatenate((np.full_like(predictor.errors,deviation),stdev))
    
    sd = pd.DataFrame()
    sd['stdev'] = stdev
    sd['mse'] = error
    sd['kpPred'] = kp_pred
    sd['tauPred'] = tau_pred
    sd['thetaPred'] = theta_pred
    sd['kpTrue'] = kp_true
    sd['tauTrue'] = tau_true
    sd['thetaTrue'] = theta_true
    
    sd.to_csv(errorCSV, index=False)
    
else:
    try:
        sd = pd.read_csv(errorCSV).drop(['Unnamed: 0'],axis=1)
        sd.drop(sd.tail(1).index,inplace=True)
    except:
        sd = pd.read_csv(errorCSV)
        sd.drop(sd.tail(1).index,inplace=True)
    
    

prefixes = ['kp','tau','theta']
for prefix in prefixes:
    sd[prefix+'Error'] = (sd[prefix+'Pred']-sd[prefix+'True'])

    hist = np.histogram(sd[prefix+'Error'],bins=100)
    plt.figure(dpi=100)
    plt.hist(sd[prefix+'Error'],bins=100)
    
    h = np.std(sd[prefix+'Error'])
    print(h)
    
    plt.figure(dpi=200)
    plt.plot(sd[prefix+'True'],sd[prefix+'Pred'],'.')
    haha = np.linspace(1,10)+h
    plt.plot(np.linspace(1,10),np.linspace(1,10),'r--')
    plt.plot(np.linspace(1,10),np.linspace(1,10)+h,'g--')
    plt.plot(np.linspace(1,10),np.linspace(1,10)-h,'g--')


