#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 15:07:39 2020

@author: RileyBallachay
"""
from Signal import Signal
from Model import Model
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from numpy.polynomial.polynomial import polyfit
import matplotlib
import pandas as pd
import seaborn as sns
import numpy as np
import time
import os

# These constants are also defined in the Signal module 
# Don't change here unless you also change them there
NUMTRIALS = 10000
batchSize = 16
plots = 5

valPath = '/Users/RileyBallachay/Documents/Fifth Year/RNNSystemIdentification/Model Validation/'
model_paths = [f.path for f in os.scandir(valPath) if f.is_dir()]

inDims = range(1,2)
outDims = range(1,2)

for (inDimension,outDimension) in zip(inDims,outDims): 
    name ='MIMO ' + str(inDimension) + 'x' + str(outDimension)
    path = valPath + name + '/Checkpoints/'
    
    start_time = time.time()
    numTrials=int(NUMTRIALS/(inDimension*outDimension))
    sig = Signal(inDimension,outDimension,numTrials,numPlots=plots)

    # In this case, since we are only loading the model, not trying to train it,
    # we can use function simulate and preprocess 
    xData,yData = sig.system_validation_multi(disturbance=False,a_possible_values=[0.01,0.99],b_possible_values=[0.01,.99],k_possible_values=[0,1],not_prbs=False)
    print("--- %s seconds ---" % (time.time() - start_time))
    
    # Initialize the models that are saved using the parameters declared above
    predictor = Model()
    predictor.load_model(sig,path)

    # Function to make predictions based off the simulation 
    kp_yhat = predictor.predict_multinomial(sig,stepResponse=False)

    Values = abs(kp_yhat[0]['b'] - sig.yData['a'][:,1])
    Uncertainties = kp_yhat[1]['b']
    
    df = pd.DataFrame()
    df['L'] = sig.yData['b'][:,0]
    
    minima = min(df['L'])
    maxima = max(df['L'])
    norm = matplotlib.colors.Normalize(vmin=minima, vmax=maxima, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.Greys_r)
    colors = []
    for L in df['L']:
        colors.append(mapper.to_rgba(L)[0])
    df['C'] = colors
    df['V'] = Values
    df['U'] = Uncertainties
    #df = df[df['V'] <0.15]
    df = df[df['U'] <0.15]
    df = df[df['V'] >0.005]
    plt.figure(dpi=200,figsize=(10,5))
    plt.grid()
    sns.set()
    b, m = polyfit(np.log10(df['V']), np.log10(df['U']), 1)
    plt.scatter(np.log10(df['V']),np.log10(df['U']),c=df['C'])
    x = np.linspace(min(np.log10(df['V'])),max(np.log10(df['V'])))
    plt.plot(x, b + m * x, '-k')
    axes = plt.gca()
    #axes.set_xlim([0,.15])
    #axes.set_ylim([0,.15])
    plt.xlabel('Error in Paramter Estimate ($log_{10}$)')
    plt.ylabel('Parameter Uncertainty ($log_{10}$)')
    
    print("--- %s seconds ---" % (time.time() - start_time))
    
    