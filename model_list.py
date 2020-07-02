#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 17:32:17 2020

@author: RileyBallachay
"""
import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from scipy.fft import fft
from sklearn.preprocessing import StandardScaler
from scipy.stats import kurtosis,skew,entropy,variation,gmean
from sklearn.metrics import r2_score

class model:    
    
    def __init__(self,nstep):
        self.nstep=nstep
        return
    
    def model_1(self):
        model = keras.Sequential()
        
        # I tried almost every permuation of LSTM architecture and couldn't get it to work
        model.add(layers.GRU(100, activation='tanh',input_shape=(self.nstep,1)))
        model.add(layers.Dense(100, activation='linear',))
        model.add(layers.Dense(1, activation='linear'))
        
        # Compile the model
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model   
    
   

    
    def tau_model(self):
        model = keras.Sequential()
        
        # I tried almost every permuation of LSTM architecture and couldn't get it to work
        model.add(layers.GRU(100, activation='tanh',input_shape=(self.nstep,1)))
        model.add(layers.Dense(100,activation='linear'))
        model.add(layers.Dense(1, activation='linear'))
        
        # Compile the model
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        #batch size of 16 and at least 200 epochs
        return model