#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 10:24:14 2020

@author: RileyBallachay
"""

import os
from os import path
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras import layers
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from Signal import Signal
from tensorflow.keras import backend as K
import datetime
import control as control
import random
import math
from multiprocessing import Pool

class MyThresholdCallback(tf.keras.callbacks.Callback):
    """
    Callback class to stop training model when a target
    coefficient of determination is achieved on validation set.
    """
    
    def __init__(self, threshold):
        super(MyThresholdCallback, self).__init__()
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None): 
        val_acc = logs["val_coeff_determination"]
        if val_acc >= self.threshold:
            self.model.stop_training = True


pool= Pool()

NUM_ITER = 10000
LEN_ARRAY = 100

b0 = np.linspace(0.1,10,100)
b1 = np.linspace(0.1,10,100)
x=np.linspace(0,100,100)

predictions = np.zeros((NUM_ITER,LEN_ARRAY))
coeffs = np.zeros((NUM_ITER,2))

def f(x,b0,b1):
    y = b0+b1*x
    return y

for ran in range(NUM_ITER):
    index1 = np.random.randint(0,99)
    index2 = np.random.randint(0,99)
    predictions[ran,:] = f(x,b0[index1],b1[index2])
    coeffs[ran,:] = [b0[index1],b1[index2]]

xData=predictions;yData=coeffs
# Randomly pick training and validation indices 
index = range(0,len(coeffs))
train = random.sample(index,int(0.7*NUM_ITER))
test = [item for item in list(index) if item not in train]

def preprocess(xData,yData,test,train):
    """This function uses the training and testing indices produced during
    simulate() to segregate the training and validation sets"""
    # If array has more than 2 dimensions, use 
    # axis=2 when reshaping, otherwise set to 1
    try:
        _,_,numDim= xData.shape
    except:
        numDim=1
       
    # Select training and validation data based on training
    # and testing indices set during simulation
    trainspace = xData[train]
    valspace = xData[test] 
    
    x_train= trainspace.reshape((math.floor(NUM_ITER*0.7),100,numDim))    
    x_val = valspace.reshape((math.floor(NUM_ITER*(1-0.7)),100,numDim))

    y_val = np.array([yData[i] for i in test])
    y_train = np.array([yData[i] for i in train])
        
    return x_train,x_val,y_train,y_val,numDim

MTC = MyThresholdCallback(0.95)

def coeff_determination(y_true, y_pred):
    "Coefficient of determination for callback"
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )  
    
 # Specify the surrogate posterior over `keras.layers.Dense` `kernel` and `bias`.
def posterior_mean_field(kernel_size, bias_size=0, dtype=None):
    n = kernel_size + bias_size
    c = 0.01*np.log(np.expm1(1.))
    return tf.keras.Sequential([
        tfp.layers.VariableLayer(2 * n, dtype=dtype,initializer='glorot_uniform'),
        tfp.layers.DistributionLambda(lambda t: tfd.Independent(
            tfd.Normal(loc=t[..., :n],scale=1e-5 + tf.nn.softplus(c + t[..., n:])),reinterpreted_batch_ndims=1)),
    ])

# Specify the prior over `keras.layers.Dense` `kernel` and `bias`.
def prior_trainable(kernel_size, bias_size=0, dtype=None):
    n = kernel_size + bias_size
    return tf.keras.Sequential([
        tfp.layers.VariableLayer(n, dtype=dtype,initializer='glorot_uniform'),
        tfp.layers.DistributionLambda(lambda t: tfd.Independent(
            tfd.Normal(loc=t, scale=1),reinterpreted_batch_ndims=1)),
    ])
  
def builder(x_train,y_train): 
        "Probabilistic model for SISO data"
        negloglik = lambda y, rv_y: -rv_y.log_prob(y[:])
        model = tf.keras.Sequential([
        tf.keras.layers.GRU(10, activation='tanh',input_shape=(100,1)),
        tfp.layers.DenseVariational(4,posterior_mean_field,prior_trainable,activation='linear',kl_weight=1/x_train.shape[0]),
        tfp.layers.DistributionLambda(lambda t: tfd.Normal(loc=[t[..., :2],t[..., :2]],
        scale=[1e-3 + tf.math.softplus(0.1 * t[...,2:]),1e-3 + tf.math.softplus(0.1 * t[...,2:])])),])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=.05), loss=negloglik,metrics=[coeff_determination])
        return model


x_train,x_val,y_train,y_val,numDim = preprocess(xData,yData,test,train)

model=builder(x_train,y_train)
print(model.summary())

history = model.fit(
                x_train,
                y_train,
                batch_size=16,
                epochs=5,
                # We pass some validation for
                # monitoring validation loss and metrics
                # at the end of each epoch
                validation_data=(x_val, y_val),
                callbacks=[MTC]
            )

yhat=model(x_val)
means=np.array(yhat.mean())
s = np.array(yhat.stddev())

plt.figure(dpi=200)
plt.plot(y_val[:,0],means[0,:,0],'b.')
plt.errorbar(y_val[:,0],means[0,:,0],yerr=s[0,:,0]*2,fmt='none',ecolor='green')

plt.figure(dpi=200)
plt.plot(y_val[:,1],means[0,:,1],'b.')
plt.errorbar(y_val[:,1],means[0,:,1],yerr=s[0,:,1]*2,fmt='none',ecolor='green')