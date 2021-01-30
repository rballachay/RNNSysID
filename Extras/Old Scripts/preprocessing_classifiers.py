#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 10:54:27 2020

@author: RileyBallachay
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from Signal import Signal
from Model import Model
from scipy import signal
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import r2_score


class MyThresholdCallback(tf.keras.callbacks.Callback):
    def __init__(self, threshold):
        super(MyThresholdCallback, self).__init__()
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None): 
        val_acc = logs["val_acc"]
        if val_acc >= self.threshold:
            self.model.stop_training = True
       
        
# These constants are also defined in the Signal module 
# Don't change here unless you also change them there
numTrials = 10
nstep = 100
timelength = 100
trainFrac = .7
valFrac = 1 - trainFrac

# Calls the module Signal with the initialization parameters
# then simulates using the initialized model
sig1 = Signal(numTrials,nstep,timelength,trainFrac)
sig1.training_simulation(KpRange=[0.5,10],tauRange=[0.5,10],thetaRange=[0.5,10])

# Calls the module Signal with the initialization parameters
# then simulates using the initialized model
sig2 = Signal(numTrials,nstep,timelength,trainFrac)
sig2.training_simulation(KpRange=[-0.5,-10],tauRange=[0.5,10],thetaRange=[0.5,10])

index = range(0,len(sig2.yArray)*2)
train = random.sample(index,int(trainFrac*numTrials*2))
test = [item for item in list(index) if item not in train]

# In this case, since we are only loading the model, not trying to train it,
# we can use function simulate and preprocess
yArray1 = sig1.yArray
uArray1 = sig1.uArray
correlation1 = (yArray1-np.mean(yArray1))/np.std(yArray1) - uArray1
yData1 = np.ones((numTrials))

# In this case, since we are only loading the model, not trying to train it,
# we can use function simulate and preprocess
yArray2 = sig2.yArray
uArray2 = sig2.uArray
correlation2 = (yArray2-np.mean(yArray2))/np.std(yArray2) - uArray2
yData2 = np.zeros((numTrials))

xData = np.concatenate((correlation1,correlation2))
yData = np.concatenate((yData1,yData2))
numDim = xData.ndim - 1

trainspace = xData[train,:]
valspace = xData[test,:] 

x_train= trainspace.reshape((int(numTrials*trainFrac*2),nstep,1))
y_train = np.array([yData[i] for i in train])

x_val = valspace.reshape((int(numTrials*valFrac*2),nstep,1))
y_val = np.array([yData[i] for i in test])


model = keras.Sequential()

# I tried almost every permuation of LSTM architecture and couldn't get it to work
model.add(layers.GRU(100, activation='tanh',input_shape=(nstep,1)))
model.add(layers.Dense(100,activation='sigmoid'))
model.add(layers.Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error',metrics=['accuracy'])

model.summary()

my_callback = MyThresholdCallback(threshold=1.0000)

print("Fit model on training data")
history = model.fit(
    x_train,
    y_train,
    batch_size=16,
    epochs=200,
    # We pass some validation for
    # monitoring validation loss and metrics
    # at the end of each epoch
    validation_data=(x_val, y_val)
)

modelpath = "/Users/RileyBallachay/Documents/Fifth Year/RNNSystemIdentification/Models/Preprocessing Models/negative_gain.h5"
model.save(modelpath)
predictions = model.predict(x_val)

plt.figure(dpi=100)
plt.plot(y_val,predictions,'g.')
plt.plot(np.linspace(1,10),np.linspace(1,10),'r--')
plt.xlabel('Predicted Value of Theta')
print(r2_score(y_val,predictions))

