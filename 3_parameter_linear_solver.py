#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 14:50:26 2020

@author: RileyBallachay
"""
import os
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.integrate import odeint,ode
import scipy.signal as signal
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from scipy.fft import fft
from sklearn.preprocessing import StandardScaler
from scipy.stats import kurtosis,skew,entropy,variation,gmean
from sklearn.metrics import r2_score
from mpl_toolkits.mplot3d import Axes3D
from Signal import Signal
from model_list import model

# These constants are also defined in the Signal module 
# Don't change here unless you also change them there
numTrials = 1000
nstep = 100
timelength = 100
trainFrac = .7

# Calls the module Signal with the initialization parameters
# then simulates using the initialized model
sig = Signal(numTrials,nstep,timelength,trainFrac)
uArray,yArray,corrArray,conArray,taus,kps,thetas,train,test = sig.simulate()

# This may need to be implemented iteratively to determine if stacking
# the three different parameters has any significant impact on predicting the
# value of any of the 3 parameters
xDatas = [yArray,yArray,(yArray-np.mean(yArray))/np.std(yArray) - uArray]
yDatas = [taus, kps, thetas]

m = model(nstep)
modelList = []

uSum = np.sum(uArray,axis=0)
plt.figure()
plt.plot(np.linspace(0,100,100),uSum)


for j in range(0,len(xDatas)):
    xData = xDatas[j]
    yData = yDatas[j]   
    x_train,x_val,y_train,y_val,numDim = sig.preprocess(xData,yData)
    
    model = m.model_1()

    print("Fit model on training data")
    history = model.fit(
        x_train,
        y_train,
        batch_size=16,
        epochs=100,
        # We pass some validation for
        # monitoring validation loss and metrics
        # at the end of each epoch
        validation_data=(x_val, y_val),
    )
    modelList.append(model)
    modelpath = "/Users/RileyBallachay/Documents/Fifth Year/RNNSystemIdentification/Models/"+str(j)+".h5"
    model.save(modelpath)
    plt.figure(dpi=100)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
    predictions = modelList[j].predict(x_val)
    
    plt.figure(dpi=100)
    plt.plot(y_val,predictions,'g.')
    plt.plot(np.linspace(1,10),np.linspace(1,10),'r--')
    plt.title('Predictive accuracy')
    plt.xlabel('Predicted Value of Tau')
    plt.ylabel('True Value of Tau')
    print(r2_score(y_val,predictions))
    plt.show()

# This segment of code is for loading the models if they are already saved
"""
for filename in os.listdir("/Users/RileyBallachay/Documents/Fifth Year/RNNSystemIdentification/Models/"):
    if filename.endswith(".h5"):
        name = "/Users/RileyBallachay/Documents/Fifth Year/RNNSystemIdentification/Models/" +filename
        print(name)
        modelList.append(keras.models.load_model(name))
"""   

xData = xDatas[0]
yData = yDatas[0]   
x_train,x_val,y_train,y_val,numDim = sig.preprocess(xData,yData)
tauPredictions = modelList[2].predict(x_val)
kpPredictions = modelList[1].predict(x_val)

xData = xDatas[2]
yData = yDatas[2]  
x_train,x_val,y_train,y_val,numDim = sig.preprocess(xData,yData)
thetaPredictions = modelList[0].predict(x_val)


uArrays = uArray[test,:]
yArrays = yArray[test,:]

for (i,index) in enumerate(test):
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
