#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 17:32:17 2020

@author: RileyBallachay
"""
import os
import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from scipy.fft import fft
from sklearn.preprocessing import StandardScaler
from scipy.stats import kurtosis,skew,entropy,variation,gmean
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

class Model:    
    
    # Initialize the model instace 
    def __init__(self,nstep):         
        self.nstep=nstep
        self.names = ["kp","tau","theta"]
        self.modelDict = {}
        return
    
    def load_FOPTD(self):
        modelList = []
        for filename in os.listdir("/Users/RileyBallachay/Documents/Fifth Year/RNNSystemIdentification/Models/"):
            if filename.endswith(".h5"):
                name = "/Users/RileyBallachay/Documents/Fifth Year/RNNSystemIdentification/Models/" +filename
                print(name)
                modelList.append(keras.models.load_model(name))
        for i in range(0,3):
            self.modelDict[self.names[i]] = modelList[i]
        return
    
    def train_FOPTD(self,xDatas,yDatas,sig=False,plotLoss=True,plotVal=False):
    
        # You have to construct a signal with all the necessary parameters before 
        if not(sig):
            print("Please initialize the class signal with model parameters first")
            return 
        for j in range(0,len(xDatas)):
            xData = xDatas[j]
            yData = yDatas[j]   
            x_train,x_val,y_train,y_val,numDim = sig.preprocess(xData,yData)
            
            model = self.FOPTD()
            modelList = []
        
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
            self.modelDict[self.names[j]] = modelList[j]

            modelpath = "/Users/RileyBallachay/Documents/Fifth Year/RNNSystemIdentification/Models/"+self.names[j]+".h5"
            fileOverwriter=0
            while os.path.isfile(modelpath):
                modelpath = "/Users/RileyBallachay/Documents/Fifth Year/RNNSystemIdentification/Models/"+self.names[j]+"_"+str(fileOverwriter)+".h5"
                fileOverwriter+=1
            model.save(modelpath)
            
            if plotLoss:
                plt.figure(dpi=100)
                plt.plot(history.history['loss'])
                plt.plot(history.history['val_loss'])
                plt.title('model loss')
                plt.ylabel('loss')
                plt.xlabel('epoch')
                plt.legend(['train', 'test'], loc='upper left')
                plt.show()
            
            predictions = modelList[j].predict(x_val)
            
            if plotVal:
                plt.figure(dpi=100)
                plt.plot(y_val,predictions,'g.')
                plt.plot(np.linspace(1,10),np.linspace(1,10),'r--')
                plt.title('Predictive accuracy')
                plt.xlabel('Predicted Value of Tau')
                plt.ylabel('True Value of Tau')
                print(r2_score(y_val,predictions))
                plt.show()
            return
        
        def FOPTD(self):
            model = keras.Sequential()
            
            # I tried almost every permuation of LSTM architecture and couldn't get it to work
            model.add(layers.GRU(100, activation='tanh',input_shape=(self.nstep,1)))
            model.add(layers.Dense(100, activation='linear',))
            model.add(layers.Dense(1, activation='linear'))
            
            # Compile the model
            model.compile(optimizer='adam', loss='mean_squared_error')
            return model   

   