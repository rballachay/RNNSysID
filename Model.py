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
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from Signal import Signal
from tensorflow.keras import backend as K
import datetime

class MyThresholdCallback(tf.keras.callbacks.Callback):
    def __init__(self, threshold):
        super(MyThresholdCallback, self).__init__()
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None): 
        val_acc = logs["val_coeff_determination"]
        if val_acc >= self.threshold:
            self.model.stop_training = True
            
class Model:    
    
    # Initialize the model instace 
    def __init__(self,nstep):         
        self.nstep=nstep
        self.names = ["kp","tau","theta"]
        self.modelDict = {}
    
    def load_FOPTD(self):
        modelList = []
        for filename in os.listdir("/Users/RileyBallachay/Documents/Fifth Year/RNNSystemIdentification/Models/"):
            if filename.endswith(".h5"):
                name = "/Users/RileyBallachay/Documents/Fifth Year/RNNSystemIdentification/Models/" +filename
                print(name)
                modelList.append(keras.models.load_model(name))
        for i in range(0,3):
            self.modelDict[self.names[i]] = modelList[i]

    
    def train_FOPTD(self,sig=False,plotLoss=True,plotVal=True,probabilistic=True,epochs=100,saveModel=True):
        yArray = sig.yArray; uArray = sig.uArray
        taus=sig.taus; kps=sig.kps; thetas=sig.thetas
        xDatas = [yArray,yArray,(yArray-np.mean(yArray))/np.std(yArray) - uArray]
        yDatas = [kps, taus, thetas]
        
        # You have to construct a signal with all the necessary parameters before 
        if not(sig):
            print("Please initialize the class signal with model parameters first")
            return 
        
        if saveModel:
            parentDir = "/Users/RileyBallachay/Documents/Fifth Year/RNNSystemIdentification/Models/"
            time = str(datetime.datetime.now())[:16]
            plotDir = parentDir + time + '/'
            os.mkdir(plotDir)
        
        modelList = []
        for j in range(0,len(xDatas)):
            xData = xDatas[j]
            yData = yDatas[j]   
            x_train,x_val,y_train,y_val,numDim = sig.preprocess(xData,yData)
            
            if probabilistic:
                 probModel = "prob"
                 model = self.__FOPTD_probabilistic()
            else:
                probModel = ""
                model = self.__FOPTD()
                
            MTC = MyThresholdCallback(0.95)
            print("Fit model on training data")
            history = model.fit(
                x_train,
                y_train,
                batch_size=16,
                epochs=epochs,
                # We pass some validation for
                # monitoring validation loss and metrics
                # at the end of each epoch
                validation_data=(x_val, y_val),
                callbacks=[MTC]
            )  
            
            modelList.append(model)
            self.modelDict[self.names[j]] = model
            
            if saveModel:
                modelpath = "/Users/RileyBallachay/Documents/Fifth Year/RNNSystemIdentification/Models/"+self.names[j]+probModel+".h5"
                fileOverwriter=0
                while os.path.isfile(modelpath):
                    modelpath = "/Users/RileyBallachay/Documents/Fifth Year/RNNSystemIdentification/Models/"+self.names[j]+probModel+"_"+str(fileOverwriter)+".h5"
                    fileOverwriter+=1
                model.save(modelpath)
            
            if plotLoss:
                plt.figure(dpi=100)
                plt.plot(history.history['loss'])
                plt.plot(history.history['val_loss'])
                plt.title('model loss for '+ self.names[j])
                plt.ylabel('loss')
                plt.xlabel('epoch')
                plt.legend(['train', 'test'], loc='upper left')
                if saveModel:
                    plt.savefig(plotDir+self.names[j]+'_loss'+'.png')
                plt.show()
                
            if probabilistic:
                predictions = modelList[j].predict(x_val)
                yhat = model(x_val)
                assert isinstance(yhat, tfd.Distribution)
                m = np.array(yhat.mean())
                s = np.array((yhat.stddev()))
            
            isWithin = 0
            for (i,mi) in enumerate(m):
                true = y_val[i]
                if ((m[i]+2*s[i] > true) and ((m[i]-2*s[i] < true))):
                    isWithin+=1
                else:
                    continue
            
            percentage = isWithin/len(m)
            print(percentage)

            if plotVal:
                plt.figure(dpi=100)
                plt.plot(y_val,m,'b.')
                plt.errorbar(y_val,m,yerr=s*2,fmt='none',ecolor='green')
                r2 =("r\u00b2 = %.3f" % r2_score(y_val,predictions))

                plt.plot(np.linspace(0,10),np.linspace(0,10),'r--',label = r2)
                    
                plt.title('Predictive accuracy')
                plt.ylabel('Predicted Value of ' + self.names[j])
                plt.xlabel('True Value of ' + self.names[j])
                plt.legend()
                if saveModel:
                    plt.savefig(plotDir+self.names[j]+'.png')
                plt.show()
            
        return m,s
    
    def MSE(self,y_true,y_pred):
        mse = sum((y_true-y_pred)**2)
        return mse
    
    def predict(self,sig,plotPredict=True,savePredict=False):
        if not(isinstance(sig,Signal)):
            print("You need to predict with an instance of signal!")
            return
        
        self.kpPredictions = self.modelDict['kp'].predict(sig.xData['kp'])
        self.tauPredictions = self.modelDict['tau'].predict(sig.xData['tau'])
        self.thetaPredictions = self.modelDict['theta'].predict(sig.xData['theta'])
        
        self.errors = []
        uArrays = sig.uArray[sig.train,:]
        yArrays = sig.yArray[sig.train,:]
        
        for (i,index) in enumerate(sig.train):
            taup = self.tauPredictions[i]
            Kp = self.kpPredictions[i]
            theta = self.thetaPredictions[i]
            t = np.linspace(0,sig.timelength,self.nstep)
            u = uArrays[i,:]
            yPred = (odeint(sig.FOmodel,0,t,args=(t,u,Kp,taup,theta),hmax=1.).ravel())
            yTrue = yArrays[i,:]
            self.errors.append(self.MSE(yPred,yTrue))
            
            if plotPredict:
                plt.figure(dpi=100)
                plt.plot(t,u,label='Input Signal')
                
                s1 = ("Modelled: Kp:%.1f τ:%.1f θ:%.1f" % (sig.kps[i],sig.taus[i],sig.thetas[i]))
                s2 = ("Predicted: Kp:%.1f τ:%.1f θ:%.1f" % (Kp,taup,theta))

                plt.plot(t,yTrue, label=s1)
                plt.plot(t,yPred,'--', label=s2)
                plt.xlabel("Time (s)")
                plt.ylabel("Change from set point")
                
                plt.legend()
           
            if savePredict:
                savePath = "/Users/RileyBallachay/Documents/Fifth Year/RNNSystemIdentification/Predictions/" + str(i) + ".png"
                plt.savefig(savePath)
                
    
    def __FOPTD(self):
        model = keras.Sequential()
        # I tried almost every permuation of LSTM architecture and couldn't get it to work
        model.add(layers.GRU(100, activation='tanh',input_shape=(self.nstep,1)))
        model.add(layers.Dense(100, activation='linear',))
        model.add(layers.Dense(100, activation='linear',))
        model.add(layers.Dense(1, activation='linear'))
        # Compile the model
        model.compile(optimizer='adam', loss='mean_squared_error',metrics=[self.coeff_determination])
        return model   
    
    def scale_data(self,sig):
        inputMax = np.max(np.abs(sig.uArray))
        outputMax = np.max(np.abs(sig.yArray))
        
        if inputMax!=1:
            print("Warning, the input data isn't bounded in range\
                  [-1,1]. Input and output scaling will be accounted for in the\
                final result")
            self.inputScaler = inputMax
            sig.uArray = sig.uArray/inputMax
            sig.yArray = sig.yArray/inputMax
        
        if outputMax>10:
            print("Warning, the output data isn't bounded in range\
                  [-10,10]. Output scaling will be accounted for in the\
                final result")
            self.outputScaler = outputMax/10
            sig.uArray = sig.uArray/outputMax
            sig.yArray = sig.yArray/outputMax 
    
    def coeff_determination(self,y_true, y_pred):
        SS_res =  K.sum(K.square( y_true-y_pred ))
        SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
        return ( 1 - SS_res/(SS_tot + K.epsilon()) )

    def __FOPTD_probabilistic(self): 
        model = tf.keras.Sequential([
        tf.keras.layers.GRU(100, activation='tanh',input_shape=(self.nstep,1)),
        tf.keras.layers.Dense(100, activation='linear'),
        tf.keras.layers.Dense(100, activation='linear'),
        tf.keras.layers.Dense(1 + 1),
        tfp.layers.DistributionLambda(
              lambda t: tfd.Normal(loc=t[..., :1],scale= 0.5 + 50*(tf.math.softplus(0.05*t[...,1:])))),
        ])
        model.compile(optimizer='adam', loss='mean_squared_error',metrics=[self.coeff_determination])
        return model
            
   