#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 10:44:44 2020

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
    
class Probability:
    
    def __init__(self,maxError=5,numTrials=1000,plotUncertainty=True):
        self.plotUncertainty = plotUncertainty
        self.maxError = maxError
        self.deviations = np.arange(0,maxError)
        self.numTrials = numTrials
        self.nstep = 100
        self.timelength = 100
        self.trainFrac = 0.7
        if (maxError==5 and numTrials==1000):
            suffix = "Default"
        else:
            suffix = 'error_' + str(maxError) + '_nTrials_' + str(numTrials)
        self.errorCSV = "/Users/RileyBallachay/Documents/Fifth Year/RNNSystemIdentification/Uncertainty/"+suffix+".csv"
        self.produce_simulation()
        
    def produce_simulation(self):    
        
        if not(path.exists(self.errorCSV)):
            print("No simulation for these parameters exists in Uncertainty data. Proceeding with simulation")
           
            # Initialize the models that are saved using the parameters declared above
            predictor = Model(self.nstep)
            predictor.load_FOPTD()
                
            deviations = np.arange(0,self.maxError)
            
            stdev = np.array([0])
            error=np.array([0])
            kp_pred = np.array([0])
            theta_pred = np.array([0])
            tau_pred = np.array([0])
            
            kp_true = np.array([0])
            theta_true = np.array([0])
            tau_true = np.array([0])
            
            for deviation in deviations:
                numTrials = self.numTrials; nstep = self.nstep
                timelength = self.timelength; trainFrac = self.trainFrac
                # then simulates using the initialized model
                sig = Signal(numTrials,nstep,timelength,trainFrac)
                sig.training_simulation(KpRange=[1,10],tauRange=[1,10],thetaRange=[1,10])
                
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
            
            sd.to_csv(self.errorCSV, index=False)
            
        else:
            print("Data exists for the parameters, proceeding to producing uncertainty estimate")
            try:
                sd = pd.read_csv(self.errorCSV).drop(['Unnamed: 0'],axis=1)
                sd.drop(sd.tail(1).index,inplace=True)
            except:
                sd = pd.read_csv(self.errorCSV)
                sd.drop(sd.tail(1).index,inplace=True)
            
        self.errorDict = {}  
        
        prefixes = ['kp','tau','theta']
        for prefix in prefixes:
            sd[prefix+'Error'] = (sd[prefix+'Pred']-sd[prefix+'True'])
            h = np.std(sd[prefix+'Error'])
            self.errorDict[prefix] = h
            
            if self.plotUncertainty:
                plt.figure(dpi=200)
                plt.hist(sd[prefix+'Error'],bins=100)
                plt.xlabel('Standard Error in '+ prefix)
                plt.ylabel("Frequency Distribution")
                
                plt.figure(dpi=200)
                plt.plot(sd[prefix+'True'],sd[prefix+'Pred'],'.')
                plt.plot(np.linspace(1,10),np.linspace(1,10),'r--')
                plt.plot(np.linspace(1,10),np.linspace(1,10)+h,'g--')
                plt.plot(np.linspace(1,10),np.linspace(1,10)-h,'g--')
        
        def get_errors(self):
            return self.errorDict.values

