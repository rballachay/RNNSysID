#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 16:25:30 2020

@author: RileyBallachay
"""
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

class Signal:
 
    # Going to need this to be interactive 
    # i.e. the function waits until the 
    def __init__(self,numTrials,nstep,timelength,trainFrac,numPlots=5,stdev=5):
        self.numTrials = numTrials
        self.nstep = nstep
        self.timelength = timelength
        self.trainFrac = trainFrac
        self.valFrac = 1-trainFrac
        self.numPlots = numPlots
        self.stdev=stdev

    # This module is for simulating a wave function with 
    # random frequency and amplitude within the specified range
    def random_signal(self):
        y = np.random.rand(self.nstep)
        y[:10] = 0
        y[-10:] = 0     
        windowlength = np.random.randint(5,20)
        win = signal.hann(windowlength)-0.5
        filtered = signal.convolve(y, win, mode='same') / sum(win)
        return filtered
    
    # This function returns a pseudo-random binary sequence 
    # which ranges between -1 and +1
    def PRBS(self, prob_switch=0.1, Range=[-1.0, 1.0]):        
        min_Range = min(Range)
        max_Range = max(Range)
        gbn = np.ones(self.nstep)
        gbn = gbn*random.choice([-1,1])
        for i in range(self.nstep - 1):
            prob = np.random.random()
            gbn[i + 1] = gbn[i]
            if prob < prob_switch:
                gbn[i + 1] = -gbn[i + 1]
        for i in range(self.nstep):
            if gbn[i] > 0.:
                gbn[i] = max_Range
            else:
                gbn[i] = min_Range
        return gbn
 
    # This function plots the parameter space for a first 
    # order plus time delay model in 3D coordinates
    def plotParameterSpace(self,x,y,z,trainID,valID):
        x=np.array(x); y=np.array(y); z=np.array(z)
        figgy = plt.figure(dpi=200)
        ax = Axes3D(figgy)
        xT = x[trainID]; xV = x[valID] 
        yT = y[trainID]; yV = y[valID]
        zT = z[trainID]; zV = z[valID]
        ax.scatter(xT,yT,zT,c='g',label="Training Data")
        ax.scatter(xV,yV,zV,c='purple',label="Validation Data")
        ax.set_xlabel("τ (Time Constant)")
        ax.set_ylabel("Kp (Gain)")
        ax.set_zlabel("θ (Delay)")
        ax.legend()
    
    # Generate gaussian noise with mean and standard deviation
    # of 5% of the maximum returned value. 
    def gaussNoise(self,array,stdev):
        noise = np.random.normal(0,(stdev/100)*max(array),len(array))
        return array+noise
    
    # Central point derivative of continuous data to be used 
    # in second order process modelling
    def CPMethod(self,array,timestep):
        derivative = [np.mean(array[i-1:i+1])/(2*timestep) for i in range(1,len(array)-1)]
        
        initial =(array[1] - array[0])/timestep
        derivative.insert(0,initial)
        
        final = (array[-1] - array[-2])/timestep
        derivative.insert(len(derivative),final)
        
        return derivative
    
    # Function which is used in combination with the ODEint method in
    # order to use input U as a quasi-continuous array
    def find_nearest(self,array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx
    
    # First order plus time delay model to be used for simulation
    # Note that this model includes time delay implicit within 
    # t=t-theta. 
    def FOmodel(self,y,t,timearray,inputarray,Kp,taup,theta): 
        t=t-theta
        if t<0:
            u=0
        else:
            index = self.find_nearest(timearray,t)
            u = inputarray[index]
        return (-y + Kp * u)/taup
    
    # Second order plus time delay model to be used for simulation once 
    # the fundamentals for first order plus time delay are hammered out
    def SOmodel(self,yy,t,timearray,inputarray,Kp,tau,zeta,theta):
        t=t-theta
        index = self.find_nearest(timearray,t)
        du = (inputarray[index])
        y = yy[0]
        dydt = yy[1]
        dy2dt2 = (-2.0*zeta*tau*dydt - y + Kp*du)/tau**2
        return [dydt,dy2dt2]
    
    # Function which simulates a signal and returns it in whichever 
    def training_simulation(self,KpRange=[1,10],tauRange=[1,10],thetaRange=[1,10],zetaRange=[0.1,1],stdev=5):
        # Access all the attributes from initialization
        numTrials=self.numTrials; nstep=self.nstep;
        timelength=self.timelength; trainFrac=self.trainFrac
        
        # Initialize the arrays which will store the simulation data
        uArray = np.full((numTrials,nstep),0.)
        yArray = np.full((numTrials,nstep),0.)
        y_1Array = np.full((numTrials,nstep),0.)
        corrArray = np.full((numTrials,nstep),0.)
        conArray = np.full((numTrials,nstep),0.)
        
        # Make arrays containing parameters tau, theta
        KpSpace = np.linspace(KpRange[0],KpRange[1],nstep)
        taupSpace = np.linspace(tauRange[0],tauRange[1],nstep)
        zetaSpace = np.linspace(zetaRange[0],zetaRange[1],nstep)
        thetaSpace = np.linspace(thetaRange[0],thetaRange[1],nstep)
        
        KpSpace[KpSpace==0] = 0.01
        taupSpace[taupSpace==0] = 0.01
        thetaSpace[thetaSpace==0] = 0.01
        
        taus = []
        thetas=[]
        kps=[]
        t = np.linspace(0,timelength,nstep)
        iterator=0
        
        while(iterator<numTrials):
            index = np.random.randint(0,nstep)
            index1 = np.random.randint(0,nstep)
            index2 = np.random.randint(0,nstep) 
            
            Kp = KpSpace[index]
            taup = taupSpace[index1]
            theta = thetaSpace[index2]
            
            u = (self.PRBS())   
            y = self.gaussNoise(odeint(self.FOmodel,0,t,args=(t,u,Kp,taup,theta),hmax=1.).ravel(),stdev)
            
            uArray[iterator,:] = u
            yArray[iterator,:]= y
            taus.append(taup)
            thetas.append(theta)
            kps.append(Kp)
            
            convolution =  signal.convolve(u, y, mode='same')
            convolution  = (convolution-np.mean(convolution))/np.std(convolution)
            
            correlation = signal.correlate(u,y,mode='same')
            correlation = (correlation-np.mean(correlation))/np.std(correlation)
            
            corrArray[iterator,:] = correlation
            conArray[iterator,:] = convolution
            
            
            # Only plot every 100 input signals
            if (iterator)<self.numPlots:
                plt.figure(dpi=100)
                plt.plot(t[:200],u[:200],label='Input Signal')
                plt.plot(t[:200],y[:200], label='FOPTD Response')
                #plt.plot(t,correlation,label='Correlated')
                #plt.plot(t,convolution,label='Convolution')
                plt.xlabel((taup))
                plt.legend()
                plt.show()
                
            # Subsequently update the iterator to move down row
            iterator+=1
        
        index = range(0,len(yArray))
        if self.trainFrac!=1:  
            train = random.sample(index,int(trainFrac*numTrials))
            test = [item for item in list(index) if item not in train]
        else:
            train=index
            test=[]
        
        # Make it so that any of these attributes can be accessed 
        # without needing to return them all from the function
        self.plotParameterSpace(taus,kps,thetas,train,test)
        self.uArray = uArray
        self.yArray = yArray
        self.correlation = correlation
        self.convolution = convolution
        self.taus = taus
        self.kps = kps
        self.thetas = thetas
        self.train = train
        self.test = test
        
        return uArray,yArray,corrArray,conArray,taus,kps,thetas,train,test
    
    # This function uses the training and testing indices produced during
    # simulate() to segregate the training and validation sets
    def preprocess(self,xData,yData):
        try:
            _,_,numDim= xData.shape
        except:
            numDim=1
            
        trainspace = xData[self.train,:]
        valspace = xData[self.test,:] 
        
        x_train= trainspace.reshape((int(self.numTrials*self.trainFrac),self.nstep,1))
        y_train = np.array([yData[i] for i in self.train])
        
        x_val = valspace.reshape((int(self.numTrials*(1-self.trainFrac)),self.nstep,1))
        y_val = np.array([yData[i] for i in self.test])
        
        return x_train,x_val,y_train,y_val,numDim
    
    # This function makes it easier to run a bunch of simulations and 
    # automatically return the validation and testing sets without 
    # calling each function separately. 
    def simulate_and_preprocess(self):
        # Since no training is occurring, can skip separation of testing and validation sets
        self.trainFrac = 1
        
        uArray,yArray,corrArray,conArray,taus,kps,thetas,train,test = self.training_simulation(stdev=self.stdev)
        xDatas = [yArray,yArray,(yArray-np.mean(yArray))/np.std(yArray) - uArray]
        yDatas = [taus, kps, thetas]
        
        self.xData ={};
        self.yData={}
        self.names = ["kp","tau","theta"]
        
        for (i,name) in enumerate(self.names):
            x,_,y,_,_ = self.preprocess(xDatas[i],yDatas[i])
            self.xData[name] = x
            self.yData[name] = y
        
        return self.xData,self.yData