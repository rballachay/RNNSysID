#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 14:50:26 2020

@author: RileyBallachay
"""
import numpy as np
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
from sippy import *

class Signal:
 
    # Going to need this to be interactive 
    # i.e. the function waits until the 
    def __init__(self,nstep):
        self.nstep = nstep 

    def random_signal(self):
        y = np.random.rand(self.nstep)
        y[:10] = 0
        y[-10:] = 0     
        windowlength = np.random.randint(5,20)
        win = signal.hann(windowlength)-0.5
        filtered = signal.convolve(y, win, mode='same') / sum(win)
        return filtered
    
    def PRBS(self):
        # random signal generation
        b = np.random.randint(50,100, nstep) # range for frequency
        b[0:int(.2*self.nstep)] = 0
        
        for i in range(int(.2*self.nstep),np.size(b)):
            b[i] = b[i-1]+b[i] 
        
        # PRBS
        a = np.zeros(nstep)
        j = 0
        while j < nstep:
            a[j] = 1
            a[j+1] = 0
            j = j+2
        i=0
        prbs = np.zeros(nstep)
        while b[i]<np.size(prbs):
            k = b[i]
            prbs[k:] = a[i]
            i=i+1
        
        i=0
        while i<len(prbs)-5:
            if prbs[i+5]==1 and prbs[i]==0:
                prbs[i+1]=0.2
                prbs[i+2]=0.4
                prbs[i+3]=0.6
                prbs[i+4]=0.8
                i+=5
            elif prbs[i+2]==0 and prbs[i]==1:
                prbs[i+4]=0.2
                prbs[i+3]=0.4
                prbs[i+2]=0.6
                prbs[i+1]=0.8
                i+=5
            else:
                i+=1
      
        return prbs
    
# Generate gaussian noise
def gaussNoise(array):
    noise = np.random.normal(0,0.05,len(array))
    return array+noise

# Central point derivative of continuous data
def CPMethod(array,timestep):
    derivative = [np.mean(array[i-1:i+1])/(2*timestep) for i in range(1,len(array)-1)]
    
    initial =(array[1] - array[0])/timestep
    derivative.insert(0,initial)
    
    final = (array[-1] - array[-2])/timestep
    derivative.insert(len(derivative),final)
    
    return derivative

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

# ODE Integrator
def FOmodel(y,t,timearray,inputarray,Kp,taup,theta): 
    t=t-theta
    index = find_nearest(timearray,t)
    u = inputarray[index]
    return (-y + Kp * u)/taup

# ODE Integrator
def SOmodel(yy,t,timearray,inputarray,Kp,tau,zeta):
    index = find_nearest(timearray,t)
    du = (inputarray[index])
    y = yy[0]
    dydt = yy[1]
    dy2dt2 = (-2.0*zeta*tau*dydt - y + Kp*du)/tau**2
    return [dydt,dy2dt2]

def preprocess(y3):
    # This are parameters which are used to transform a first order system
    # into an appropriate range
    ceiling = max(y3)
    return ceiling

# Definition of constants for simulation
numTrials = 100
nstep = 100
timelength = 100
timestep = nstep/timelength
trainFrac = 0.7
valFrac  = 1-trainFrac


# Simulate taup * dy/dt = -y + K*u
uArray = np.full((numTrials,nstep),0.)
yArray = np.full((numTrials,nstep),0.)
y_1Array = np.full((numTrials,nstep),0.)
corrArray = np.full((numTrials,nstep),0.)
conArray = np.full((numTrials,nstep),0.)

KpSpace = np.linspace(1,10,nstep)
taupSpace = np.linspace(1,10,nstep)
zetaSpace = np.linspace(0.1,1,nstep)
thetaSpace = np.linspace(0,0,nstep)

taus = []
thetas=[]
kps=[]
Kout = []
t = np.linspace(0,timelength,nstep)
iterator=0
Signal = Signal(nstep)

while(iterator<numTrials):
    index = np.random.randint(0,nstep)
    index1 = np.random.randint(0,nstep)
    index2 = np.random.randint(0,nstep)
    
    Kp = KpSpace[index]
    taup = taupSpace[index1]
    theta = thetaSpace[index2]
    
    u = (Signal.random_signal())   
    y = (odeint(FOmodel,0,t,args=(t,u,Kp,taup,theta),hmax=5.).ravel())
    
    # THis was commented out because it messed with my results
    #y = (y-np.mean(y))/np.std(y)
    #u = (u-np.mean(u))/np.std(u)
    
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
    if (iterator)<3:
        plt.figure(dpi=100)
        plt.plot(t,u,label='Input Signal')
        plt.plot(t,y, label='FOPTD Response')
        plt.plot(t,correlation,label='Correlated')
        plt.plot(t,convolution,label='Convolution')
        plt.xlabel((taup))
        plt.legend()
        
    # Subsequently update the iterator to move down row
    iterator+=1
    
    
index = range(0,len(yArray))
train = random.sample(index,int(trainFrac*numTrials))
test = [item for item in list(index) if item not in train]

#xData = yArray
xData = np.stack((corrArray,conArray,yArray),axis=1)
yData = taus
numDim = xData.ndim 

trainspace = xData[train,:]
valspace = xData[test,:] 

x_train= trainspace.reshape((int(numTrials*trainFrac),nstep,numDim))
y_train = np.array([yData[i] for i in train])

x_val = valspace.reshape((int(numTrials*valFrac),nstep,numDim))
y_val = np.array([yData[i] for i in test])

model = keras.Sequential()

model.add(layers.LSTM(25, return_sequences=True,activation='tanh',input_shape=(nstep,numDim)))
model.add(layers.LSTM(5,activation='tanh'))
model.add(layers.Dense(1, activation='linear'))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

model.summary()

print("Fit model on training data")
history = model.fit(
    x_train,
    y_train,
    batch_size=32,
    epochs=250,
    # We pass some validation for
    # monitoring validation loss and metrics
    # at the end of each epoch
    validation_data=(x_val, y_val),
)

predictions = model.predict(x_val)

plt.figure(dpi=100)
plt.plot(y_val,predictions,'g.')
plt.plot(np.linspace(1,10),np.linspace(1,10),'r--')
plt.xlabel('Predicted Value of Theta')
print(r2_score(y_val,predictions))

# Note that convolving and correlating the random signal works, but the same 
# doesn't work for the PRBS - not sure why this is 


# This is the historical model that worked with the randomly oscillating data
#model.add(layers.LSTM(100, return_sequences=True,activation='tanh',input_shape=(nstep,numDim)))
#model.add(layers.LSTM(50,activation='tanh'))
#model.add(layers.Dense(10,activation='linear'))
#model.add(layers.Dense(1, activation='linear'))