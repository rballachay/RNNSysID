#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 14:50:26 2020

@author: RileyBallachay
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import scipy.signal
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from scipy.fft import fft
from sklearn.preprocessing import StandardScaler
from scipy.stats import kurtosis,skew,entropy,variation,gmean
from sklearn.metrics import r2_score
import random

class Signal:
 
    # Going to need this to be interactive 
    # i.e. the function waits until the 
    def __init__(self,nstep):
        self.nstep = nstep 

    def random_signal(self):
        y = np.random.rand(self.nstep)
        y[:10] = 0
        y[-10:] = 0     
        win = scipy.signal.hann(10)-0.5
        filtered = scipy.signal.convolve(y, win, mode='same') / sum(win)
        return filtered

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
def FOmodel(y,t,timearray,inputarray,Kp,taup):   
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

numTrials = 1000

# Simulate taup * dy/dt = -y + K*u
u3Array = np.full((numTrials,100),0.)
y3Array = np.full((numTrials,100),0.)
y3_1Array = np.full((numTrials,100),0.)

KpSpace = np.linspace(1,10,100)
taupSpace = np.linspace(1,10,100)
zetaSpace = np.linspace(0.1,1,100)

nstep = 100
timelength = 100
timestep = nstep/timelength
FOP = []
Kout = []
t3 = np.linspace(0,timelength,nstep)
iterator=0
Signal = Signal(nstep)

while(iterator<numTrials):
    index = np.random.randint(0,100)
    index1 = np.random.randint(0,100)
    index2 = np.random.randint(0,100)
    
    Kp = KpSpace[index]
    taup = taupSpace[index1]
    zeta = zetaSpace[index2]
    
    u3 = gaussNoise(Signal.random_signal())
    du3 = CPMethod(u3,timestep)
    y3 = gaussNoise(odeint(FOmodel,0,t3,args=(t3,u3,Kp,taup)).ravel())
    y3_1 = gaussNoise(odeint(SOmodel,[0,0],t3,args=(t3,u3,Kp,taup,0.5))[:,0])
    
    u3Array[iterator,:] = u3
    y3Array[iterator,:]= y3
    y3_1Array[iterator,:] = y3_1
    FOP.append((Kp,taup))
    iterator+=1
    

    plt.figure(dpi=100)
    #plt.plot(t3,u3,label='Input Signal')
    #plt.plot(t3,du3,label='Derivative of Input Signal')
    plt.plot(t3,y3-u3, label='First Order Response')
    plt.plot(t3,y3_1-u3, label="Second Order Response")
    plt.legend()


index = range(0,2*len(y3Array))
train = random.sample(index,int(0.7*2*numTrials))
test = [item for item in list(index) if item not in train]

xData = np.concatenate((y3Array,y3_1Array))
yData = np.concatenate((np.zeros(numTrials),np.ones(numTrials)))

trainspace = xData[train,:]
valspace = xData[test,:]

x_train= trainspace.reshape((int(numTrials*2*0.7),nstep,1))
y_train = yData[train]

x_val = valspace.reshape((int(numTrials*2*0.3),nstep,1))
y_val = yData[test]

model = keras.Sequential()

model.add(layers.LSTM(10,activation='tanh',input_shape=(nstep,1)))
model.add(layers.Dense(5,activation='sigmoid'))
model.add(layers.Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

model.summary()

print("Fit model on training data")
history = model.fit(
    x_train,
    y_train,
    batch_size=64,
    epochs=1000,
    # We pass some validation for
    # monitoring validation loss and metrics
    # at the end of each epoch
    validation_data=(x_val, y_val),
)

predictions = model.predict(x_val)