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
        # random signal generation
        a = np.random.randint(-1, 2, self.nstep)  # range for amplitude
        b = np.random.randint(5, 10, self.nstep) # range for freuency
        
        for index in range(1,np.size(b)):
            b[index] = b[index-1]+b[index]
         
        # Random Signal
        index=0
        random_signal = np.zeros(nstep)
        while b[index]<np.size(random_signal):
            k = b[index]
            random_signal[k:] = float(a[index])
            index=index+1
            
        return random_signal
    
    def wave_signal(self):
        #y = np.random.rand(self.nstep)
        #y[:10] = 0
        #y[-10:] = 0     
        #win = scipy.signal.hann(10)
        #filtered = scipy.signal.convolve(y, win, mode='same') / sum(win)
        t=np.full((100),0.)
        t[25:-25]=np.linspace(0,50,50)/np.pi
        filtered = np.sin(t)
        return filtered

    def pulsed_signal(self):
        # pulsed signal generation
        signal = np.zeros(nstep)
        start1 = int(nstep/5)
        end1 = int(start1*2)
        start2 = int(start1*3)
        end2 = int(start1*4)
        
        signal[start1:end1] = 1
        signal[start2:end2] = 1
            
        return signal


    def step_function(self):  
        t = np.zeros((self.nstep))
        t[0:]= 1
        return t
  
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

# ODE Integrator
def model(y,t,timearray,inputarray,Kp,taup):   
    index = find_nearest(timearray,t)
    u = inputarray[index]
    return (-y + Kp * u)/taup

# ODE Integrated
def firstOrderModelSolved(t,Kp,taup):
    y = Kp*(1-np.exp(-t*taup))
    return y


# Model for a first order response to a step change of A=1
def FOSC(u,t,Kp,taup):
    response=np.zeros((100))
    response = Kp*(1-np.exp((t-25)/taup))
    response[0:25] = 0
    return response

def preprocess(y3):
    # This are parameters which are used to transform a first order system
    # into an appropriate range
    ceiling = max(y3)
    return ceiling

def noise(array):
    perturbation = np.random.uniform(-0.1,0.1,size=(100,))
    return array+perturbation

# Simulate taup * dy/dt = -y + K*u
u3Array = np.full((1000,100),0.)
y3Array = np.full((1000,100),0.)

KpSpace = np.linspace(1,10,10)
taupSpace = np.linspace(1,10,100)
nstep = 100
FOP = []
Kout = []
t3 = np.linspace(0,100,nstep)
iterator=0
Signal = Signal(nstep)

for (i,Kp) in enumerate(KpSpace):
    for (j,taup) in enumerate(taupSpace):
        y3 = firstOrderModelSolved(t3,Kp,taup)
        ceiling = preprocess(y3)
        Kout.append(ceiling)   
        u3 = noise(list(Signal.step_function())[:])
        u3Array[iterator,:] = u3
        y3Array[iterator,:]= noise(odeint(model,0,t3,args=(t3,u3,Kp,taup)).ravel()/ceiling)
        FOP.append((Kp,taup))
        iterator+=1

diffspace = y3Array - u3Array

'''
scaler = StandardScaler()
scaler.fit(diffspace)
diffspace = scaler.transform(diffspace)
'''

index = range(0,len(diffspace))

train = random.sample(index,700)
test = [item for item in list(index) if item not in train]

KpSpace = np.array([i[0] for i in FOP]).reshape(1000,1)
taupSpace = np.array([i[1] for i in FOP]).reshape(1000,1)

trainspace = diffspace[train,:]
valspace = diffspace[test,:]

x_train= trainspace.reshape((700,100))
y_train = np.full((700),0.)
#y_train[:,0] = KpSpace[::2,0]
y_train[:] = taupSpace[train,0]

x_val = valspace.reshape((300,100))
y_val = np.full((300),0.)
#y_val[:,0] = KpSpace[1::2,0]
y_val[:] = taupSpace[test,0]

model = keras.Sequential()

model.add(layers.Dense(250,activation='sigmoid',input_shape=(100,)))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(50,activation='sigmoid'))
model.add(layers.Dense(1, activation='linear'))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

model.summary()

print("Fit model on training data")
history = model.fit(
    x_train,
    y_train,
    batch_size=2,
    epochs=100,
    # We pass some validation for
    # monitoring validation loss and metrics
    # at the end of each epoch
    validation_data=(x_val, y_val),
)

predictions = model.predict(x_val)

plt.plot(y_val,predictions,'g.')
plt.plot(np.linspace(1,10),np.linspace(1,10),'r--')
print(r2_score(y_val,predictions))
