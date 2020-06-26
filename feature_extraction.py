#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 19:19:14 2020

@author: RileyBallachay
"""

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

class Signal:
 
    # Going to need this to be interactive 
    # i.e. the function waits until the 
    def __init__(self,nstep):
        self.nstep = nstep 


    def step_function(self):  
        t = np.zeros((self.nstep))
        t[0:] = 1
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

# Model for a first order response to a step change of A=1
def FOSC(u,t,Kp,taup):
    response=np.zeros((100))
    response = Kp*(1-np.exp((t-25)/taup))
    response[0:25] = 0
    return response

def preprocess(u3,y3):
    # These functions assume that the step function will start at zero and proceed for the entire 
    
    MAXSTEP = max(u3)
    FEATLEN = 25 # Number of features to feed to the neural network
    MAX = float(max(y3)) # Max value from the array 
    MAXIN =len(y3) # The length of the data being fed into the neural network
    
    featureMap = np.full((FEATLEN,),0.0) # Full map of features (length can be adjusted)
    
    peak5 = featureMap[0:5] = (y3.argsort()[-5:][::-1]) # Index of peak 5 values
    featureMap[5:10] = y3[peak5[:]] # The peaks in raw form
    featureMap[0:5] = featureMap[0:5]/MAXIN # Transform index into fraction of impulse length
    
    peakfft5 = featureMap[10:15] = ((fft(y3).real).argsort()[-5:][::-1]) # Index of peak 5 fouriers
    featureMap[15:20] = fft(y3).real[peakfft5[:]] # Transform index into max fft vals
    featureMap[10:15] = featureMap[10:15]/MAXIN # Index as fraction of array length
    
    featureMap[20] = np.mean(y3) # Mean of the input data
    featureMap[21] = np.var(y3) # Variance of input data
    featureMap[22] = min(y3)/MAXSTEP
    featureMap[23] = max(y3)/MAXSTEP
    featureMap[24] = (max(y3) - min(y3))/MAXSTEP
    
    return featureMap[:]

mapped = np.full((100,25),0.)

KpSpace = np.linspace(1,10,10)
taupSpace = np.linspace(1,10,10)
nstep = 100
FOP = []
Sig = Signal(nstep)

for (i,Kp) in enumerate(KpSpace):
    for (j,taup) in enumerate(taupSpace):
        t3 = np.linspace(0,100,nstep)
        u3 = Sig.step_function()
        y3 = odeint(model,0,t3,args=(t3,u3,Kp,taup))
        mapped[10*i+j,:] = preprocess(u3,y3.ravel())        

        FOP.append((Kp,taup))
        
KpSpace = np.array([i[0] for i in FOP]).reshape(100,1)
taupSpace = np.array([i[1] for i in FOP]).reshape(100,1)
        
scaler = StandardScaler()
scaler.fit(mapped)
mapped = scaler.transform(mapped)


trainspace = mapped[::2,:]
valspace = mapped[1::2,:]

x_train= trainspace.reshape((50,25,))
y_train = np.full((50,1),0.)
y_train[:,0] = KpSpace[::2,0]
#y_train[:,0] = taupSpace[::2,0]

x_val = valspace.reshape((50,25,))
y_val = np.full((50,1),0.)
y_val[:,0] = KpSpace[1::2,0]
#y_val[:,0] = taupSpace[1::2,0]

model = keras.Sequential()

model.add(layers.Dense(10,activation='sigmoid',input_shape=(25,)))
model.add(layers.Dense(5,activation='sigmoid'))
model.add(layers.Dense(1, activation='linear'))

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