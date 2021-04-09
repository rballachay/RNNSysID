#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 11:08:56 2021

@author: RileyBallachay
"""

import control as control
import numpy as np
import pandas as pd
from Signal import Signal
import matplotlib.pyplot as plt
import seaborn as sns

path = '/Users/RileyBallachay/Documents/Fifth Year/RNNSystemIdentification/Model Validation/Paper Results 3.csv'

df = pd.read_csv(path)
df = df.loc[:, ~df.columns.str.contains('Unnamed')]

df[df['tau_python']>1] = 0
sig = Signal(1,1,10,numPlots=0)
sig.sys_simulation(stdev=5,order=1)
disturbances = np.zeros((1000,600))

sses=[] 
for i in range(0,1000):
   #plt.figure(i,dpi=200)
    #plt.plot(sig.ySteps[i,:,0])
   #plt.plot(sig.test_sequence,'k--',linewidth=1,label='Excitation')
    
    u = sig.test_sequence
    allY=np.zeros((600,1))
    # Iterate over each of the input dimensions
    # and add to the numerator array
    numTemp = [df['kp_true'][i]]
    denTemp =[1.,-df['tau_true'][i]]
    thetas = [df['theta_true'][i]]
    
    bigU=np.transpose(u)
    if thetas[0]<1:
        thetas[0]=1
    uSim = np.concatenate((np.zeros(thetas[0]),bigU[:-thetas[0]]))

    numTemp=np.array(numTemp);denTemp=np.array(denTemp)
    sys = control.tf(numTemp,denTemp,1)
    _,realy,_ = control.forced_response(sys,U=uSim,T=np.linspace(0,599,600))
    disturb = sig.add_disturbance()[:,0]
    amax = np.max(realy)/np.max(disturb)
    ratio = np.random.normal(0,0)*amax
    disturb = disturb*ratio
    disturbances[i,:]=disturb
    realy=realy+disturb
   #plt.plot(sig.gauss_noise(realy,'variable'),'green',linewidth=1,label='Real Response')
    
    
    for k in range (0,10):
        numgauss = np.random.normal(df['kp_MATLAB'][i],df['MATLAB_cov'][i]/2)
        dengauss = np.random.normal(df['tau_MATLAB'][i],df['MATLAB_cov.1'][i]/2)
        numTemp = [numgauss]
        denTemp =[1.,-dengauss]
        thetas = [df['theta_MATLAB'][i]]
        
        bigU=np.transpose(u)
        uSim=np.zeros_like(bigU)
        if thetas[0]<1:
            thetas[0]=1
        uSim = np.concatenate((np.zeros(thetas[0]),bigU[:-thetas[0]]))
        numTemp=np.array(numTemp);denTemp=np.array(denTemp)
        sys = control.tf(numTemp,denTemp,1)
        
        try:
            _,y,_ = control.forced_response(sys,U=uSim,T=np.linspace(0,599,600))
        except:
            y = np.zeros((600,1))
        
        
        #if k==9:
           #plt.plot(y,'b',linewidth=.5,label='Gaussian Predictions') 
        #else:
           #plt.plot(y,'b',linewidth=.5) 
    
    numgauss = df['kp_MATLAB'][i]
    dengauss = df['tau_MATLAB'][i]
    numTemp = [numgauss]
    denTemp =[1.,-dengauss]
    thetas = [df['theta_MATLAB'][i]]
    
    bigU=np.transpose(u)
    uSim=np.zeros_like(bigU)
    try:
        uSim = np.concatenate((np.zeros(thetas[0]),bigU[:-thetas[0]]))
    except:
        uSim = bigU

    numTemp=np.array(numTemp);denTemp=np.array(denTemp)
    sys = control.tf(numTemp,denTemp,1)
    try:
        _,y,_ = control.forced_response(sys,U=uSim,T=np.linspace(0,599,600))
    except:
        y = np.zeros((600,1))
    


    sse = np.sum((y.flatten()-realy.flatten())**2)
    sses.append(sse)
   #plt.plot(y,'darkblue',linewidth=1,label='Best MSE=%.3e'%(sse/600) )
   #plt.xlabel('Time Step (s)')
   #plt.ylabel('System Response')
    sns.set(font_scale = .75)
   #plt.legend()

sse1=sses
print(np.mean(sses))

sses=[]    
for i in range(0,1000):
   #plt.figure(100*i,dpi=200)
    #plt.plot(sig.ySteps[i,:,0])
   #plt.plot(sig.test_sequence,'k--',linewidth=1,label='Excitation')
    
    u = sig.test_sequence
    allY=np.zeros((600,1))
    # Iterate over each of the input dimensions
    # and add to the numerator array
    numTemp = [df['kp_true'][i]]
    denTemp =[1.,-df['tau_true'][i]]
    thetas = [df['theta_true'][i]]
    
    bigU=np.transpose(u)
    uSim=np.zeros_like(bigU)
    if thetas[0]<1:
        thetas[0]=1
    uSim = np.concatenate((np.zeros(thetas[0]),bigU[:-thetas[0]]))

    numTemp=np.array(numTemp);denTemp=np.array(denTemp)
    sys = control.tf(numTemp,denTemp,1)
    _,realy,_ = control.forced_response(sys,U=uSim,T=np.linspace(0,599,600))
    realy=realy+disturbances[i,:]
   #plt.plot(sig.gauss_noise(realy,'variable'),'green',linewidth=1,label='Real Response')
    
    
    
    for k in range (0,10):
        numgauss = np.random.normal(df['kp_python'][i],df['python_cov'][i]/2)
        dengauss = np.random.normal(df['tau_python'][i],df['python_cov.1'][i]/2)
        numTemp = [numgauss]
        denTemp =[1.,-dengauss]
        thetas = [df['theta_python'][i]]
        
        bigU=np.transpose(u)
        uSim=np.zeros_like(bigU)
        if thetas[0]<1:
            thetas[0]=1
        uSim = np.concatenate((np.zeros(thetas[0]),bigU[:-thetas[0]]))
    
        numTemp=np.array(numTemp);denTemp=np.array(denTemp)
        sys = control.tf(numTemp,denTemp,1)
        try:
            _,y,_ = control.forced_response(sys,U=uSim,T=np.linspace(0,599,600))
        except:
            y = np.zeros((600,1))
        
        #if k==9:
           #plt.plot(y,'r',linewidth=.5,label='Gaussian Predictions') 
        #else:
           #plt.plot(y,'r',linewidth=.5) 
            
        
    numgauss = df['kp_python'][i]
    dengauss = df['tau_python'][i]
    numTemp = [numgauss]
    denTemp =[1.,-dengauss]
    thetas = [df['theta_python'][i]]
    
    bigU=np.transpose(u)
    uSim=np.zeros_like(bigU)
    if thetas[0]<1:
        thetas[0]=1
    uSim = np.concatenate((np.zeros(thetas[0]),bigU[:-thetas[0]]))  

    numTemp=np.array(numTemp);denTemp=np.array(denTemp)
    sys = control.tf(numTemp,denTemp,1)
    try:
        _,y,_ = control.forced_response(sys,U=uSim,T=np.linspace(0,599,600))
    except:
        y = np.zeros((600,1))
        
        
    sse = np.sum((y.flatten()-realy.flatten())**2)
    sses.append(sse)
   #plt.plot(y,'darkred',linewidth=1,label='Best MSE=%.3e'%(sse/600)) 
   #plt.xlabel('Time Step (s)')
   #plt.ylabel('System Response')
    sns.set(font_scale = .75)
   #plt.legend()
print(np.mean(sses))