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

path = '/Users/RileyBallachay/Documents/Fifth Year/RNNSystemIdentification/Model Validation/Paper Results_order2.csv'

df = pd.read_csv(path)
df = df.loc[:, ~df.columns.str.contains('Unnamed')]

sig = Signal(1,1,10,numPlots=0)
sig.sys_simulation(stdev='variable',order=1)

sses=[] 
for i in range(10,20):
    plt.figure(i,dpi=200)
    #plt.plot(sig.ySteps[i,:,0])
    plt.plot(sig.test_sequence,'k--',linewidth=1)
    
    u = sig.test_sequence
    allY=np.zeros((600,1))
    # Iterate over each of the input dimensions
    # and add to the numerator array
    numTemp = [df['kp_real'][i],0]
    denTemp =[1.,-df['tau_real_1'][i],-df['tau_real_2'][i]]
    thetas = [df['theta_real'][i]]
    
    bigU=np.transpose(u)
    uSim=np.zeros_like(bigU)
    uSim = np.concatenate((np.zeros(thetas[0]),bigU[:-thetas[0]]))

    numTemp=np.array(numTemp);denTemp=np.array(denTemp)
    sys = control.tf(numTemp,denTemp,1)
    _,realy,_ = control.forced_response(sys,U=uSim,T=np.linspace(0,599,600))
    plt.plot(sig.gauss_noise(realy,'variable'),'green')
    
    
    
    for k in range (0,10):
        numgauss = np.random.normal(df['kp_MATLAB'][i],df['unc_MATLAB'][i]/2)
        dengauss = np.random.normal(df['tau_MATLAB'][i],df['unc_MATLAB.1'][i]/2)
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
        _,y,_ = control.forced_response(sys,U=uSim,T=np.linspace(0,599,600))
        plt.plot(y,'b',linewidth=.5)
    
    numgauss = df['kp_MATLAB'][i]
    dengauss = df['tau_MATLAB'][i]
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
    _,y,_ = control.forced_response(sys,U=uSim,T=np.linspace(0,599,600))
    
    sse = np.sum((y.flatten()-realy.flatten())**2)
    sses.append(sse)
    plt.plot(y,'darkblue',linewidth=1,label='MSE=%.3f'%(sse/600)) 
    plt.xlabel('Time Step (s)')
    plt.ylabel('System Response')
    sns.set(font_scale = .75)
    plt.legend()

print(np.mean(sses))

sses=[]    
for i in range(10,20):
    plt.figure(100*i,dpi=200)
    #plt.plot(sig.ySteps[i,:,0])
    plt.plot(sig.test_sequence,'k--',linewidth=1)
    
    u = sig.test_sequence
    allY=np.zeros((600,1))
    # Iterate over each of the input dimensions
    # and add to the numerator array
    numTemp = [df['kp_real'][i],0]
    denTemp =[1.,-df['tau_real_1'][i],-df['tau_real_2'][i]]
    thetas = [df['theta_real'][i]]
    
    bigU=np.transpose(u)
    uSim=np.zeros_like(bigU)
    if thetas[0]<1:
        thetas[0]=1
    uSim = np.concatenate((np.zeros(thetas[0]),bigU[:-thetas[0]]))

    numTemp=np.array(numTemp);denTemp=np.array(denTemp)
    sys = control.tf(numTemp,denTemp,1)
    _,realy,_ = control.forced_response(sys,U=uSim,T=np.linspace(0,599,600))
    plt.plot(sig.gauss_noise(realy,'variable'),'green')
    
    
    
    for k in range (0,10):
        numgauss = np.random.normal(df['kp_python'][i],df['unc_python'][i]/2)
        dengauss = np.random.normal(df['tau_python'][i],df['unc_python.1'][i]/2)
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
        _,y,_ = control.forced_response(sys,U=uSim,T=np.linspace(0,599,600))
        plt.plot(y,'r',linewidth=.5)   
        
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
    _,y,_ = control.forced_response(sys,U=uSim,T=np.linspace(0,599,600))
    
    sse = np.sum((y.flatten()-realy.flatten())**2)
    sses.append(sse)

    plt.plot(y,'darkred',linewidth=1,label='MSE=%.3f'%(sse/600)) 
    plt.xlabel('Time Step (s)')
    plt.ylabel('System Response')
    sns.set(font_scale = .75)
    plt.legend()
print(np.mean(sses))