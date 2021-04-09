#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  6 09:23:09 2020

@author: RileyBallachay
"""

from Signal import Signal
from Model import Model
import tkinter as Tkinter
import random
import time
import control as control
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

class RollingSignal:
    
    def __init__(self):
        self.inDim=1
        self.outDim=1
        return
    
    def PRBS(self,length):  
         """Returns a pseudo-random binary sequence 
         which ranges between -1 and +1. This algorithm
         assumes the maximum time constant is 10, and uses
         the time constant to determine the """
         gbn = np.zeros(length)
         mean = 10*2*(1)
         loc=0
         currentval = np.random.choice([-1,1])
         i=0
         while loc<(length-1):
             stride = int(round(np.random.normal(mean,mean/2),0))
             if loc+stride>(length-1):
                 gbn[loc:] = currentval     
             else:
                 gbn[loc:loc+stride] = currentval
                 currentval = -currentval
             loc = loc+stride
             i+=1
        
         self.u = gbn[::-1]
         return gbn[::-1]
     
    def gauss_noise(self,array):
        """Generate gaussian noise with mean and standard deviation
        of 5% of the maximum returned value."""
        # If the array has 2 dimensions, this will capture it
        # Otherwise, it will evaluate the length of 1D array
        stdev = random.choice([1,2,3,4,5])
        noise = np.random.normal(0,(stdev/100)*np.amax(array),array.shape)
        return array+noise
    
    def y_func(self,u,kp,tau,theta):

        # Iterate over each of the input dimensions
        # and add to the numerator array
        numTemp = kp
        denTemp = [tau,1.]
        theta = int(theta)
        uSim = np.concatenate((np.zeros(theta),u[:-theta]))

        sys = control.tf(numTemp,denTemp)
        print(sys)
        arr = np.linspace(0,len(u),len(u))
        _,y,_ = control.forced_response(sys,T=arr,U=uSim)
        self.y = self.gauss_noise(y)
        
        return self.y
    
    def load_model(self,sig,directory=False,check=False):
        """Loads one of two first order models: probability or regular. Iterates 
        over directory and loads alphabetically. If more than 3 models exist in the 
        directory, it will load them indiscriminately."""
        modelList = []

        loadDir = directory
            
        models = ['kp.cptk','tau.cptk','theta.cptk']
        for filename in models:
            if filename=='.DS_Store':
                continue
            print(filename)

            model = Model.mutable_model_noAttribute(np.zeros((2,100,1)),np.zeros((3,1)))
                    
            model.load_weights(loadDir+filename)
            modelList.append(model)
        
        self.names = ['kp','tau','theta']
        self.modelDict  = {}
        for i in range(0,3):
            self.modelDict[self.names[i]] = modelList[i]

class App:
    def __init__(self, master):
        self._job = None
        self.max_time = 500
        # Create a container
        self.frame = Tkinter.Frame(master)
        self.master=master
        # Create 2 buttons      
        self.kp = Tkinter.Scale(self.frame, from_=1., to=10.,resolution=0.1,label="K  ",command=self.resetparam)
        self.kp.pack()
        
        self.tau = Tkinter.Scale(self.frame, from_=1., to=10.,resolution=0.1,label="τ   ",command=self.resetparam)
        self.tau.pack()
        
        self.theta = Tkinter.Scale(self.frame, from_=1., to=9.,resolution=0.5,label="θ",command=self.resetparam)
        self.theta.pack()

        self.fig = Figure((10,5))
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlim(0,500)
        
        self.fig2 = Figure((10,5))
        self.ax2 = self.fig2.add_subplot(111)
        
        sig = Signal(1,1,2,numPlots=0)
        sig.sys_simulation()
        self.masterRS = RollingSignal()
        self.masterRS.load_model(sig,directory='/Users/RileyBallachay/Documents/Fifth Year/RNNSystemIdentification/Model Validation/MIMO 1x1/Checkpoints/')
        
        self._roll_over(initial=True)

    def gaussian(self, param):
        mu,sig = param
        mu = np.array(mu).flatten()
        sig=np.array(sig).flatten()
        x=np.linspace(0,10,1000)
        return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
    
    
    def resetparam(self, event):
        if self._job:
            self.master.after_cancel(self._job)
        self._job = self.master.after(2000, self._roll_over)
     
    def _roll_over(self,initial=False):
        self.rs = RollingSignal()
        self.uTotal = self.rs.PRBS(self.max_time)
        self.time_elapsed = 0
        self._execute(initial)
        
    def _execute(self,initial=False):
        self.time_elapsed+=10
        if self.time_elapsed>self.max_time:
            self._roll_over()
        
        self.u = self.uTotal[:self.time_elapsed]
        self.y = self.rs.y_func(self.u,self.kp.get(),self.tau.get(),self.theta.get())
        self.ax.clear()
        self.ax2.clear()
        self.l = self.ax.plot(self.u)
        self.li = self.ax.plot(self.y)
        self.ax.set_xlim(0,500)
        
        
        in1 =  (self.u*self.y).reshape(1,self.time_elapsed,1)
        in2 = (self.y-self.u).reshape(1,self.time_elapsed,1)
        
        kp = (self.masterRS.modelDict['kp'](in1).mean(),self.masterRS.modelDict['kp'](in1).stddev())
        tau = (self.masterRS.modelDict['tau'](in2).mean(),self.masterRS.modelDict['tau'](in2).stddev())
        theta = (self.masterRS.modelDict['theta'](in2).mean(),self.masterRS.modelDict['theta'](in2).stddev())
        
        self.line3 = self.ax2.plot(np.linspace(0,10,1000),self.gaussian(kp),label='Kp=%.1f' %kp[0])
        self.line4 = self.ax2.plot(np.linspace(0,10,1000),self.gaussian(tau),label='τ=%.1f' %tau[0])
        self.line4 = self.ax2.plot(np.linspace(0,10,1000),self.gaussian(theta),label='θ=%.1f' %theta[0])
        self.ax2.set_xticks(range(0,10), minor=True)
        self.ax2.legend()
        
        
        if initial:
            self.canvas = FigureCanvasTkAgg(self.fig,master=self.master)
        
        self.canvas.draw()
        
        if initial:
            self.canvas.get_tk_widget().pack(side='right', fill='both')
            
        if initial:
            self.canvas2 = FigureCanvasTkAgg(self.fig2,master=self.master)
        
        self.canvas2.draw()
        
        if initial:
            self.canvas2.get_tk_widget().pack(side='left', fill='both')
            self.frame.pack()
        

def main():
    root = Tkinter.Tk()
    app = App(root)
    while True:
        time.sleep(2)
        root.update_idletasks()
        root.update()
        app._execute()
#root.mainloop()

if __name__ == '__main__':
    main()
    
    
    