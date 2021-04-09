#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 16:25:30 2020

@author: RileyBallachay
"""
import os
import numpy as np
import random
import math
import scipy
from pylfsr import LFSR
import matplotlib.pyplot as plt
from pathlib import Path
import scipy.signal as signal
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import max_len_seq
from joblib import Parallel, delayed
import control as control
import time

class Signal:
    """
    Class that produces input signals and output of system response.
    
    Uses either a wave-based input signal created using hann signal 
    or a pseudo-binary input signal with 2-10 steps in the signal window.
    System response can currently only be simulated as first order plus time
    delay SISO. In the future, MIMO will also be utilized. 
    
    The purpose of this class is to produce random input and output signals 
    from simulated systems that can be used to train the class Model, which
    will then predict the system parameters of other first-order systems. 
    
    Parameters
    ----------
    numTrials : int, default=100
        Integer ideally bounded between 10-1000. Warning, simulation can take
        a very long time if greater than 1000. Will determine the number 
        of simulations produced.
    
    nstep : int, default=100
        Number of steps in time frame. Majority of indexing used in signal 
        and response depend on the index of the time and input signal array,
        so this can be more important than timelength.
        
    timelength : float, default=100.
        The length of the interpreted input/output signal data. In order to 
        scale time constants appropriately, must be in seconds. Need more robust
        scaling method for absolute value of system parameters.
    
    trainFrac : float, default=0.7
        The fraction of data used for validation/testing after fitting model.
        If model is only used to predict, trainFrac is forced to 1.
    
    stdev : float, default=5.
        The standard deviation of Gaussian noise applied to output signal data
        to simulate error in real-world measurements. 
    
    Attributes
    ----------
    random_signal 
        Generates hann windows with varying width and amplitude and appends to
        produce a pseudo-random wave sequence. 
    
    PRBS
        Generates a pseudo-random binary signal with varying width. Frequency is
        random, depends on probability switch. 10% probability that the signal
        changes sign every time step. Average step width of 6.5.
        
    plot_parameter_space
        Produces 3D plot of all simulated parameters (a,b,k)
    
    gauss_noise
        Adds gaussian noise to input sequence and returns array with noise.
        
    find_nearest
        odeint is built to take constant or functions as attributes. In this case,
        u is an array, so find_nearest is used to find the nearest value in u array.
    
    FOmodel
        First order plus time delay model in state space format.
    
    training_simulation
        Iterates over the input parameter space and produces simulations which
        will subsequently be used to train a model.
    
    preprocess
        Separates data into training and validation sets and reshapes for input
        to the GRU model. 
    
    simulate_and_preprocess
        Function which produces data to be used directly in prediction. Cannot be 
        used if data is to be used in training.
    
    """
    
    def __init__(self,inDim,outDim,numTrials,trainFrac=0.7,numPlots=5,stdev=5):
        self.numTrials = numTrials
        self.nstep = 3*50*(inDim+outDim) 
        self.timelength = self.nstep
        self.trainFrac = trainFrac
        self.valFrac = 1-trainFrac
        self.numPlots = numPlots
        self.stdev=stdev
        self.special_value=-99
        self.startTime=time.time()
        self.inDim = inDim
        self.outDim = outDim
        self.maxLen = 5
        self.length = 300
        self.STDEVS = []
        self.pd = str(Path(os.getcwd()).parent)
        if self.inDim>1:
            self.trim = 300
        else:
            self.trim=0
    
    def PRBS_parameterization(self,numRegisters=9,force_maximum=True):
        """This function serves to determine the best PRBS 
        parameters to allow for identification. Based on
        work from Rivera and Jung, 2000: ”An integrated 
        identification and control design methodology 
        for multivariable process system applications”
        
        T_sw: Switching time 
        nr: Number of registers
        tau_L: Minimum time constant 
        tau_H: Maximum time constant 
        alpha: Closed-loop response speed 
        """
        try:
            if not(force_maximum):
                tau_L = self.a_possible_values[0]
                tau_H = self.a_possible_values[1]
            else: 
                tau_L = 0.01 ; tau_H = 0.99
        except:
            print("You haven't initialized a system, continuting with default PRBS")
            tau_L = 0.01 ; tau_H = 0.99
        
        # Five times the settling time is 99% settled
        T_sw = int(278*tau_H/20)
        
        """Yields MLSequence of 511. Sequence repeats after
        nr*T_sw sampling intervals. 
        """
        nr = numRegisters
        Ns = 2**nr-1
        
        return T_sw,Ns*T_sw

    
    def PRBS(self):  
        """Returns a pseudo-random binary sequence 
        which ranges between -1 and +1. This algorithm
        assumes the maximum time constant is 10, and uses
        the time constant to determine the """
        
        sample,max_len = self.PRBS_parameterization()
            
        if self.length>max_len:
            self.length=max_len
        
        L = LFSR(fpoly=[9,5],initstate ='random',verbose=False)
        L.runKCycle(int(np.floor(self.length/sample))+1)
        seq = L.seq
        PRBS = np.zeros(self.length)
        for i in range(0,int(self.length/sample)+1):                                
            if seq[i]==0:
                seq[i]=-1
            
            if sample*i+sample>=self.length:
                PRBS[sample*i:] = seq[i]
                break
        
            PRBS[sample*i:sample*i+sample]=seq[i]
        
        return PRBS
    
    def random_process(self):
        
        sample,max_len = self.PRBS_parameterization()
            
        if self.length>max_len:
            self.length=max_len
         
        # random signal generation

        a_range = [-1,1]
        a = np.random.rand(self.length) * (a_range[1]-a_range[0]) + a_range[0] # range for amplitude
        
        b_range = [10, 15]
        b = np.random.rand(self.length) *(b_range[1]-b_range[0]) + b_range[0] # range for frequency
        b = np.round(b)
        b = b.astype(int)
        
        b[0] = 0
        
        for i in range(1,np.size(b)):
            b[i] = b[i-1]+b[i]
            
        # Random Signal
        i=0
        random_signal = np.zeros(self.length)
        while b[i]<np.size(random_signal):
            k = b[i]
            random_signal[k:] = a[i]
            i=i+1
        return random_signal
        
         
    def plot_parameter_space(self,x,y,z,trainID,valID):
        """This function plots the parameter space for a first 
        order plus time delay model in 3D coordinates"""
        x=np.array(x); y=np.array(y); z=np.array(z)
        figgy = plt.figure(dpi=200)
        ax = Axes3D(figgy)
        xT = x[trainID]; xV = x[valID] 
        yT = y[trainID]; yV = y[valID]
        zT = z[trainID]; zV = z[valID]
        ax.scatter(xT,yT,zT,c='g',label="Training Data")
        ax.scatter(xV,yV,zV,c='purple',label="Validation Data")
        ax.set_xlabel("A (Denominator)")
        ax.set_ylabel("B (Numerator)")
        ax.set_zlabel("K (Time-Shift Operator)")
        ax.legend()
    
    def gauss_noise(self,array,stdev):
        """Generate gaussian noise with mean and standard deviation
        of 5% of the maximum returned value."""
        # If the array has 2 dimensions, this will capture it
        # Otherwise, it will evaluate the length of 1D array
        if stdev=='variable':
            stdev = abs(np.random.normal(5,1))
            self.STDEVS.append(stdev)
        noise = np.random.normal(0,(stdev/100)*np.max(abs(array)),array.shape)
        return array+noise 
    
    def serialized_checkpoint(self,iteration):
        """Checkpoint which is called when 
        .2 fraction of the way thru the data"""
        try:
            checkTime = float(time.time()) - self.Lasttime
        except:
            checkTime = float(time.time()) - self.startTime
            
        self.Lasttime = float(time.time())
        checkpoint = int(100*iteration/self.numTrials)
        print("Produced %i%% of the serialized data" %checkpoint)
        print("Estimated Time Remaining: %.1f s\n" % (checkTime*(100-checkpoint)/20))
    
    def add_disturbance(self):
        
        a,_,_ = self.uArray.shape
        self.randint = np.random.randint(a)
        # Create first PRBS outside of loop
        u = self.uArray[self.randint,:,:]
        # The transfer function from the 2nd input to the 1st output is
        # (3z + 4) / (6z^2 + 5z + 4).
        # num = [[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]]
        # Iterate over each of the output dimensions and
        # add to numerator 
        allY=np.zeros((self.nstep,self.outDim))
        for j in range(0,self.outDim):
            # Iterate over each of the input dimensions
            # and add to the numerator array
            numTemp = [random.choice(self.b_real_params) for i in range(self.inDim)]
            denTemp = [[1.,-random.choice(self.a_real_params)[0]] for i in range(self.inDim)]
            kVals = [random.choice(self.k_real_params)[0] for i in range(self.inDim)]
            
            bigU=np.transpose(u)
            uSim=np.zeros_like(bigU)
            for (idx,row) in enumerate(bigU):
                try:
                    uSim[idx,:] = np.concatenate((np.zeros(kVals[idx]),row[:-kVals[idx]]))
                except:
                    uSim[idx,:] = row
            numTemp=np.array([numTemp]);denTemp=np.array([denTemp])
            sys = control.tf(numTemp,denTemp,1)
            _,y,_ = control.forced_response(sys,U=uSim,T=self.t)
            
            steps = int(self.nstep/self.length)
            l=self.length
            for step in range(0,steps):
                allY[l*step:l*(step+1),j] = self.gauss_noise(y[l*step:l*(step+1)],self.stdev)
          
        return allY
    
    
    def y_map_function(self,iterator):
        """Function for producing system responses from 
        simulated parameter arrays produced in signal"""
        # If some combination of 20% done running, checkpoint to console  
        if iterator in self.milestones:
            self.serialized_checkpoint(iterator)
        
        # Create first PRBS outside of loop
        u = self.uArray[iterator,:,:]
        # The transfer function from the 2nd input to the 1st output is
        # (3z + 4) / (6z^2 + 5z + 4).
        # num = [[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]]
        # Iterate over each of the output dimensions and
        # add to numerator 
        allY=np.zeros((self.nstep,self.outDim))
        for j in range(0,self.outDim):
            # Iterate over each of the input dimensions
            # and add to the numerator array
            numTemp = [[self.b_real_params[iterator,self.outDim*j+i]] for i in range(self.inDim)]
            denTemp = [[1.,-self.a_real_params[iterator,self.outDim*j+i]] for i in range(self.inDim)]
            kVals = [self.k_real_params[iterator,self.outDim*j+i] for i in range(self.inDim)]
            
            bigU=np.transpose(u)
            uSim=np.zeros_like(bigU)
            for (idx,row) in enumerate(bigU):
                try:
                    uSim[idx,:] = np.concatenate((np.zeros(kVals[idx]),row[:-kVals[idx]]))
                except:
                    uSim[idx,:] = row
            numTemp=np.array([numTemp]);denTemp=np.array([denTemp])
            sys = control.tf(numTemp,denTemp,1)
            _,y,_ = control.forced_response(sys,U=uSim,T=self.t)
            
            steps = int(self.nstep/self.length)
            l=self.length
            
            if self.disturbance:
                disturbed = self.add_disturbance()
            else:
                disturbed = np.zeros_like(y)
            
            for step in range(0,steps):
                allY[l*step:l*(step+1),j] = self.gauss_noise(y[l*step:l*(step+1)],self.stdev)
            
            if self.disturbance:
                amax = np.max(allY)/np.max(disturbed)
                ratio = .25*amax
                disturbed = disturbed*ratio
                allY = allY + disturbed
        
        # Colors for plotting input/output signals properly
        colors = ['midnightblue','gray','darkgreen','crimson','olive','navy','lightcoral','indigo','darkcyan',
                  'coral','darkorange','navy','r']
        
        '''
        if iterator<10:
            fig, axes = plt.subplots(3, 1,figsize=(15,5),dpi=400) 
            plt.figure(figsize=(10,5),dpi=200)
            label1 = '$u_' + str(1) + '(t)$' 
            label2 = '$w_' + str(1) + '(t)$' 
            label3 = '$x_' + str(1) + '(t)$'
            label4 = '$y_' + str(1) + '(t)$'
            #plt.plot(self.t,self.uArray[iterator,:,:],colors[0],label=label1)
            #plt.plot(self.t,self.uArray[self.randint,:,:],colors[1],label=label2)
            axes[1].plot(self.t,disturbed,'navy',label=label2)
            axes[1].set(ylabel='w(t)')
            axes[0].plot(self.t,allY-disturbed,'red',label=label3)
            axes[0].set(ylabel='x(t)')
            axes[2].plot(self.t,allY,'purple',label=label4)
            #plt.ylabel("Measured Signal (5% Noise)")
            axes[2].set(ylabel='y(t)',xlabel='Time (s)')
        '''
        return allY

    
    def sys_simulation(self,stdev=5,b_possible_values=[.01,.99],
                       a_possible_values=[.01,.99],k_possible_values=[1,10],
                       order=False,disturbance=True,not_prbs=False):
        """
        Module which produces simulation of SISO/MIMO system given the input parameters. 
        Contains a loop which iterates for the total number of samples and appends
        to an array. 
        
        Uses pseudo-random binary signal with amplitude in [-1,1] and linear first 
        order plus dead time system, modelled using transfer function class in
        the Control package to simulate the linear system.
        
        Purpose is to produce simulated system responses with varying quantities 
        of noise to simulate real linear system responses in order to train and
        validate models built with the Model class.
        
        Parameters
        ----------
        inDim : int, default=2 
            Number of input variables to MIMO system. Currently only set up
            to handle MIMO system with 2 inputs and 2 outputs.
            
        outDim : int, default=2
            Number of output variables from MIMO system. Currently only 
            configured to handle MIMO with 2 inputs and 2 outputs.
        
        stdev : float, default=5.
            Standard deviation of gaussian error added to the simulated system.
            
        b_possible_values : tuple, default=(1,10)
            Possible range for gains. An equally spaced array between the maximum 
            and minimum are chosen based on the number of simulations.
        
        a_possible_values : tuple, default=(1,10)
            Possible range for time constant. An equally spaced array between the 
            maximum and minimum are chosen based on the number of simulations.
            
        """
        # Access all the attributes from initialization
        numTrials=self.numTrials; nstep=self.nstep; self.stdev = stdev
        timelength=self.timelength; trainFrac=self.trainFrac
        self.b_possible_values = b_possible_values; self.a_possible_values = a_possible_values
        self.k_possible_values = k_possible_values; self.order = order
        self.disturbance=disturbance ; self.not_prbs = not_prbs
        
        self.milestones = []
        # Loop to create milestones to checkpoint data creation
        for it in [2,4,6,8]:
           self.milestones.append(int((it/10)*numTrials))
            
        # Set the type of the simulation to inform the data split
        self.type = "MIMO"
        
        # Initialize the arrays which will store the simulation data
        orderList = []
        
        # Make arrays containing parameters a,b and k
        b_params_sampled = np.linspace(b_possible_values[0],b_possible_values[1],nstep)
        a_params_sampled = np.linspace(a_possible_values[0],a_possible_values[1],nstep)
        k_params_sampled = np.arange(k_possible_values[0],k_possible_values[1])
        self.t = np.linspace(0,timelength-1,nstep)
        
        # Make random number arrays for parameters
        b_rand_int = np.random.randint(0,nstep,(numTrials*self.outDim*self.inDim))
        a_rand_int = np.random.randint(0,nstep,(numTrials*self.outDim*self.inDim))
        k_rand_int = np.random.randint(0,len(k_params_sampled),(numTrials*self.outDim*self.inDim))
        
        # Create parameter arrays and reshape
        self.b_real_params = np.array([b_params_sampled[i] for i in b_rand_int]).reshape((numTrials,self.outDim*self.inDim))
        self.a_real_params = np.array([a_params_sampled[i] for i in a_rand_int]).reshape((numTrials,self.outDim*self.inDim))
        self.k_real_params = np.array([k_params_sampled[i] for i in k_rand_int]).reshape((numTrials,self.outDim*self.inDim))
        
        
        if self.inDim>1:
            self.uArray = np.zeros((self.numTrials,self.nstep,self.inDim))
            for trial in range(0,numTrials):
                if self.not_prbs:
                    seq=self.random_process()
                else:
                    seq = self.PRBS()
                seq[-300:] = 0 
                for dim in range(0,self.inDim):
                    dimseq = np.zeros(self.nstep)
                    dimseq[dim*self.length:dim*self.length+self.length] = seq
                    self.uArray[trial,:,dim] = dimseq
        else:
            # Make uArray for all data
            self.uArray = np.zeros((self.numTrials,self.nstep,self.inDim))
            for trial in range(0,numTrials):
                if self.not_prbs:
                    seq=self.random_process()
                else:
                    seq = self.PRBS()
                for dim in range(0,self.inDim):
                    dimseq = np.zeros(self.nstep)
                    for i in range(0,int(self.nstep/self.length)):
                        dimseq[i*self.length:(i*self.length+self.length)]=seq
                    self.uArray[trial,:,dim] = dimseq
                    seq = -seq[::-1]
                
        
        # Iterate over each of the simulations and add
        # to simulation arrays
        iterator=range(numTrials)
        self.step=False
        if not(self.order):
            self.yArray = np.array(list(map(self.y_map_function,iterator)))
        elif self.order>1:
            #self.b_real_params = np.zeros((numTrials,self.outDim*self.inDim*self.order))
            self.a_real_params = np.zeros((numTrials,self.outDim*self.inDim*self.order))
            #self.k_real_params = np.array([k_params_sampled[i] for i in k_rand_int]).reshape((numTrials,self.outDim*self.inDim))
            self.ySteps = np.zeros((self.numTrials,self.nstep,self.inDim))
            self.yArray = np.array(list(map(self.y_map_higher_function,iterator)))
        else:
            self.ySteps = np.zeros((self.numTrials,self.nstep,self.inDim))
            self.yArray = np.array(list(map(self.y_map_higher_function,iterator)))
        # Colors for plotting input/output signals properly
        colors = ['midnightblue','gray','darkgreen','crimson','olive','navy','lightcoral','indigo','darkcyan',
                  'coral','darkorange','navy','r']
        
        
        # Only plot every 100 input signals
        for outit in range(self.numPlots):
            plt.figure(figsize=(10,5),dpi=200)
            plt.grid()
            for it in range(self.uArray.shape[-1]): 
                label1 = '$u_' + str(it+1) + '(t)$' 
                label2 = '$y_' + str(it+1) + '(t)$'
                plt.plot(self.t,self.uArray[outit,:,it],colors[it],label=label1)
                plt.plot(self.t,self.yArray[outit,:,it],colors[self.inDim+it],label=label2)
            plt.ylabel("Measured Signal (5% Noise)")
            plt.xlabel("Time Step (s)")
            plt.legend()
            plt.show()
        
        print(len(self.yArray))
        # Randomly pick training and validation indices 
        index = range(0,len(self.yArray)*self.inDim*self.outDim)
        if self.trainFrac!=1:  
            train = random.sample(index,int(trainFrac*len(self.yArray)*self.inDim*self.outDim))
            test = [item for item in list(index) if item not in train]
        else:
            train=range(0,len(self.yArray*self.inDim*self.outDim))
            test=[]
        
        
        # Make it so that any of these attributes can be accessed 
        # without needing to return them all from the function
        self.aVals = self.a_real_params
        self.bVals = self.b_real_params
        self.kVals = self.k_real_params
        self.orderList = orderList
        self.train = train
        self.test = test
        #if self.numPlots>0:
            #self.plot_parameter_space(self.a_real_params,self.b_real_params,self.k_real_params,train,test)
        
        return self.uArray,self.yArray,self.a_real_params,self.b_real_params,self.k_real_params,train,test
     
    def closed_loop_forced(self):
        nTotal = self.numTrials ; trainFrac = self.trainFrac
        
        directory = self.pd + '/MATLAB code/'
        csv_1 = directory + 'inputArray.csv'
        csv_2 = directory + 'outputArray.csv'
        inputData = np.transpose(np.genfromtxt(csv_1,delimiter=','))
        outputData = np.transpose(np.genfromtxt(csv_2,delimiter=','))
        
        a,b = inputData.shape
        self.uArray = inputData[:,1:].reshape(a,b-1,1)
        self.yArray = outputData[:,1:].reshape(a,b-1,1)
        self.a_real_params = np.linspace(0.01,0.99,nTotal).reshape(a,1)
        self.b_real_params = np.linspace(0.01,0.99,nTotal).reshape(a,1)
        self.k_real_params = np.array([i%9 for i in range(0,100000)]).reshape(a,1)
        
        # Randomly pick training and validation indices 
        index = range(0,len(self.yArray)*self.inDim*self.outDim)
        if self.trainFrac!=1:  
            train = random.sample(index,int(trainFrac*len(self.yArray)*self.inDim*self.outDim))
            test = [item for item in list(index) if item not in train]
        else:
            train=range(0,len(self.yArray*self.inDim*self.outDim))
            test=[]
 
        # Make it so that any of these attributes can be accessed 
        # without needing to return them all from the function
        self.aVals = self.a_real_params
        self.bVals = self.b_real_params
        self.kVals = self.k_real_params
        self.train = train
        self.test = test

        return self.uArray,self.yArray,self.a_real_params,self.b_real_params,self.k_real_params
    
    def preprocess(self,xData,yData,mutDim=1):
        """This function uses the training and testing indices produced during
        simulate() to segregate the training and validation sets"""
        # If array has more than 2 dimensions, use 
        # axis=2 when reshaping, otherwise set to 1
        try:
            numTrials,_,numDim= xData.shape
        except:
            numTrials,_ = xData.shape
            numDim=1
           
        # Select training and validation data based on training
        # and testing indices set during simulation
        trainspace = xData[self.train]
        valspace = xData[self.test] 
        
        x_train= trainspace.reshape((math.floor(numTrials*self.trainFrac),self.length,mutDim))    
        x_val = valspace.reshape((math.floor(numTrials*(1-self.trainFrac)),self.length,mutDim))
        
        try:
            y_val = np.array([yData[i,:] for i in self.test])
            y_train = np.array([yData[i,:] for i in self.train])
        except:
            y_val = np.array([yData[i] for i in self.test])
            y_train = np.array([yData[i] for i in self.train])
         
        if self.trim>0:
            return x_train[:,:-self.trim,:],x_val[:,:-self.trim,:],y_train,y_val,numDim   
        else:
            return x_train,x_val,y_train,y_val,numDim
    
    def stretch_MIMO(self,name):
        """Function that takes the input parameters and stacks into one 
        array, then processes so that data can be used for any size
        MIMO system. Not used if SISO system"""
        bVals=self.bVals; aVals=self.aVals; kVals=self.kVals
        uArray=self.uArray; yArray=self.yArray
        a,b,c = np.shape(self.uArray)
        self.xDataMat = np.full((a*self.outDim,b,c),0.)
        self.yDataMat = np.full((a*self.outDim,self.inDim),0.)
        if name=='b':
            for j in range(0,self.inDim):
                dim = self.inDim
                self.yDataMat[a*j:a*(j+1),:] = bVals[:,dim*j:dim*(j+1)]
            
            for i in range(0,self.outDim):
                for j in range(0,self.inDim):
                    self.xDataMat[a*i:a*(i+1),:,j] = yArray[:,:,i] * uArray[:,:,j]      
        elif name=='a':
            for j in range(0,self.inDim):
                dim = self.inDim
                self.yDataMat[a*j:a*(j+1),:] = aVals[:,dim*j:dim*(j+1)]

            for i in range(0,self.outDim):
                for j in range(0,self.inDim):
                    self.xDataMat[a*i:a*(i+1),:,j] = yArray[:,:,i] - uArray[:,:,j]
        else:
            for j in range(0,self.inDim):
                dim = self.inDim
                self.yDataMat[a*j:a*(j+1),:] = kVals[:,dim*j:dim*(j+1)]

            for i in range(0,self.outDim):
                for j in range(0,self.inDim):
                    self.xDataMat[a*i:a*(i+1),:,j] = yArray[:,:,i] - uArray[:,:,j]
        
        print(self.xDataMat.shape)
        return self.xDataMat,self.yDataMat
    
    def stretch_MIMO_multi(self,name):
        """Function that takes the input parameters and stacks into one 
        array, then processes so that data can be used for any size
        MIMO system. Not used if SISO system"""
        bVals=self.bVals; aVals=self.aVals; kVals=self.kVals
        uArray=self.uArray; yArray=self.yArray
        a,b,c = np.shape(self.uArray)
        self.xDataMat = np.full((a*self.outDim*self.inDim,self.length),0.)
        self.yDataMat = np.full((a*self.outDim*self.inDim,2),0.)
        k = self.length

        for i in range(0,self.outDim):
            for j in range(0,self.inDim):
                dim=self.inDim
                self.yDataMat[a*(dim*i+j):a*(dim*i+j+1),0] = aVals[:,(dim*i+j):(dim*i+j+1)][:,0]
                self.yDataMat[a*(dim*i+j):a*(dim*i+j+1),1] = bVals[:,(dim*i+j):(dim*i+j+1)][:,0]
        
        for i in range(0,self.outDim):
            for j in range(0,self.inDim):
                self.xDataMat[a*(self.inDim*i+j):a*(self.inDim*i+j+1),...] = yArray[:,j*k:(j+1)*k,i] - uArray[:,j*k:(j+1)*k,j]     

        return self.xDataMat,self.yDataMat  
    
    def stretch_MIMO_sequential(self,name):
        """Function that takes the input parameters and stacks into one 
        array, then processes so that data can be used for any size
        MIMO system. Not used if SISO system"""
        bVals=self.bVals; aVals=self.aVals; kVals=self.kVals
        uArray=self.uArray; yArray=self.yArray
        a,b,c = np.shape(self.uArray)
        self.xDataMat = np.full((a*self.outDim*self.inDim,self.length),0.)
        self.yDataMat = np.full((a*self.outDim*self.inDim),0.)
        k = self.length
        if name=='b':
            for i in range(0,self.outDim):
                for j in range(0,self.inDim):
                    dim=self.inDim
                    self.yDataMat[a*(dim*i+j):a*(dim*i+j+1)] = bVals[:,(dim*i+j):(dim*i+j+1)][:,0]
            
            for i in range(0,self.outDim):
                for j in range(0,self.inDim):
                    self.xDataMat[a*(self.inDim*i+j):a*(self.inDim*i+j+1),...] = yArray[:,j*k:(j+1)*k,i] - uArray[:,j*k:(j+1)*k,j]  
       
        elif name=='a':
            for i in range(0,self.outDim):
                for j in range(0,self.inDim):
                    dim=self.inDim
                    self.yDataMat[a*(dim*i+j):a*(dim*i+j+1)] = aVals[:,(dim*i+j):(dim*i+j+1)][:,0]

            for i in range(0,self.outDim):
                for j in range(0,self.inDim):
                    self.xDataMat[a*(self.inDim*i+j):a*(self.inDim*i+j+1),...] = yArray[:,j*k:(j+1)*k,i] - uArray[:,j*k:(j+1)*k,j]    
        else:
            for i in range(0,self.outDim):
                for j in range(0,self.inDim):
                    dim=self.inDim
                    self.yDataMat[a*(dim*i+j):a*(dim*i+j+1)] = kVals[:,(dim*i+j):(dim*i+j+1)][:,0]

            for i in range(0,self.outDim):
                for j in range(0,self.inDim):
                    self.xDataMat[a*(self.inDim*i+j):a*(self.inDim*i+j+1),...] = yArray[:,j*k:(j+1)*k,i] - uArray[:,j*k:(j+1)*k,j]    

        return self.xDataMat,self.yDataMat   
    
    def stretch_convolutional(self,name):
        """Function that takes the input parameters and stacks into one 
        array, then processes so that data can be used for any size
        MIMO system. Not used if SISO system"""
        bVals=self.bVals; aVals=self.aVals; kVals=self.kVals
        uArray=self.uArray; yArray=self.yArray
        a,b,c = np.shape(self.uArray)
        self.xDataMat = np.full((a*self.outDim*self.inDim,self.length,5),0.)
        self.yDataMat = np.full((a*self.outDim*self.inDim,2),0.)
        k = self.length
        if name=='b':
            for i in range(0,self.outDim):
                for j in range(0,self.inDim):
                    dim=self.inDim
                    self.yDataMat[a*(dim*i+j):a*(dim*i+j+1),0] = bVals[:,(dim*i+j):(dim*i+j+1)][:,0]
                    self.yDataMat[a*(dim*i+j):a*(dim*i+j+1),1] = aVals[:,(dim*i+j):(dim*i+j+1)][:,0]
            
            for i in range(0,self.outDim):
                for j in range(0,self.inDim):
                    self.xDataMat[a*(self.inDim*i+j):a*(self.inDim*i+j+1),:,0] = yArray[:,j*k:(j+1)*k,i]     
                    self.xDataMat[a*(self.inDim*i+j):a*(self.inDim*i+j+1),:,1] = uArray[:,j*k:(j+1)*k,j]
                    self.xDataMat[a*(self.inDim*i+j):a*(self.inDim*i+j+1),:,2] = yArray[:,j*k:(j+1)*k,i] * uArray[:,j*k:(j+1)*k,j]
                    self.xDataMat[a*(self.inDim*i+j):a*(self.inDim*i+j+1),:,3] = yArray[:,j*k:(j+1)*k,i] - uArray[:,j*k:(j+1)*k,j]
                    for zz in range(0,a):
                        self.xDataMat[zz,:,4] = np.convolve(yArray[zz,j*k:(j+1)*k,i],uArray[zz,j*k:(j+1)*k,j],'same')
       
        elif name=='a':
            for i in range(0,self.outDim):
                for j in range(0,self.inDim):
                    dim=self.inDim
                    self.yDataMat[a*(dim*i+j):a*(dim*i+j+1)] = aVals[:,(dim*i+j):(dim*i+j+1)][:,0]

            for i in range(0,self.outDim):
                for j in range(0,self.inDim):
                    self.xDataMat[a*(self.inDim*i+j):a*(self.inDim*i+j+1),:,0] = yArray[:,j*k:(j+1)*k,i]     
                    self.xDataMat[a*(self.inDim*i+j):a*(self.inDim*i+j+1),:,1] = uArray[:,j*k:(j+1)*k,j]
        else:
            for i in range(0,self.outDim):
                for j in range(0,self.inDim):
                    dim=self.inDim
                    self.yDataMat[a*(dim*i+j):a*(dim*i+j+1)] = kVals[:,(dim*i+j):(dim*i+j+1)][:,0]

            for i in range(0,self.outDim):
                for j in range(0,self.inDim):
                    self.xDataMat[a*(self.inDim*i+j):a*(self.inDim*i+j+1),:,0] = yArray[:,j*k:(j+1)*k,i]     
                    self.xDataMat[a*(self.inDim*i+j):a*(self.inDim*i+j+1),:,1] = uArray[:,j*k:(j+1)*k,j]   

        return self.xDataMat,self.yDataMat   
    
    
    def system_validation(self,b_possible_values=[.01,.99],a_possible_values=[.01,.99],
                          k_possible_values=[1,10],order=False,disturbance=False):
        """This function makes it easier to run a bunch of simulations and 
        automatically return the validation and testing sets without 
        calling each function separately. """
        # Since no training is occurring, can skip separation of testing and validation sets
        self.trainFrac = 1
        
        uArray,yArray,aVals,bVals,kVals,train,test = self.sys_simulation(stdev=self.stdev,
                    b_possible_values=b_possible_values,a_possible_values=a_possible_values,
                    k_possible_values=k_possible_values,order=order,disturbance=disturbance)
        
        a,b,c = np.shape(uArray)
        
        self.xData ={};
        self.yData={}
        self.names = ["b","a","k"]
        
        for (i,name) in enumerate(self.names):
            # Develop separate model for each output variable
            x,y = self.stretch_MIMO_sequential(name)
            
            a,b = x.shape
            x = x.reshape((a,b,1))
            
            self.xData[name] = x
            self.yData[name] = y
        
        return self.xData,self.yData 
    
    def system_validation_multi(self,b_possible_values=[.01,.99],
                                a_possible_values=[.01,.99],k_possible_values=[0,1],
                                order=False,disturbance=False,not_prbs=False):
        """This function makes it easier to run a bunch of simulations and 
        automatically return the validation and testing sets without 
        calling each function separately. """
        # Since no training is occurring, can skip separation of testing and validation sets
        self.trainFrac = 1
        
        uArray,yArray,aVals,bVals,kVals,train,test = self.sys_simulation(stdev=self.stdev,
                    b_possible_values=b_possible_values,a_possible_values=a_possible_values,
                    k_possible_values=k_possible_values,order=order,disturbance=disturbance,not_prbs=not_prbs)
        
        a,b,c = np.shape(uArray)
        
        self.xData ={};
        self.yData={}
        self.names = ["b","a"]
        
        for (i,name) in enumerate(self.names):
            # Develop separate model for each output variable
            x,y = self.stretch_MIMO_multi(name)
            
            a,b = x.shape
            x = x.reshape((a,b,1))
            
            self.xData[name] = x
            self.yData[name] = y
        
        return self.xData,self.yData 
    
    def closed_loop_validation(self,b_possible_values=[.01,.99],
                               a_possible_values=[.01,.99],k_possible_values=[1,10],order=False):
        """This function makes it easier to run a bunch of simulations and 
        automatically return the validation and testing sets without 
        calling each function separately. """
        # Since no training is occurring, can skip separation of testing and validation sets
        self.trainFrac = 1
        
        uArray,yArray,aVals,bVals,kVals = self.closed_loop_forced()
        
        a,b,c = np.shape(uArray)
        
        self.xData ={};
        self.yData={}
        self.names = ["b","a","k"]
        
        for (i,name) in enumerate(self.names):
            # Develop separate model for each output variable
            x,y = self.stretch_MIMO_sequential(name)
            
            a,b = x.shape
            x = x.reshape((a,b,1))
            
            self.xData[name] = x
            self.yData[name] = y
        
        return self.xData,self.yData 

 

    def y_map_higher_function(self,iterator):
        """This function simulates higher order systems in order to 
        validate the model as a predictor of higher order systems.
        Cannot be used to generate signals for training"""
        # If some combination of 20% done running, checkpoint to console  
        if iterator in self.milestones:
            self.serialized_checkpoint(iterator)
        
        # Create first PRBS outside of loop
        u = self.uArray[iterator,:,:]
        a,b = self.b_real_params.shape
        self.bFinal = np.zeros((a,b,self.order))
        self.aFinal = np.zeros((a,b,self.order+1))
        if not(hasattr(self, 'test_sequence')):
            self.test_sequence=self.PRBS()
        
        # The transfer function from the 2nd input to the 1st output is
        # (3s + 4) / (6s^2 + 5s + 4).
        # num = [[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]]
        # Iterate over each of the output dimensions and
        # add to numerator 
        allY=np.zeros((self.nstep,self.outDim))
        stepY=np.zeros((self.nstep,self.outDim))
        for j in range(0,self.outDim):
            # Iterate over each of the input dimensions
            # and add to the numerator array
            num=[];denom=[]
            for i in range(self.inDim): 
                b = self.b_real_params[iterator,self.outDim*j+i]
                a = self.a_real_params[iterator,self.outDim*j+i]
                numTemp=np.zeros(self.order);denTemp=np.zeros(self.order+1)
                denTemp[0]=1.
                for o in range(1,self.order+1):
                    if self.order==1:
                        numo = b
                        deno = -a
                        numTemp[0]=numo
                        denTemp[1]=deno
                    else: 
                        deno = np.random.uniform(0.2,self.a_possible_values[1]/2)
                        denTemp[o] = -deno
                        
                        while np.sum(abs(denTemp[1:]))>=1:
                            deno=deno/2
                            denTemp[o]=deno
                
                        self.a_real_params[iterator,self.inDim*self.order*j+i*self.order+o-1] = deno
                        
                    #denTemp.append(-deno)
                    #mean = abs(np.sum(denTemp[1:]))

                num.append([b])
                denom.append(denTemp)
            kVals = [self.k_real_params[iterator,self.outDim*j+i] for i in range(self.inDim)]
            
            bigU=np.transpose(u)
            uSim=np.zeros_like(bigU)
            for (idx,row) in enumerate(bigU):
                uSim[idx,:] = np.concatenate((np.zeros(kVals[idx]),row[:-kVals[idx]]))
            num=np.array([num]);denom=np.array([denom])
            sys = control.tf(num,denom,1)
            _,y,_ = control.forced_response(sys,U=uSim,T=self.t)
            _,sY,_ = control.forced_response(sys,U=self.test_sequence,T=np.linspace(0,self.length-1,self.length))
            
            allY[:,j] = self.gauss_noise(y,self.stdev)
            try:
                stepY[:,j] = self.gauss_noise(sY,self.STDEVS[-1])
            except:
                stepY[:,j] = self.gauss_noise(sY,self.stdev)
            for idx in range(stepY.shape[-1]):
                row = stepY[...,i]
                temp=np.zeros_like(row)
                temp[kVals[i]:]= row[:-kVals[i]]
                temp[:kVals[i]] = 0
                stepY[...,i] = temp
        self.ySteps[iterator,...]=stepY
        return allY
        