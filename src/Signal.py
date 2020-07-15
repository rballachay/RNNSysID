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
import control as control
import control.matlab as matlab

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
        Produces 3D plot of all simulated parameters (tau,kp,theta)
    
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
    
    # Going to need this to be interactive 
    # i.e. the function waits until the 
    def __init__(self,numTrials=100,nstep=100,timelength=100,trainFrac=0.7,numPlots=5,stdev=5):
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
    def plot_parameter_space(self,x,y,trainID,valID,z=False):
        if z:
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
        else:
            x = np.array(x.ravel()); y = np.array(y.ravel())
            plt.figure(dpi=200)
            plt.plot(x,y,'.b')
            plt.ylabel("τ (Time Constant)")
            plt.xlabel("Kp (Gain)")
    
    # Generate gaussian noise with mean and standard deviation
    # of 5% of the maximum returned value. 
    def gauss_noise(self,array,stdev):
        # If the array has 2 dimensions, this will capture it
        # Otherwise, it will evaluate the length of 1D array
        try:
            length,width = np.shape(array)
            noise = np.random.normal(0,(stdev/100)*np.amax(array),(length,width))
        except:
            length = len(array)
            noise = np.random.normal(0,(stdev/100)*np.amax(array),(length,))
        return array+noise
    
    """
    Module which produces simulation of SISO system given the input parameters. 
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
    KpRange : tuple, default=(1,10)
        Possible range for gains. An equally spaced array between the maximum 
        and minimum are chosen based on the number of simulations.
    
    tauRange : tuple, default=(1,10)
        Possible range for time constant. An equally spaced array between the 
        maximum and minimum are chosen based on the number of simulations.
     
    thetaRange : tuple, default=(1,10)
        Possible range for time delays. An equally spaced array between the 
        maximum and minimum are chosen based on the number of simulations.
        
    """
    def SISO_simulation(self,KpRange=[1,10],tauRange=[1,10],thetaRange=[1,10],stdev=5):
        # Access all the attributes from initialization
        numTrials=self.numTrials; nstep=self.nstep;
        timelength=self.timelength; trainFrac=self.trainFrac
        
        # Set the type of the simulation to inform the data split
        self.type = "SISO"
        
        # Initialize the arrays which will store the simulation data
        uArray = np.full((numTrials,nstep),0.)
        yArray = np.full((numTrials,nstep),0.)
        
        # Make arrays containing parameters tau, theta
        KpSpace = np.linspace(KpRange[0],KpRange[1],nstep)
        taupSpace = np.linspace(tauRange[0],tauRange[1],nstep)
        thetaSpace = np.arange(thetaRange[0],thetaRange[1])
        
        # Zeros wouldn't be beneficial for solving the 
        # system so don't use
        KpSpace[KpSpace==0] = 0.01
        taupSpace[taupSpace==0] = 0.01
        thetaSpace[thetaSpace==0] = 0.01
        
        # Empty lists for filling with parameters
        taus = []
        thetas=[]
        kps=[]
        
        # While loop which iterates over each of the parameter scenarios
        iterator=0
        while(iterator<numTrials):
            # Randomize each index so they aren't linearly dependent 
            # on each other
            index = np.random.randint(0,nstep)
            index1 = np.random.randint(0,nstep)
            index2 = np.random.randint(0,9) 
            
            # Select parameter using random index
            Kp = KpSpace[index]
            taup = taupSpace[index1]
            theta = thetaSpace[index2]
            
            # Generate random signal using
            u = (self.PRBS())  
            t = np.linspace(0,timelength,nstep)
            
            # Subtract time delay and get the 'simulated time' which has
            # no physical significance. Fill the delay with zeros and
            # start signal after delay is elapsed
            tsim = t - theta
            yindStart = next((i for i, x in enumerate(tsim) if x>0), None)
            tInclude = tsim[yindStart-1:]
            uInclude = u[yindStart-1:]
            
            # Use transfer function module from control to simulate 
            # system response after delay then add to zeros
            sys = control.tf([Kp,] ,[taup,1.])
            _,yEnd,_ = control.forced_response(sys,U=uInclude,T=tInclude)
            yEnd = self.gauss_noise(yEnd,stdev)
            y = np.concatenate((np.zeros((len(t)-len(tInclude))),yEnd))
            
            # Add simulation to array with all simulations
            uArray[iterator,:] = u
            yArray[iterator,:]= y
            taus.append(taup)
            thetas.append(theta)
            kps.append(Kp)

            # Only plot every 100 input signals
            if (iterator)<self.numPlots:
                plt.figure(dpi=100)
                plt.plot(t[:200],u[:200],label='Input Signal')
                plt.plot(t[:200],y[:200], label='FOPTD Response')
                plt.xlabel((theta))
                plt.legend()
                plt.show()
                
            # Subsequently update the iterator to move down row
            iterator+=1
        
        # Randomly select train and test indices from sample data. 
        # If the prediction module is used, trainFrac will default 
        # to one and this portion will be skipped
        index = range(0,len(yArray))
        if self.trainFrac!=1:  
            train = random.sample(index,int(trainFrac*numTrials))
            test = [item for item in list(index) if item not in train]
        else:
            train=index
            test=[]
        
        # Make it so that any of these attributes can be accessed 
        # without needing to return them all from the function
        self.plot_parameter_space(taus,kps,train,test,thetas)
        self.uArray = uArray
        self.yArray = yArray
        self.taus = taus
        self.kps = kps
        self.thetas = thetas
        self.train = train
        self.test = test
        
        return uArray,yArray,taus,kps,thetas,train,test
    
    """
    Module which produces simulation of MIMO system given the input parameters. 
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
        
    KpRange : tuple, default=(1,10)
        Possible range for gains. An equally spaced array between the maximum 
        and minimum are chosen based on the number of simulations.
    
    tauRange : tuple, default=(1,10)
        Possible range for time constant. An equally spaced array between the 
        maximum and minimum are chosen based on the number of simulations.
        
    """
    def MIMO_simulation(self,stdev=5,inDim=2,outDim=2,KpRange=[1,10],tauRange=[1,10]):
        # Access all the attributes from initialization
        numTrials=self.numTrials; nstep=self.nstep;
        timelength=self.timelength; trainFrac=self.trainFrac
        self.inDim = inDim; self.outDim = outDim
        
        # Set the type of the simulation to inform the data split
        self.type = "MIMO"
        
        # Initialize the arrays which will store the simulation data
        uArray = np.full((numTrials,nstep,inDim),0.)
        yArray = np.full((numTrials,nstep,outDim),0.)
        KpArray = np.full((numTrials,outDim*inDim),0.)
        tauArray = np.full((numTrials,outDim*inDim),0.)
        orderList = []
        
        # Make arrays containing parameters tau, theta
        KpSpace = np.linspace(KpRange[0],KpRange[1],nstep)
        taupSpace = np.linspace(tauRange[0],tauRange[1],nstep)
        t = np.linspace(0,timelength,nstep)
        
        # Iterate over each of the simulations and add
        # to simulation arrays
        iterator=0
        while(iterator<numTrials):
            u = self.PRBS(prob_switch=0.05)
            
            # Run a new PRBS for each input
            # variable and stack in input
            for i in range(1,inDim):
                prbs = self.PRBS(prob_switch=0.05)
                u = np.stack((u,prbs),axis=1)
            
            uArray[iterator,:,:] = u
            nums = []
            dens = []
            
            # The transfer function from the 2nd input to the 1st output is
            # (3s + 4) / (6s^2 + 5s + 4).
            # num = [[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]]
            # Iterate over each of the output dimensions and
            # add to numerator 
            for j in range(0,self.outDim):
                numTemp = []
                denTemp = []
                # Iterate over each of the input dimensions
                # and add to the numerator array
                for i in range(0,self.inDim):
                    if len(orderList)<(outDim*inDim):
                        string = "Input # " + str(i+1) + " Output # " + str(j+1)
                        orderList.append(string)
                    index = np.random.randint(0,nstep)
                    index2 = np.random.randint(0,nstep)
                    KpArray[iterator,(2*j)+i] = KpSpace[index]
                    numTemp.append([KpSpace[index]])
                    tauArray[iterator,(2*j)+i] = taupSpace[index2]
                    denTemp.append([taupSpace[index2],1.])
                nums.append(numTemp)
                dens.append(denTemp)
              
            # Use transfer function class to simulate system response
            # to MIMO input and randomized parameters 
            nums=np.array(nums)
            dens=np.array(dens)
            sys = control.tf(nums,dens)
            _,y,_ = control.forced_response(sys,U=np.transpose(u),T=t)
            y = np.transpose(self.gauss_noise(y,stdev))
            yArray[iterator,:,:] = y
             
            # Only plot every 100 input signals
            if (iterator)<self.numPlots:
                plt.figure(dpi=100)
                plt.plot(t[:200],u[:200],label='Input Signal')
                plt.plot(t[:200],y[:200], label='FOPTD Response')
                plt.legend()
                plt.show()
                
            # Subsequently update the iterator to move down row
            iterator+=1
        
        # Randomly pick training and validation indices 
        index = range(0,len(yArray))
        if self.trainFrac!=1:  
            train = random.sample(index,int(trainFrac*numTrials))
            test = [item for item in list(index) if item not in train]
        else:
            train=range(0,len(yArray))
            test=[]
        
        # Make it so that any of these attributes can be accessed 
        # without needing to return them all from the function
        self.uArray = uArray
        self.yArray = yArray
        self.taus = tauArray
        self.kps = KpArray
        self.orderList = orderList
        self.train = train
        self.test = test
        self.plot_parameter_space(tauArray,KpArray,train,test)
        
        return uArray,yArray,tauArray,KpArray,train,test
            
    # This function uses the training and testing indices produced during
    # simulate() to segregate the training and validation sets
    def preprocess(self,xData,yData):
        # If array has more than 2 dimensions, use 
        # axis=2 when reshaping, otherwise set to 1
        try:
            _,_,numDim= xData.shape
        except:
            numDim=1
           
        # Select training and validation data based on training
        # and testing indices set during simulation
        trainspace = xData[self.train]
        valspace = xData[self.test] 
        
        x_train= trainspace.reshape((int(self.numTrials*self.trainFrac),self.nstep,numDim))    
        x_val = valspace.reshape((int(self.numTrials*(1-self.trainFrac)),self.nstep,numDim))
        
        if self.type=="MIMO":
            y_val = np.array([yData[i,:] for i in self.test])
            y_train = np.array([yData[i,:] for i in self.train])
        else:
            y_val = [yData[i] for i in self.test]
            y_train = [yData[i] for i in self.train]
            
        return x_train,x_val,y_train,y_val,numDim
    
    
    # This function makes it easier to run a bunch of simulations and 
    # automatically return the validation and testing sets without 
    # calling each function separately. 
    def SISO_validation(self):
        # Since no training is occurring, can skip separation of testing and validation sets
        self.trainFrac = 1
        
        uArray,yArray,taus,kps,thetas,train,test = self.SISO_simulation(stdev=self.stdev)
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
    
    # Function that takes the input parameters and stacks into one 
    # array, then processes so that data can be used for any size
    # MIMO system. Hasn't been validated on greater that 2x2
    def stretch_MIMO(self,name):
        kps=self.kps; taus=self.taus
        uArray=self.uArray; yArray=self.yArray
        a,b,c = np.shape(self.uArray)
        self.xDataMat = np.full((a*self.outDim,b,c),0.)
        self.yDataMat = np.full((a*self.outDim,2),0.)
        if name=='kp':
            for j in range(0,self.inDim):
                dim = self.inDim
                self.yDataMat[a*j:a*(j+1),:] = kps[:,dim*j:dim*(j+1)]
            
            for i in range(0,self.outDim):
                for j in range(0,self.inDim):
                    self.xDataMat[a*i:a*(i+1),:,j] = uArray[:,:,j]*yArray[:,:,i]
        else:
            for j in range(0,self.inDim):
                dim = self.inDim
                self.yDataMat[a*j:a*(j+1),:] = taus[:,dim*j:dim*(j+1)]

            for i in range(0,self.outDim):
                for j in range(0,self.inDim):
                    self.xDataMat[a*i:a*(i+1),:,j] = yArray[:,:,i] - uArray[:,:,j]
        
        return self.xDataMat,self.yDataMat
    
    # This function makes it easier to run a bunch of simulations and 
    # automatically return the validation and testing sets without 
    # calling each function separately. 
    def MIMO_validation(self):
        # Since no training is occurring, can skip separation of testing and validation sets
        self.trainFrac = 1
        
        uArray,yArray,taus,kps,train,test = self.MIMO_simulation(stdev=self.stdev)
        a,b,c = np.shape(uArray)
        
        self.xData ={};
        self.yData={}
        self.names = ["kp","tau"]
        
        for (i,name) in enumerate(self.names):
            # Develop separate model for each output variable
            x,y = self.stretch_MIMO(name)
            
            self.xData[name] = x
            self.yData[name] = y
        
        return self.xData,self.yData       
        