#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 16:25:30 2020

@author: RileyBallachay
"""
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import scipy.signal as signal
from mpl_toolkits.mplot3d import Axes3D
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
    
    def __init__(self,numTrials=100,nstep=100,timelength=100,trainFrac=0.7,numPlots=5,stdev=5):
        self.numTrials = numTrials
        self.nstep = nstep
        self.timelength = timelength
        self.trainFrac = trainFrac
        self.valFrac = 1-trainFrac
        self.numPlots = numPlots
        self.stdev=stdev
        self.special_value=-99
        self.startTime=time.time()
    
    
    def random_signal(self):
        """Module that simulates a wave function with 
        random frequency and amplitude within the specified range"""
        y = np.random.rand(self.nstep)
        y[:10] = 0
        y[-10:] = 0     
        windowlength = np.random.randint(5,20)
        win = signal.hann(windowlength)-0.5
        filtered = signal.convolve(y, win, mode='same') / sum(win)
        return filtered
    
    def PRSS(self,length,prob_switch=0.5):
        gbn = np.ones(length)
        gbn = gbn*random.choice([-1,1])
        magnitude = np.ones(length)
        for i in range(0,length-5,5):
            # For changing sign
            prob=np.random.random()
            gbn[i:i+5] = gbn[i]
            magnitude[i:i+5] = magnitude[i]
            if prob<prob_switch:
                mag = random.choice([-0.5,-1,-2,0.5,1,2])
                gbn[i:i+5] = gbn[i:i+5] + mag
                
        gbn=gbn*magnitude
        return np.array(gbn)
                
    
    def PRBS(self, prob_switch=0.1, Range=[-1.0, 1.0]):  
        """Returns a pseudo-random binary sequence 
        which ranges between -1 and +1"""
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
        gbn=gbn.reshape((len(gbn),1))
        return gbn
 
    def plot_parameter_space(self,x,y,trainID,valID,z=False):
        """This function plots the parameter space for a first 
        order plus time delay model in 3D coordinates"""
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
    
    def gauss_noise(self,array,stdev):
        """Generate gaussian noise with mean and standard deviation
        of 5% of the maximum returned value."""
        # If the array has 2 dimensions, this will capture it
        # Otherwise, it will evaluate the length of 1D array
        try:
            length,width = np.shape(array)
            noise = np.random.normal(0,(stdev/100)*np.amax(array),(length,width))
        except:
            length = len(array)
            noise = np.random.normal(0,(stdev/100)*np.amax(array),(length,))
        return array+noise
    
    def get_xData(uArray,yArray):
        """ Need an isolated and mutable method for 
        preprocessing SISO xData that can be updated and 
        used to preprocess training and prediction data"""
        def preprocess_theta(ySlice):
            amax = np.amax(abs(ySlice))
            ySlice = ySlice/amax
            return ySlice
        
        amax = np.amax(abs(uArray))
        uArray = uArray/amax
        yArray = yArray/amax
        
        thetaData1 = np.apply_along_axis(preprocess_theta, 1, yArray)
        thetaData2 = np.apply_along_axis(preprocess_theta, 1, uArray)
        thetaData = thetaData1-thetaData2
        xDatas = [yArray,yArray-uArray,thetaData]
        return xDatas
    
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
    
    def SISO_simulation(self,KpRange=[1,10],tauRange=[1,10],thetaRange=[1,10]):
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
        # Access all the attributes from initialization
        numTrials=self.numTrials; nstep=self.nstep;
        timelength=self.timelength; trainFrac=self.trainFrac
        
        milestones = []
        # Loop to create milestones to checkpoint data creation
        for it in [2,4,6,8]:
            milestones.append(int((it/10)*numTrials))
        
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
            # If some combination of 10% done running, checkpoint to console          
            if iterator in milestones:
                self.serialized_checkpoint(iterator)
                
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
            u = (self.PRBS()[:,0])  
            t = np.linspace(0,nstep,nstep)
            
            # Subtract time delay and get the 'simulated time' which has
            # no physical significance. Fill the delay with zeros and
            # start signal after delay is elapsed
            uInclude = np.concatenate((np.zeros(theta),u[:-theta]))
            
            # Use transfer function module from control to simulate 
            # system response after delay then add to zeros
            sys = control.tf([Kp,],[taup,1.])
            _,yEnd,_ = control.forced_response(sys,U=uInclude,T=t)
            y = self.gauss_noise(yEnd,self.stdev)
           
   
            uArray[iterator,:] = u
            yArray[iterator,:]= y
            taus.append(taup)
            thetas.append(theta)
            kps.append(Kp)
            
            # Only plot every 100 input signals
            if (iterator)<self.numPlots:
                plt.figure(dpi=100)
                plt.plot(t,u,label='Input Signal')
                plt.plot(t,y, label='FOPTD Response')
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
            train = np.sort(random.sample(index,int(trainFrac*numTrials)))
            test = np.sort(np.array([item for item in list(index) if item not in train]))
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
        self.numTrials = numTrials
        
        return uArray,yArray,taus,kps,thetas,train,test
    
    
    def MIMO_simulation(self,stdev=5,inDim=2,outDim=2,KpRange=[1,10],tauRange=[1,10],thetaRange=[1,10]):
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
        # Access all the attributes from initialization
        numTrials=self.numTrials; nstep=self.nstep;
        timelength=self.timelength; trainFrac=self.trainFrac
        self.inDim = inDim; self.outDim = outDim
        
        milestones = []
        # Loop to create milestones to checkpoint data creation
        for it in [2,4,6,8]:
            milestones.append(int((it/10)*numTrials))
            
        # Set the type of the simulation to inform the data split
        self.type = "MIMO"
        
        # Initialize the arrays which will store the simulation data
        uArray = np.full((numTrials,nstep,inDim),0.)
        yArray = np.full((numTrials,nstep,outDim),0.)
        KpArray = np.full((numTrials,outDim*inDim),0.)
        tauArray = np.full((numTrials,outDim*inDim),0.)
        thetaArray = np.full((numTrials,outDim*inDim),0.)
        orderList = []
        
        # Make arrays containing parameters tau, theta
        KpSpace = np.linspace(KpRange[0],KpRange[1],nstep)
        taupSpace = np.linspace(tauRange[0],tauRange[1],nstep)
        thetaSpace = np.arange(thetaRange[0],thetaRange[1])
        t = np.linspace(0,timelength,nstep)
        
        # Iterate over each of the simulations and add
        # to simulation arrays
        iterator=0
        while(iterator<numTrials):
            # If some combination of 10% done running, checkpoint to console          
            if iterator in milestones:
                self.serialized_checkpoint(iterator)
            
            # Create first PRBS outside of loop
            u = self.PRBS(prob_switch=0.1/inDim)
            
            # Run a new PRBS for each input
            # variable and stack in input
            for i in range(1,inDim):
                prbs = self.PRBS(prob_switch=0.1/inDim)
                u = np.concatenate((u,prbs),axis=1)
            
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
                thetas = []
                # Iterate over each of the input dimensions
                # and add to the numerator array
                for i in range(0,self.inDim):
                    if len(orderList)<(outDim*inDim):
                        string = "Input # " + str(i+1) + " Output # " + str(j+1)
                        orderList.append(string)
                    index = np.random.randint(0,nstep)
                    index2 = np.random.randint(0,nstep)
                    index3 = np.random.randint(0,9)
                    KpArray[iterator,(self.outDim*j)+i] = KpSpace[index]
                    numTemp.append([KpSpace[index]])
                    tauArray[iterator,(self.outDim*j)+i] = taupSpace[index2]
                    denTemp.append([taupSpace[index2],1.])
                    thetaArray[iterator,(self.outDim*j)+i] = thetaSpace[index3]
                    thetas.append(thetaSpace[index3])
                #nums.append(numTemp)
                #dens.append(denTemp)
                
                bigU=np.transpose(u)
                uSim=np.zeros_like(bigU)
                for (idx,row) in enumerate(bigU):
                    uSim[idx,:] = np.concatenate((np.zeros(thetas[idx]),row[:-thetas[idx]]))
                numTemp=np.array([numTemp]);denTemp=np.array([denTemp])
                sys = control.tf(numTemp,denTemp)
                _,y,_ = control.forced_response(sys,U=uSim,T=t)
                y=self.gauss_noise(y,stdev).reshape((len(y),1))
                
                if j==0:
                    allY = y
                else:
                    allY = np.concatenate((allY,y),axis=1)
                    
            
              
            # Use transfer function class to simulate system response
            # to MIMO input and randomized parameters 
            #nums=np.array(nums)
            #dens=np.array(dens)
            #sys = control.tf(nums,dens)
            #_,y,_ = control.forced_response(sys,U=np.transpose(u),T=t)
            #y = np.transpose(self.gauss_noise(y,stdev))
            yArray[iterator,:,:] = allY
             
            # Only plot every 100 input signals
            if (iterator)<self.numPlots:
                plt.figure(dpi=100)
                plt.plot(t,u,label='Input Signal')
                plt.plot(t,allY, label='FOPTD Response')
                if self.inDim<3:
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
     
    
    def preprocess(self,xData,yData):
        """This function uses the training and testing indices produced during
        simulate() to segregate the training and validation sets"""
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
        
        x_train= trainspace.reshape((math.floor(self.numTrials*self.trainFrac),self.nstep,numDim))    
        x_val = valspace.reshape((math.floor(self.numTrials*(1-self.trainFrac)),self.nstep,numDim))
        
        if self.type=="MIMO":
            y_val = np.array([yData[i,:] for i in self.test])
            y_train = np.array([yData[i,:] for i in self.train])
        else:
            y_val = np.array([yData[i] for i in self.test])
            y_train = np.array([yData[i] for i in self.train])
            
        return x_train,x_val,y_train,y_val,numDim
    
    
    def SISO_validation(self,KpRange=[1,10],tauRange=[1,10],thetaRange=[1,10]):
        """This function makes it easier to run a bunch of simulations and 
        automatically return the validation and testing sets without 
        calling each function separately. """
        # Since no training is occurring, can skip separation of testing and validation sets
        self.trainFrac = 1
        
        uArray,yArray,taus,kps,thetas,train,test = self.SISO_simulation(KpRange,tauRange,thetaRange)
        xDatas = Signal.get_xData(uArray,yArray)
        yDatas = [taus, kps, thetas]
        
        self.xData ={};
        self.yData={}
        self.names = ["kp","tau","theta"]
        
        for (i,name) in enumerate(self.names):
            x,_,y,_,_ = self.preprocess(xDatas[i],yDatas[i])
            self.xData[name] = x
            self.yData[name] = y
        
        return self.xData,self.yData
    
    
    def stretch_MIMO(self,name):
        """Function that takes the input parameters and stacks into one 
        array, then processes so that data can be used for any size
        MIMO system. Hasn't been validated on greater that 2x2"""
        kps=self.kps; taus=self.taus
        uArray=self.uArray; yArray=self.yArray
        a,b,c = np.shape(self.uArray)
        self.xDataMat = np.full((a*self.outDim,b,c),0.)
        self.yDataMat = np.full((a*self.outDim,self.inDim),0.)
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
        print(self.xDataMat)
        return self.xDataMat,self.yDataMat
    
    
    def MIMO_validation(self):
        """This function makes it easier to run a bunch of simulations and 
        automatically return the validation and testing sets without 
        calling each function separately. """
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
        