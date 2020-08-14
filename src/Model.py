#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 17:32:17 2020

@author: RileyBallachay
"""
import os
from os import path
from pathlib import Path
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras import layers
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from tkinter import Tk
from tkinter.filedialog import askdirectory
import matplotlib.pyplot as plt
from Signal import Signal
from tensorflow.keras import backend as K
from tensorflow.keras import losses
import datetime
import control as control

class MyThresholdCallback(tf.keras.callbacks.Callback):
    """
    Callback class to stop training model when a target
    coefficient of determination is achieved on validation set.
    """
    
    def __init__(self, threshold):
        super(MyThresholdCallback, self).__init__()
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None): 
        val_acc = logs["val_coeff_determination"]
        if val_acc >= self.threshold:
            self.model.stop_training = True
            
class Model:    
    """
    Class for fitting and predicting first order state space
    models. All models are built to be used on pseudo-binary 
    (aka step-input) data.
    
    Builds models using keras with tensorflow backend. Currently
    only works for SISO systems. Will work with MIMO in the future.
    
    The Purpose of this class is to fit theta, tau and kp for first
    order systems using data simulated in the Signal class. Models are
    stored using the h5 format and can be used to estimate accuracy. 
    
    Parameters
    ----------
    Modeltype : string, default='regular'
        Two types of models exist: regular and probability. Probability 
        model uses lamba function embedded at the final layer of keras
        network to estimate mean and standard deviation for regression. 
        Regular model is built using keras Dense output and probability
        is estimated based on uncertainty in training data with a set 
        gaussian noise with standard deviation stdev, set as a parameter
        in the call to Signal. 
        
    nstep : int, default=100
        Parameter shared with the signal class. Will inherit in the future.
    
    """

    def __init__(self,nstep=100,Modeltype='probability'):   
        self.Modeltype = Modeltype
        self.nstep=nstep
        self.names = ["kp","tau","theta"]
        self.modelDict = {}
        self.special_value = -99
        self.cwd = str(os.getcwd())
        self.pd = str(Path(os.getcwd()).parent)
    
    def load_Model(self,sig):
        """Loads one of two first order models: probability or regular. Iterates 
        over directory and loads alphabetically. If more than 3 models exist in the 
        directory, it will load them indiscriminately."""
        modelList = []
        loadDir =  askdirectory(title='Select Folder With Trained Model')
        for filename in os.listdir(loadDir):
            if filename=='.DS_Store':
                continue
            print(filename)
            negloglik = lambda y, rv_y: -rv_y.log_prob(y[:])
            losses.custom_loss = negloglik
            dependencies = {'coeff_determination': self.coeff_determination,'loss':negloglik}
            model = tf.keras.models.load_model(loadDir+filename, custom_objects=dependencies,compile = False)
            model.compile(optimizer='adam', loss=negloglik,metrics=[self.coeff_determination])
            modelList.append(model)
        for i in range(0,3):
            self.modelDict[self.names[i]] = modelList[i]
     
        
    def load_and_train(self,sig,epochs=50,probabilistic=True,saveModel=True):
        """Calls load_SISO and continues training model, saving the updated 
        model to a new folder to avoid overwriting"""
        self.train_SISO(sig,newModel=False,probabilistic=probabilistic,saveModel=saveModel,epochs=epochs)

    
    def train_model(self,sig=False,plotLoss=True,plotVal=True,epochs=50,saveModel=True,
                    checkpoint=True,batchSize=16):
        """
        Takes Signal object with input and output data, separates data in training
        and validation sets, transform data for neural network and uses training 
        set to produce neural network of specified architecture. Works for MIMO.
        
        Separate models are produced for each output variable. Systems are 
        considered to behave as linear combinations of linear systems.
            Kp: System response (y) and kp used to construct model
            tau: System response (y) and tau used to construct model
        
        Parameters
        ----------
        sig: Signal (object), default=False
            Must provide instance of signal class in order to build model or use
            for prediction. Else, will fail to train. 
        
        plotLoss: bool, default=True
            Plot training and validation loss after training finishes. Saves to 
            a folder with the date and time if saveModel is set to true.
        
        plotVal: bool, default=True
            Plots validation data with coefficient of determination after training
            model. Saves to a folder with the date and time if saveModel is 
            set to true.
            
        probabilistic: bool, default=True
            Choice between probabilistic model (calls FOPTD_probabilistic) and 
            regular (FOPTD). If regular, prediction will include uncertainties
            built in with the validation set.   
            
        saveModel: bool, default=True
            Decide whether or not to save the models from training 
        """
        parameters = ['kp','tau','theta']
        
        # You have to construct a signal with all the necessary parameters before 
        if not(sig):
            print("Please initialize the class signal with model parameters first")
            return 
        
        # If the loss and accuracy plots are gonna be saved, saveModel must = True
        parentDir = self.pd+"/Models/"
        time = str(datetime.datetime.now())[:16]
        plotDir = parentDir + time + '/'
        checkptDir = plotDir + 'Checkpoints/'
        os.mkdir(plotDir)
        os.mkdir(checkptDir)
        
        # iterate over each of the parameters, train the model, save to the model
        # path and plot loss and validation curves
        modelList = []
        
        # Iterate over tau and kp parameters
        for (k,parameter) in enumerate(parameters):
            xData,yData = sig.stretch_MIMO(parameter)
            x_train,x_val,y_train,y_val,numDim = sig.preprocess(xData,yData)
            self.outDim = sig.outDim;self.inDim=sig.inDim
            
            # Check the dimension of data to ensure that an architecture 
            # exists for the shape of the data. If not, then it will 
            # prompt the user to make a new architecture
            model = self.mutable_model(x_train,y_train)
                
            print(model.summary())
            
            # Set threshold to stop training when coefficient of determinatio gets to 0.9. 
            # Section for including callbacks (custom and checkpoint)
            checkpoint_path = checkptDir + self.names[k] + '.cptk'
            cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                        save_weights_only=True,save_best_only=True,verbose=1) 
            MTC = MyThresholdCallback(0.95)
            
            print("Fit model on training data")
            history = model.fit(
                x_train,
                y_train,
                batch_size=batchSize,
                epochs=epochs,
                # We pass some validation for
                # monitoring validation loss and metrics
                # at the end of each epoch
                validation_data=(x_val, y_val),
                callbacks=[MTC,cp_callback]
            )
            
            # Save the training and validation loss to a text file
            numpy_loss_history = np.array(history.history['loss'])
            numpy_val_history = np.array(history.history['val_loss'])
            np.savetxt(plotDir+parameter+'_loss.txt', numpy_loss_history, delimiter=",")
            np.savetxt(plotDir+parameter+'_val_loss.txt', numpy_val_history, delimiter=",")
            
            # Store each model in dictionary and list
            modelList.append(model)
            self.modelDict[self.names[k]] = model
            yhat = model(x_val)
            predictions = np.array(yhat.mean())
            stddevs = np.array(yhat.stddev())

            # Only save model if true
            if saveModel:
                modelpath = plotDir+self.names[k]
                fileOverwriter=0
                while os.path.isfile(modelpath):
                    modelpath = plotDir+self.names[k]+"_"+str(fileOverwriter)
                    fileOverwriter+=1
                model.save(modelpath)
            
            # Plot learning curve for each parameter
            if plotLoss:
                plt.figure(dpi=200)
                plt.plot(history.history['loss'])
                plt.plot(history.history['val_loss'])
                plt.title('model loss for '+ parameter)
                plt.ylabel('loss')
                plt.xlabel('epoch')
                plt.legend(['train', 'test'], loc='upper left')
                if saveModel:
                    plt.savefig(plotDir+parameter+'_loss'+'.png')
                plt.show()
            
            # Plot predicted and actual value for each paramter and
            # coefficient of determination
            if plotVal:
                fig, axes = plt.subplots(1, sig.inDim,dpi=200) 
                fig.add_subplot(111, frameon=False)
                plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
                
                for (i,ax) in enumerate(axes):
                    ax.plot(y_val[:,i],predictions[0,:,i],'.b',label=parameter)
                    print(stddevs.shape)
                    r2 =("r\u00b2 = %.3f" % r2_score(y_val[:,i],predictions[0,:,i]))
                    ax.errorbar(y_val[:,i],predictions[0,:,i],yerr=stddevs[0,:,i]*2,fmt='none',ecolor='green')
                    ax.plot(np.linspace(0,10),np.linspace(0,10),'--r',label = r2)
                    ax.legend(prop={'size': 8})
                
                plt.ylabel("Predicted Value of "+ parameter)
                plt.xlabel("True Value of "+ parameter)

                if saveModel:
                    plt.savefig(plotDir+self.names[k]+'.png')
                plt.show()  
        
        return  
 
    
    def MSE(self,y_true,y_pred):
        """Calculate mean squared error of 
        predicted and actual system response."""
        mse = sum((y_true-y_pred)**2)
        return mse
    
    
    def predict_SISO(self,sig,plotPredict=True,savePredict=False,probabilityCall=False,notSignal=False):
        """
        Takes Signal object with input and output data, uses stored model
        to predict parameters based off simulated data, then plots
        system output based on predicted parameters.
        
        Plots response using predicted parameters along with real response,
        as well as real and predicted parameters.
        
        Parameters
        ----------
        sig : Signal (object), default=False
            Must provide instance of signal class in order to build model or use
            for prediction. Else, will fail to train. 
        
        plotPredict : bool, default=True
            Plots predicted response with real response.
        
        savePredict : bool,default=False
            Determine whether or not to save predicted responses.
            
        probabilityCall : bool, default=False
            Included to avoid building probability object while trying to 
            build probability object, leading to endless loop.
        """
        if not(isinstance(sig,Signal)):
            print("You need to predict with an instance of signal!")
            return
        
        # Predict errors and standard deviations if using probability model
        if self.Modeltype=='probability':
            
            self.kpPredictions = np.array(self.modelDict['kp'](sig.xData['kp']).mean())
            self.tauPredictions = np.array(self.modelDict['tau'](sig.xData['tau']).mean())
            self.thetaPredictions = np.array(self.modelDict['theta'](sig.xData['theta']).mean())
            
            self.kpError = np.array(2*self.modelDict['kp'](sig.xData['kp']).stddev())
            self.tauError = np.array(2*self.modelDict['tau'](sig.xData['tau']).stddev())
            self.thetaError = np.array(2*self.modelDict['theta'](sig.xData['theta']).stddev())           
          
        # Otherwise, solve for parameters using regular model
        else:
            if not(probabilityCall):
                prob = Probability(sig)
                Kperror,tauperror,thetaerror = prob.get_errors()
        
            self.kpPredictions = self.modelDict['kp'].predict(sig.xData['kp'])
            self.tauPredictions = self.modelDict['tau'].predict(sig.xData['tau'])
            self.thetaPredictions = self.modelDict['theta'].predict(sig.xData['theta'])
            
        self.errors = []
        uArrays = sig.uArray[sig.train,:]
        yArrays = sig.yArray[sig.train,:]
        
        for (i,index) in enumerate(sig.train):
            taup = self.tauPredictions[i][0]
            Kp = self.kpPredictions[i][0]
            theta = self.thetaPredictions[i][0]
            
            if self.Modeltype=='probability':
                Kperror=self.kpError[i]
                tauperror=self.tauError[i]
                thetaerror=self.thetaError[i]
                
            # Generate random signal using
            u = uArrays[i,:] 
            t = np.linspace(0,sig.timelength,sig.nstep)
            
            # Subtract time delay and get the 'simulated time' which has
            # no physical significance. Fill the delay with zeros and
            # start signal after delay is elapsed
            tsim = t - theta
            yindStart = next((i for i, x in enumerate(tsim) if x>0), None)
            tInclude = tsim[yindStart-1:]
            uInclude = u[:len(u)-(yindStart-1)]
            
            # Use transfer function module from control to simulate 
            # system response after delay then add to zeros
            sys = control.tf([Kp,] ,[taup,1.])
            _,yEnd,_ = control.forced_response(sys,U=uInclude,T=tInclude)
            yPred = np.concatenate((np.zeros((len(t)-len(tInclude))),yEnd))
            yTrue = yArrays[i,:]
            self.errors.append(self.MSE(yPred,yTrue))
            
            if plotPredict:
                plt.figure(dpi=200)
                plt.plot(t,u,label='Input Signal')
                s1 = ("Modelled: Kp:%.1f τ:%.1f θ:%.1f  %i%% Noise" % (sig.kps[i],sig.taus[i],sig.thetas[i],sig.stdev))
                s2 = ("Predicted: Kp:%.1f (%.1f) τ:%.1f (%.1f) θ:%.1f (%.1f)" % (Kp,Kperror,taup,tauperror,theta,thetaerror))
                plt.plot(t,yTrue, label=s1)
                plt.plot(t,yPred,'--', label=s2)
                plt.xlabel("Time (s)")
                plt.ylabel("Change from set point")
                plt.legend()
           
            if savePredict:
                savePath = self.pd+"/Predictions/SISO/" + str(i) + ".png"
                plt.savefig(savePath)       
    
    def predict_MIMO(self,sig,plotPredict=True,savePredict=False,probabilityCall=False):
        """
        Takes Signal object with input and output data, uses stored model
        to predict parameters based off simulated data, then plots
        system output based on predicted parameters.
        
        Plots response using predicted parameters along with real response,
        as well as real and predicted parameters.
        
        Parameters
        ----------
        sig : Signal (object), default=False
            Must provide instance of signal class in order to build model or use
            for prediction. Else, will fail to train. 
        
        plotPredict : bool, default=True
            Plots predicted response with real response.
        
        savePredict : bool,default=False
            Determine whether or not to save predicted responses.
            
        probabilityCall : bool, default=False
            Included to avoid building probability object while trying to 
            build probability object, leading to endless loop.
        """
        if not(isinstance(sig,Signal)):
            print("You need to predict with an instance of signal!")
            return
        
        if not(probabilityCall):
            prob = Probability(sig)
            Kperror,tauperror = prob.get_errors()
        
        # Model is used to predict all out the output variables linearly,
        # so the array is cut depending on which transfer function parameters 
        # correspond to each output variable, then stacked as new trials
        kp_yhat = self.modelDict['kp'](sig.xData['kp'])
        tau_yhat = self.modelDict['tau'](sig.xData['tau'])
        
        kpPredPrimal = np.array(kp_yhat.mean())[0,...]     
        kpStdPrimal = np.array(2*kp_yhat.stddev())[0,...]  
        kpPred = np.split(kpPredPrimal,sig.outDim,axis=0)
        kpStd = np.split(kpStdPrimal,sig.outDim,axis=0)
        
        tauPredPrimal = np.array(tau_yhat.mean())[0,...]     
        tauStdPrimal = np.array(2*tau_yhat.stddev())[0,...]  
        tauPred = np.split(tauPredPrimal,sig.outDim,axis=0)
        tauStd = np.split(tauStdPrimal,sig.outDim,axis=0)
        
        self.kpPredictions = np.concatenate(kpPred,axis=1)
        self.tauPredictions = np.concatenate(tauPred,axis=1)
        self.kpErrors = np.concatenate(kpStd,axis=1)
        self.tauErrors = np.concatenate(tauStd,axis=1)
        
        self.errors = []
        uArrays = sig.uArray
        yArrays = sig.yArray
        
        # iterate over the total number of trials
        for k in range(0,sig.numTrials):
            # Get the parameters for predicted system response
            taup = self.tauPredictions[k]
            Kp = self.kpPredictions[k]
            kerr = self.kpErrors[k]
            terr = self.tauErrors[k]
            
            t = np.linspace(0,sig.timelength,self.nstep)
            u = uArrays[k,:,:]
            
            nums = []
            dens = []
            
            # The transfer function from the 2nd input to the 1st output is
            # (3s + 4) / (6s^2 + 5s + 4).
            # num = [[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]]
            # Iterate over each of the output dimensions and
            # add to numerator 
            for j in range(0,sig.outDim):
                numTemp = []
                denTemp = []
                for i in range(0,sig.inDim):
                    # Iterate over each of the input dimensions
                    # and add to the numerator array
                    numTemp.append([Kp[(sig.inDim*j)+i]])
                    denTemp.append([taup[(sig.inDim*j)+i],1.])
                nums.append(numTemp)
                dens.append(denTemp)
             
            # Use transfer function class to simulate system response
            # to MIMO input and randomized parameters 
            nums=np.array(nums)
            dens=np.array(dens)
            sys = control.tf(nums,dens)
            _,yPred,_ = control.forced_response(sys,U=np.transpose(u),T=t)
            yPred = np.transpose(yPred)
            yTrue = yArrays[k,:,:]
            
            # Make array of mean squared errors to use later
            meanerror = np.mean(self.MSE(yPred,yTrue))
            self.errors.append(meanerror)
            
            # Plot predicted and real system response 
            # based on predicted parameters
            if plotPredict:
                plt.figure(dpi=200)
                plt.plot(t,u)                   
                s1 = ("Predicted: Kp:%.1f (%.1f), %.1f (%.1f) τ:%.1f (%.1f), %.1f (%.1f)" % (Kp[0],kerr[0],Kp[1],kerr[1],taup[0],terr[0],taup[1],terr[1]))
                s2 = ("Predicted: Kp%.1f (%.1f), %.1f (%.1f) τ:%.1f (%.1f), %.1f (%.1f)" % (Kp[2],kerr[2],Kp[3],kerr[3],taup[2],terr[2],taup[3],terr[3]))
                s3 = ("Modelled: Kp:%.1f, %.1f τ:%.1f, %.1f" % (sig.kps[k,0],sig.kps[k,1],sig.taus[k,0],sig.taus[k,1]))
                s4 = ("Modelled: Kp%.1f, %.1f τ:%.1f, %.1f" % (sig.kps[k,2],sig.kps[k,3],sig.taus[k,2],sig.taus[k,3]))
                plt.plot(t,yTrue[:,0],'r',label=s3)
                plt.plot(t,yPred[:,0],'k--',label=s1)
                plt.plot(t,yTrue[:,1],'b',label=s4)
                plt.plot(t,yPred[:,1],'g--',label=s2)
                plt.xlabel("Time (s)")
                plt.ylabel("Change from set point")
                plt.legend()
           
            if savePredict and k<10:
                savePath = self.pd+"/Predictions/MIMO/" + str(k) + ".png"
                plt.savefig(savePath)
                
    def coeff_determination(self,y_true, y_pred):
        "Coefficient of determination for callback"
        SS_res =  K.sum(K.square( y_true-y_pred ))
        SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
        return ( 1 - SS_res/(SS_tot + K.epsilon()) )                
    
    # Specify the surrogate posterior over `keras.layers.Dense` `kernel` and `bias`.
    def posterior_mean_field(kernel_size, bias_size=0, dtype=None):
      n = kernel_size + bias_size
      c = np.log(np.expm1(1.))
      return tf.keras.Sequential([
          tfp.layers.VariableLayer(2 * n, dtype=dtype,initializer='glorot_uniform'),
          tfp.layers.DistributionLambda(lambda t: tfd.Independent(
              tfd.Normal(loc=t[..., :n],scale=1e-5 + tf.nn.softplus(c + t[..., n:])),reinterpreted_batch_ndims=1)),
      ])
     
  # Specify the prior over `keras.layers.Dense` `kernel` and `bias`.
    def prior_trainable(kernel_size, bias_size=0, dtype=None):
      n = kernel_size + bias_size
      return tf.keras.Sequential([
          tfp.layers.VariableLayer(n, dtype=dtype,initializer='glorot_uniform'),
          tfp.layers.DistributionLambda(lambda t: tfd.Independent(
              tfd.Normal(loc=t, scale=1),reinterpreted_batch_ndims=1)),
      ])

    def mutable_model(self,x_train,y_train):
        length,width,height = x_train.shape
        outheight,outwidth = y_train.shape
        "Probabilistic model for SISO data"
        negloglik = lambda y, rv_y: -rv_y.log_prob(y[:])
        model = tf.keras.Sequential([
        tf.keras.layers.LSTM(int(width/2), activation='tanh',input_shape=(width,height)),
        tf.keras.layers.Dense(int(self.inDim*10),activation='linear'),
        tfp.layers.DenseVariational(int(self.inDim*2),Model.posterior_mean_field,Model.prior_trainable,activation='linear',kl_weight=1/x_train.shape[0]),
        tfp.layers.DistributionLambda(lambda t: tfd.Normal(
        loc = [t[..., :self.inDim] for i in range(self.inDim)],
        scale = [1e-3 + tf.math.softplus(0.1 * t[...,self.inDim:]) for i in range(self.inDim)],)),])
        model.compile(optimizer='adam', loss=negloglik,metrics=[self.coeff_determination])
        return model
    
class Probability:
    """
    Class for running a series of simulations at a pre-known
    maximum quantity of error and getting standard deviation
    of predicted parameters from true value.
    
    Runs numTrials trials for each quantity of error between
    maximum error and zero, then plots and gets coefficient
    of determination.
    
    The Purpose of this class is to return the standard deviation 
    of each parameter based on estiamted gaussian noise in the data
    and try and predict uncertainty range of prediction.
    
    Parameters
    ----------
    sig : Signal (object), default=False
        Must provide instance of signal class in order to 
        build model or use for prediction. Else, will fail 
        to train. 
    
    maxError : float, default=5.
        Max standard deviation of gaussian noise added to 
        simulations. All other standard devations 
        between 0 and max are used to predict.
        
    numTrials : Int, default=1000
        Number of simulations run for each quantity of standard 
        deviation added to simulated response.
    
    plotUncertainty: bool, defaulyt=True
        Determine whether or not to plot histrogram and plot
        of predictions with coefficient of determination.
    
    Attributes
    ----------
    SISO_probability_estimate
        Simulates 1000 system responses with specified
        quantity of noise, predicts response with saved
        model and plots prediction/validation data 
        points and coefficient of determination.
    
    MIMO_probability_estimate
        Simulates 1000 MIMO system responses with specified
        quantity of noise, predicts response with saved
        model and plots prediction/validation data 
        points and coefficient of determination.   
    
    """
    def __init__(self,sig=False,maxError=5,numTrials=1000,plotUncertainty=True):
        self.plotUncertainty = plotUncertainty
        self.maxError = int(maxError)
        self.deviations = np.arange(0,maxError)
        self.numTrials = numTrials
        self.maxError = sig.stdev
        self.nstep = sig.nstep
        self.timelength = sig.timelength
        self.trainFrac = sig.trainFrac
        self.type = sig.type
            
        if (self.maxError==5 and numTrials==1000):
            suffix = "Default"
        else:
            suffix = '_error_' + str(self.maxError) + '_nTrials_' + str(numTrials)
        
        self.prefix = self.pd+"/Uncertainty/"
        if self.type=="SISO":
            self.errorCSV = self.prefix+"SISO/Data/propData"+suffix+".csv"
            self.SISO_probability_estimate()
        else:
            self.errorCSV = self.prefix+"MIMO/Data/propData"+suffix+".csv"
            self.MIMO_probability_estimate()
        
    def SISO_probability_estimate(self):    
        """ Simulates 1000 system responses with specified quantity of 
        noise, predicts response with saved model and plots prediction
        and validation data  points and coefficient of determination."""
        
        if not(path.exists(self.errorCSV)):
            print("No simulation for these parameters exists in \
                  Uncertainty data. Proceeding with simulation")
           
            # Initialize the models that are saved using the parameters declared above
            predictor = Model(self.nstep)
            predictor.load_SISO()
                
            deviations = np.arange(0,self.maxError)
            
            # Empty arrays to put predictions in
            stdev = np.array([0])
            error=np.array([0])
            kp_pred = np.array([0])
            theta_pred = np.array([0])
            tau_pred = np.array([0])
            
            # Empty arrays to put true parameters in
            kp_true = np.array([0])
            theta_true = np.array([0])
            tau_true = np.array([0])
            
            # Produce simulation for each standard deviation below
            # the max standard deviation to add to the gaussian
            # noise after simulation
            for deviation in deviations:
                numTrials = self.numTrials; nstep = self.nstep
                timelength = self.timelength; trainFrac = self.trainFrac
                
                # then simulates using the initialized model
                sig = Signal(numTrials,nstep,timelength,trainFrac,stdev=deviation)
                sig.SISO_simulation(KpRange=[1,10],tauRange=[1,10],thetaRange=[1,10])
                
                # In this case, since we are only loading the model, not trying to train it,
                # we can use function simulate and preprocess
                xData,yData = sig.SISO_validation()
            
                # Function to make predictions based off the simulation 
                predictor.predict_SISO(sig,savePredict=False,plotPredict=False)
                
                # Add predicted values to end of arrays
                error = np.concatenate((predictor.errors,error))
                kp_pred = np.concatenate((predictor.kpPredictions[:,0],kp_pred))
                theta_pred = np.concatenate((predictor.thetaPredictions[:,0],theta_pred))
                tau_pred = np.concatenate((predictor.tauPredictions[:,0],tau_pred))
                
                # Add true values to the end of arrays
                kp_true = np.concatenate((sig.kps,kp_true))
                theta_true = np.concatenate((sig.thetas,theta_true))
                tau_true = np.concatenate((sig.taus,tau_true))
                stdev = np.concatenate((np.full_like(predictor.errors,deviation),stdev))
            
            # Transfer to Pandas before saving to CSV 
            sd = pd.DataFrame()
            sd['stdev'] = stdev
            sd['mse'] = error
            sd['kpPred'] = kp_pred
            sd['tauPred'] = tau_pred
            sd['thetaPred'] = theta_pred
            sd['kpTrue'] = kp_true
            sd['tauTrue'] = tau_true
            sd['thetaTrue'] = theta_true
            
            sd.to_csv(self.errorCSV, index=False)
        
        # If the CSV with data already exists, can skip training and plot
        else:
            print("Data exists for the parameters, proceeding to producing uncertainty estimate")
            try:
                sd = pd.read_csv(self.errorCSV).drop(['Unnamed: 0'],axis=1)
                sd.drop(sd.tail(1).index,inplace=True)
            except:
                sd = pd.read_csv(self.errorCSV)
                sd.drop(sd.tail(1).index,inplace=True)
           
        # Iterate over each of the parameters and plot the histogram
        # as well as the prediction/true parameters
        self.errorDict = {}  
        pathPrefix = self.prefix+"SISO/Plots/"
        prefixes = ['kp','tau','theta']
        for (i,prefix) in enumerate(prefixes):
            sd[prefix+'Error'] = (sd[prefix+'Pred']-sd[prefix+'True'])
            h = np.std(sd[prefix+'Error'])
            self.errorDict[prefix] = h
            
            if self.plotUncertainty:
                plt.figure(dpi=200)
                plt.hist(sd[prefix+'Error'],bins=100,label='Max Error = %i%%' % self.maxError)
                plt.xlabel('Standard Error in '+ prefix)
                plt.ylabel("Frequency Distribution")
                plt.legend()
                savePath = pathPrefix + "histogram_" + prefix + ".png"
                plt.savefig(savePath)
                
                plt.figure(dpi=200)
                plt.plot(sd[prefix+'True'],sd[prefix+'Pred'],'.',label='Max Error = %i%%' % self.maxError)
                plt.plot(np.linspace(1,10),np.linspace(1,10),'r--',label="r\u00b2 = %.3f" % r2_score(sd[prefix+'True'],sd[prefix+'Pred']))
                plt.plot(np.linspace(1,10),np.linspace(1,10)+h,'g--',label="Stdev = %.3f" % h)
                plt.plot(np.linspace(1,10),np.linspace(1,10)-h,'g--')
                plt.ylabel("Predicted Value of "+prefix)
                plt.xlabel("True Value of "+prefix)
                plt.legend()
                savePath = pathPrefix + "determination_" + prefix + ".png"
                plt.savefig(savePath)
                
    def MIMO_probability_estimate(self):    
        """ Simulates 1000 system responses with specified quantity of 
        noise, predicts response with saved model and plots prediction
        and validation data  points and coefficient of determination."""
        
        if not(path.exists(self.errorCSV)):
            print("No simulation for these parameters exists in Uncertainty data. Proceeding with simulation")
           
            # Initialize the models that are saved using the parameters declared above
            predictor = Model(self.nstep)
            predictor.load_MIMO()
                
            deviations = np.arange(0,self.maxError)
            
            stdev = np.array([0])
            error=np.array([0])
            kp_pred = np.array([0])
            tau_pred = np.array([0])
            
            kp_true = np.array([0])
            tau_true = np.array([0])
            
            for deviation in deviations:
                numTrials = self.numTrials; nstep = self.nstep
                timelength = self.timelength; trainFrac = self.trainFrac
                
                # then simulates using the initialized model
                sig = Signal(numTrials,nstep,timelength,trainFrac,stdev=deviation)
                sig.MIMO_simulation(KpRange=[1,10],tauRange=[1,10])
                
                # In this case, since we are only loading the model, not trying to train it,
                # we can use function simulate and preprocess
                xData,yData = sig.MIMO_validation()
            
                # Function to make predictions based off the simulation 
                predictor.predict_MIMO(sig,savePredict=False,plotPredict=False,probabilityCall=True)
                
                error = np.concatenate((predictor.errors,error))
                kp_pred = np.concatenate((predictor.kpPredictions.ravel(),kp_pred))
                tau_pred = np.concatenate((predictor.tauPredictions.ravel(),tau_pred))
                
                kp_true = np.concatenate((sig.kps.ravel(),kp_true))
                tau_true = np.concatenate((sig.taus.ravel(),tau_true))
                stdev = np.concatenate((np.full_like(predictor.errors,deviation),stdev))
            
            # Transfer to pandas to save to CSV
            sd = pd.DataFrame()
            sd['kpPred'] = kp_pred
            sd['tauPred'] = tau_pred
            sd['kpTrue'] = kp_true
            sd['tauTrue'] = tau_true
            
            sd.to_csv(self.errorCSV, index=False)
          
        # If the simulation does exist for this set of parameters,
        # load from CSV and plot
        else:
            print("Data exists for the parameters, proceeding to producing uncertainty estimate")
            try:
                sd = pd.read_csv(self.errorCSV).drop(['Unnamed: 0'],axis=1)
                sd.drop(sd.tail(1).index,inplace=True)
                
            except:
                sd = pd.read_csv(self.errorCSV)
                sd.drop(sd.tail(1).index,inplace=True)
        
        # Plot histogram and predicted/true parameters
        self.errorDict = {}  
        pathPrefix = self.prefix+"MIMO/Plots/"
        prefixes = ['kp','tau']
        for (i,prefix) in enumerate(prefixes):
            sd[prefix+'Error'] = (sd[prefix+'Pred']-sd[prefix+'True'])
            h = np.std(sd[prefix+'Error'])
            self.errorDict[prefix] = h
            
            if self.plotUncertainty:
                plt.figure(dpi=200)
                plt.hist(sd[prefix+'Error'],bins=100,label='Max Error = %i%%' % self.maxError)
                plt.xlabel('Standard Error in '+ prefix)
                plt.ylabel("Frequency Distribution")
                plt.legend()
                savePath = pathPrefix + "histogram_" + prefix + ".png"
                plt.savefig(savePath)
                
                plt.figure(dpi=200)
                plt.plot(sd[prefix+'True'],sd[prefix+'Pred'],'.',label='Max Error = %i%%' % self.maxError)
                plt.plot(np.linspace(1,10),np.linspace(1,10),'r--',label="r\u00b2 = %.3f" % r2_score(sd[prefix+'True'],sd[prefix+'Pred']))
                plt.plot(np.linspace(1,10),np.linspace(1,10)+h,'g--',label="Stdev = %.3f" % h)
                plt.plot(np.linspace(1,10),np.linspace(1,10)-h,'g--')
                plt.ylabel("Predicted Value of "+prefix)
                plt.xlabel("True Value of "+prefix)
                plt.legend()
                savePath = pathPrefix + "determination_" + prefix + ".png"
                plt.savefig(savePath)
                
                
    def get_errors(self):
        """Access standard uncertainty from dictionary, 
        return as tuple"""
        return self.errorDict.values()

   