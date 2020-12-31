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

    def __init__(self,Modeltype='probability',sig=False):   
        self.Modeltype = Modeltype
        self.names = ["kp","tau","theta"]
        self.modelDict = {}
        self.special_value = -99
        self.cwd = str(os.getcwd())
        self.pd = str(Path(os.getcwd()).parent)
    
    def load_model(self,sig,directory=False,check=False):
        """Loads one of two first order models: probability or regular. Iterates 
        over directory and loads alphabetically. If more than 3 models exist in the 
        directory, it will load them indiscriminately."""
        modelList = []
        self.inDim = sig.inDim; self.outDim = sig.outDim
        if not(directory):
            loadDir =  askdirectory(title='Select Folder With Trained Model')
        else:
            loadDir = directory
            
        models = ['kp.cptk','tau.cptk','theta.cptk']
        for filename in models:
            if filename=='.DS_Store':
                continue
            print(filename)
            
            if not(check):
                model = self.mutable_model(sig.xData['kp'],sig.yData['kp'])
            else:
                model = self.mutable_model(self.x_train,self.y_train)
                
            model.load_weights(loadDir+filename)
            modelList.append(model)
        for i in range(0,3):
            self.modelDict[self.names[i]] = modelList[i]
     
        
    def load_and_train(self,sig,epochs=50,probabilistic=True,saveModel=True,batchSize=16,plotLoss=False,plotVal=False):
        """Calls load_SISO and continues training model, saving the updated 
        model to a new folder to avoid overwriting"""
        
        self.train_model(sig, saveModel=saveModel,epochs=epochs,checkpoint=True,batchSize=batchSize)

    
    def train_model(self,sig=False,plotLoss=True,plotVal=True,epochs=50,saveModel=True,
                    checkpoint=False,batchSize=16):
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
        first=True
        
        # Iterate over tau and kp parameters
        for (k,parameter) in enumerate(parameters):
            xData,yData = sig.stretch_MIMO(parameter)
            x_train,x_val,y_train,y_val,numDim = sig.preprocess(xData,yData)
            self.outDim = sig.outDim;self.inDim=sig.inDim
            self.length,self.width,self.height=x_train.shape
            self.numTrials=sig.numTrials;self.nstep=sig.nstep
            self.maxLen = sig.maxLen
            self.trainFrac = sig.trainFrac
            """
            # Check the dimension of data to ensure that an architecture 
            # exists for the shape of the data. If not, then it will 
            # prompt the user to make a new architecture
            if checkpoint:
                if first:
                    self.x_train = x_train; self.y_train = y_train
                    valPath = '/Users/RileyBallachay/Documents/Fifth Year/RNNSystemIdentification/Model Validation/'
                    name ='MIMO ' + str(sig.inDim) + 'x' + str(sig.outDim)
                    path = valPath + name + '/Checkpoints/'
                    self.load_model(sig,directory=path,check=True)
                    first = False
                    
                model = self.modelDict[parameter]
            else:
            """
            model = self.mutable_model()
                
            print(model.summary())
            
            # Set threshold to stop training when coefficient of determinatio gets to 0.9. 
            # Section for including callbacks (custom and checkpoint)
            checkpoint_path = checkptDir + self.names[k] + '.cptk'
            cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                        save_weights_only=True,save_best_only=True,verbose=1) 
            MTC = MyThresholdCallback(0.97)
            
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
            #modelList.append(model)
            #self.modelDict[self.names[k]] = model
            #yhat = model(x_val)
            #predictions = np.array(yhat.mean())
            #stddevs = np.array(yhat.stddev())

            # Only save model if true
            """
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
            """
            # Plot predicted and actual value for each paramter and
            # coefficient of determination
            """
            if plotVal:
                fig, axes = plt.subplots(1, sig.inDim,dpi=200) 
                fig.add_subplot(111, frameon=False)
                plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
                for (i,ax) in enumerate(axes):
                    ax.plot(y_val[:,i],predictions[:,i],'.b',label=parameter)
                    print(stddevs.shape)
                    r2 =("r\u00b2 = %.3f" % r2_score(y_val[:,i],predictions[:,i]))
                    ax.errorbar(y_val[:,i],predictions[:,i],yerr=stddevs[:,i]*2,fmt='none',ecolor='green')
                    ax.plot(np.linspace(0,10),np.linspace(0,10),'--r',label = r2)
                    ax.legend(prop={'size': 8})
                
                plt.ylabel("Predicted Value of "+ parameter)
                plt.xlabel("True Value of "+ parameter)

                if saveModel:
                    plt.savefig(plotDir+self.names[k]+'.png')
                plt.show()  
            """
        return  
 
    
    def MSE(self,y_true,y_pred):
        """Calculate mean squared error of 
        predicted and actual system response."""
        mse = sum((y_true-y_pred)**2)
        return mse
    
    def predict_system(self,sig,plotPredict=True,savePredict=False,probabilityCall=False):
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
        
        # Model is used to predict all out the output variables linearly,
        # so the array is cut depending on which transfer function parameters 
        # correspond to each output variable, then stacked as new trials
        kp_yhat = self.modelDict['kp'](sig.xData['kp'])
        tau_yhat = self.modelDict['tau'](sig.xData['tau'])
        theta_yhat = self.modelDict['theta'](sig.xData['theta'])
        
        predDict = dict()
        errDict = dict()
        self.stdDict = dict()
        predDict['kp'] = np.array(kp_yhat.mean()).flatten()
        errDict['kp'] = np.array(2*kp_yhat.stddev()).flatten()
        self.stdDict['kp'] = np.mean(errDict['kp'])
        
        predDict['tau'] = np.array(tau_yhat.mean()).flatten()
        errDict['tau'] = np.array(2*tau_yhat.stddev()).flatten()
        self.stdDict['tau'] = np.mean(errDict['tau'])
        
        predDict['theta'] = np.array(theta_yhat.mean()).flatten()
        errDict['theta'] = np.array(2*theta_yhat.stddev()).flatten()
        self.stdDict['theta'] = np.mean(errDict['theta'])
            
        fig, axes = plt.subplots(1, 3,figsize=(15,5),dpi=400)  
        for (idx,parameter) in enumerate(['kp','tau','theta']):
            sig.yData[parameter] = sig.yData[parameter].flatten()
            sig.xData[parameter] = sig.xData[parameter].flatten()
            ax = axes[idx]
            fig.add_subplot(111, frameon=False)
            plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
            ax.plot(sig.yData[parameter],predDict[parameter],'.b',label=parameter)
            r2 =("r\u00b2 = %.3f" % r2_score(sig.yData[parameter],predDict[parameter]))
            ax.errorbar(sig.yData[parameter],predDict[parameter],yerr=errDict[parameter],fmt='none',ecolor='green',label=('Avg. Unc.=%.2f' %self.stdDict[parameter]))
            ax.plot(np.linspace(0,10),np.linspace(0,10),'--r',label = r2)
            ax.legend()
        
        for ax in axes.flat:
            ax.label_outer()
        
        plt.ylabel("Predicted Parameter Value")
        plt.xlabel("True Parameter Value")


    def coeff_determination(y_true, y_pred):
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
              tfd.Normal(loc=t[..., :n],scale=1e-5 + tf.nn.softplus(c + t[..., n:])),reinterpreted_batch_ndims=1)),])
     
  # Specify the prior over `keras.layers.Dense` `kernel` and `bias`.
    def prior_trainable(kernel_size, bias_size=0, dtype=None):
      n = kernel_size + bias_size
      return tf.keras.Sequential([
          tfp.layers.VariableLayer(n, dtype=dtype,initializer='glorot_uniform'),
          tfp.layers.DistributionLambda(lambda t: tfd.Independent(
              tfd.Normal(loc=t, scale=1),reinterpreted_batch_ndims=1)),])

    def mutable_model(self):
        "Probabilistic model for SISO data"
        negloglik = lambda y, rv_y: -rv_y.log_prob(y[:])
        model = tf.keras.Sequential([
        tf.keras.layers.Masking(mask_value=self.special_value, input_shape=(None, self.height)),
        tf.keras.layers.LSTM(100, activation='tanh'),          
        tf.keras.layers.Dense(20,activation='linear'),
        tfp.layers.DenseVariational(2*self.inDim,Model.posterior_mean_field,Model.prior_trainable,activation='linear',kl_weight=1/self.nstep*self.trainFrac),
        tfp.layers.DistributionLambda(lambda t: tfd.Normal(loc = t[..., :self.inDim],
        scale = (1e-3 + tf.math.softplus(0.1 * t[...,self.inDim:])),)),])
        model.compile(optimizer='adam', loss=negloglik,metrics=[Model.coeff_determination])
        return model
    
    def mutable_model_noAttribute(x_train,y_train):
        length,width,height = x_train.shape
        outheight,outwidth = y_train.shape
        "Probabilistic model for SISO data"
        negloglik = lambda y, rv_y: -rv_y.log_prob(y[:])
        model = tf.keras.Sequential([
        tf.keras.layers.LSTM(int(width/2), activation='tanh',input_shape=(None,height)),          
        tf.keras.layers.Dense(int(1*10),activation='linear'),
        tfp.layers.DenseVariational(2*1,Model.posterior_mean_field,Model.prior_trainable,activation='linear',kl_weight=1/x_train.shape[0]),
        tfp.layers.DistributionLambda(lambda t: tfd.Normal(loc = t[..., :1],
        scale = (1e-3 + tf.math.softplus(0.1 * t[...,1:])),)),])
        model.compile(optimizer='adam', loss=negloglik,metrics=[Model.coeff_determination])
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
    def __init__(self,sig=False,maxError=5,numTrials=100,plotUncertainty=True):
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

   