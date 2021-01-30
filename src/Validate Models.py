 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 10:16:14 2020

@author: RileyBallachay
"""

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from Signal import Signal
from Model import Model
import time
import os
import numpy as np
import seaborn as sns
import scipy.stats as stats
import matplotlib.pylab as pylab

# These constants are also defined in the Signal module 
# Don't change here unless you also change them there
NUMTRIALS = 100
batchSize = 32
plots = 5

valPath = '/Users/RileyBallachay/Documents/Fifth Year/RNNSystemIdentification/Model Validation/'
model_paths = [f.path for f in os.scandir(valPath) if f.is_dir()]

inDims = range(1,2)
outDims = range(1,2)

for (inDimension,outDimension) in zip(inDims,outDims): 
    name ='MIMO ' + '1' + 'x' + '1'
    path = valPath + name + '/Checkpoints/'
    
    start_time = time.time()
    numTrials=int(NUMTRIALS/(inDimension*outDimension))
    sig = Signal(inDimension,outDimension,numTrials,numPlots=plots,stdev='variable')

    # In this case, since we are only loading the model, not trying to train it,
    # we can use function simulate and preprocess 
    xData,yData = sig.system_validation(b_possible_values=[.01,.99],a_possible_values=[.01,.99],
                                        k_possible_values=[1,10],order=False)
    print("--- %s seconds ---" % (time.time() - start_time))
    
    # Initialize the models that are saved using the parameters declared above
    predictor = Model()
    predictor.load_model(sig,path)
    
    # Function to make predictions based off the simulation 
    predDict,errDict = predictor.predict_system(sig,savePredict=True,stepResponse=False)
    yData['theta'] = yData['theta']/10
    
    
    sns.set()
    fig,axes = plt.subplots(4,4,dpi=400,sharex=True,sharey=True) 
    #plt.xticks([0,0.5,1])
    plt.yticks([0,0.5,1])
    #fig.suptitle('Sharing x per column, y per row')
    for (i,axis) in enumerate(axes):
        for (j,ax) in enumerate(axis):
            ax.plot([yData['tau'][len(axis)*i+j],yData['tau'][len(axis)*i+j]],[0,1.25],'r--',linewidth=1,alpha=0.7,label='a')
            ax.plot([yData['kp'][len(axis)*i+j],yData['kp'][len(axis)*i+j]],[0,1.25],'b--',linewidth=1,alpha=0.7,label='b')
            ax.plot([yData['theta'][len(axis)*i+j],yData['theta'][len(axis)*i+j]],[0,1.25],'g--',linewidth=1,alpha=0.7,label='k')
            
            mu = predDict['tau'][len(axis)*i+j]
            sigma = errDict['tau'][len(axis)*i+j]
            x = np.linspace(0,1.25, 100)
            pdf = stats.norm.pdf(x, mu, sigma)/100
            pdf=pdf/max(pdf)
            pdf[pdf<0.00001]=np.NaN
            ax.plot(x, pdf,'maroon',linewidth=1,label='a Pred.')
            
            mu = predDict['kp'][len(axis)*i+j]
            sigma = errDict['kp'][len(axis)*i+j]
            x = np.linspace(0,1.25, 100)
            pdf = stats.norm.pdf(x, mu, sigma)/100
            pdf=pdf/max(pdf)
            pdf[pdf<0.00001]=np.NaN
            ax.plot(x, pdf,'navy',linewidth=1,label='b Pred.')
            
            mu = predDict['theta'][len(axis)*i+j]/10
            sigma = errDict['theta'][len(axis)*i+j]/10
            x = np.linspace(0,1.25, 100)
            pdf = stats.norm.pdf(x, mu, sigma)
            pdf=pdf/max(pdf)
            pdf[pdf<0.000001]=np.NaN
            ax.plot(x, pdf,'darkgreen',linewidth=1,label='k Pred.')

            ax.set_xlim(-0.25, 1.25)
            ax.set_ylim(0, 1.25)
            ax.tick_params(axis='both', which='major', labelsize=8)
            
            if (len(axis)*i+j)==7:
                ax.legend(loc='center left', bbox_to_anchor=(1, 0),fontsize=7)
            
    #for ax in fig.get_axes():
        #ax.label_outer()
    
    #plt.ylabel("Predicted Parameter Value")
    #plt.xlabel("Probability Density")
    #plt.tight_layout()
    plt.subplots_adjust(wspace=0.1, hspace=0.2)
    fig.text(0.5, 0.04, 'Parameter Value', ha='center',fontsize=7)
    fig.text(0.04, 0.5, 'Probability Density', va='center', rotation='vertical',fontsize=7)
    
    print("--- %s seconds ---" % (time.time() - start_time))
    
    
    print("--- %s seconds ---" % (time.time() - start_time))