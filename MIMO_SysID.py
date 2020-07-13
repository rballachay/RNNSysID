#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 17:18:50 2020

@author: RileyBallachay
"""
from Signal import Signal
from Model import Model
import time
import numpy as np

# These constants are also defined in the Signal module 
# Don't change here unless you also change them there
numTrials = 1000
nstep = 100
timelength = 100
trainFrac = .7

start_time = time.time()
sig = Signal(numTrials,nstep,timelength,trainFrac,stdev=1)
sig.MIMO_simulation(stdev=0)
print("--- %s seconds ---" % (time.time() - start_time))
