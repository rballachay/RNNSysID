#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 11:08:56 2021

@author: RileyBallachay
"""


import pandas as pd
import Signal as Signal


path = '/Users/RileyBallachay/Documents/Fifth Year/RNNSystemIdentification/Model Validation/Paper Results.csv'

df = pd.read_csv(path)
df = df.loc[:, ~df.columns.str.contains('Unnamed')]

sig = Signal(inDimension,outDimension,numTrials,numPlots=plots)
    
