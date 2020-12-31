#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 10:22:44 2020

@author: RileyBallachay
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from Model import Model
path = '/Users/RileyBallachay/Documents/Fifth Year/RNNSystemIdentification/DAISY Data/cstr.txt'

data = pd.read_csv(path, sep="\t", header=None)[000:100]
data.columns = ["t", "q (l/min)", "c (mol/l)", "Temp (K)","Drop"]
data.drop("Drop",axis=1,inplace=True)



plt.figure(dpi=200)
plt.plot(data['t'],-(data['Temp (K)']-np.mean(data['Temp (K)'])),label='CSTR Temp (K)')
plt.plot(data['t'],(data['q (l/min)']-np.mean(data['q (l/min)'])),label='Coolant Flow (l/min)')
plt.xlabel('Time (min)')

predictor = Model(100,Modeltype='regular')
predictor.load_SISO()

uArray = np.array(((data['q (l/min)']-np.mean(data['q (l/min)']))))
yArray = np.array(-(data['Temp (K)']-np.mean(data['Temp (K)'])))

predictions = predictor.predict_real_SISO(uArray,yArray)
"""

path2 = '/Users/RileyBallachay/Documents/Fifth Year/RNNSystemIdentification/DAISY Data/dryer2.txt'

data2 = pd.read_csv(path2,sep='  ', header=None)[:500]
data2.columns = ['Time *10 (s)','Fuel Rate (l/s)','Fan Speed (hz)','Input Rate (l/s)','Dry Temp (C)', 'Wet Temp (C)','Moisture (%)']

plt.figure(dpi=200)
plt.plot(data2['Time *10 (s)'],data2['Fuel Rate (l/s)'])
plt.plot(data2['Time *10 (s)'],data2['Dry Temp (C)'])

path3 = '/Users/RileyBallachay/Documents/Fifth Year/RNNSystemIdentification/DAISY Data/exchanger.txt'

data3 = pd.read_csv(path3,sep='  ', header=None)[200:1000]
data3.columns = ['Time (s)','Flow Rate (l/s)', 'Outlet Temp (C)']

plt.figure(dpi=200)
plt.plot(data3['Time (s)'],data3['Flow Rate (l/s)']-np.mean(data3['Flow Rate (l/s)']))
plt.plot(data3['Time (s)'],data3['Outlet Temp (C)']-np.mean(data3['Outlet Temp (C)']))
"""