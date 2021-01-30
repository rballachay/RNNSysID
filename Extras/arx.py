#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 11:15:14 2020

@author: RileyBallachay
"""

import numpy as np
from gekko import GEKKO
import matplotlib.pyplot as plt
import pandas as pd
import control
from scipy import signal

# load data and parse into columns
url = 'http://apmonitor.com/do/uploads/Main/tclab_dyn_data2.txt'
data = pd.read_csv(url)
t = data['Time']
u = data[['H1']]
y = data[['T1']]

# generate time-series model
m = GEKKO(remote=False) # remote=True for MacOS

# system identification
na = 1 # output coefficients
nb = 1 # input coefficients
yp,p,K = m.sysid(t,u,y,na,nb,diaglevel=1)

plt.figure()
plt.subplot(2,1,1)
plt.plot(t,u)
plt.legend([r'$u_0$',r'$u_1$'])
plt.ylabel('MVs')
plt.subplot(2,1,2)
plt.plot(t,y)
plt.plot(t,yp)
plt.legend([r'$y_0$',r'$y_1$',r'$z_0$',r'$z_1$'])
plt.ylabel('CVs')
plt.xlabel('Time')

u = np.array(data[['H1']])[:,0]

y = np.zeros((601,1))
y[0]=19.29
for i in range(1,len(t)):
    y[i]=(0.00284898*u[i-1]+ 0.99628219*y[i-1]+0.00858412)

plt.plot(t,y,'--')

plt.show()