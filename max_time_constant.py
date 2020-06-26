#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 12:24:04 2020

@author: RileyBallachay
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 14:50:26 2020

@author: RileyBallachay
"""
import scipy
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

class Signal:
 
    # Going to need this to be interactive 
    # i.e. the function waits until the 
    def __init__(self,nstep):
        self.i = -1
        self.nstep = nstep 
        self.signal = self.pulsed_signal()

    def pulsed_signal(self):
        # pulsed signal generation
        signal = np.zeros(nstep)
        start1 = int(nstep/5)
        end1 = int(start1*2)
        start2 = int(start1*3)
        end2 = int(start1*4)
        
        signal[start1:end1] = 1
        signal[start2:end2] = 1
            
        return signal

    def increment(self):   
        
        self.i += 1
        try:
            self.signal[self.i]
        except:
            self.i -= 1  
        return self.signal[self.i]


    def step_function(self,t):  
        if isinstance(t,np.ndarray):
            t[249:750] = 1
        else:
            t = [0 if (t<25 or t>75) else 1]
        return t

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

# ODE Integrator
def model(y,t,timearray,inputarray,Kp,taup):   
    index = find_nearest(timearray,t)
    u = inputarray[index]
    return (-y + Kp * u)/taup

# Simulate taup * dy/dt = -y + K*u
simuLinkArray = np.full((10,10,2,1000),0)
Kp = -1
taup = 1
nstep = 1000

Sig = Signal(nstep)
u3 = Sig.pulsed_signal()
t3 = np.linspace(0,10000,nstep)
y3 = odeint(model,0,t3,args=(t3,u3,Kp,taup))

plt.figure(1)
plt.plot(t3,u3,'b--',linewidth=1,label='Input Signal')
plt.plot(t3,y3,'r-',linewidth=1,label='ODE Integrator')
plt.xlabel('Time')
plt.ylabel('Response (y)')
plt.legend(loc='best')
plt.show()
