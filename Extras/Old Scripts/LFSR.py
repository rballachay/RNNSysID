#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 12:22:28 2020

@author: RileyBallachay
"""
import numpy as np
from pylfsr import LFSR

L = LFSR(fpoly=[23,18],initstate ='random',verbose=True)
L.info()
max_sample = 5
L.runKCycle(1000)
L.info()
seq = L.seq

G = np.zeros(max_sample*1000)
