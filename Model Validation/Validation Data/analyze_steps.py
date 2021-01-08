#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 14:40:42 2020

@author: RileyBallachay
"""

import numpy as np
import pandas as pd
import glob
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(color_codes=True)

csvs = glob.glob('*.csv')

all_data = pd.read_csv(csvs[0])
#all_data = all_data[all_data['theta Std']<.5]

columns = all_data.columns

for csv in csvs[1:]:
    df = pd.read_csv(csv)
    all_data =  all_data.append(df)

all_data = all_data[all_data['Mean SSE']<5000]
all_data['diff'] = abs(all_data['tau'] - all_data['tau Pred'])
#all_data = all_data[all_data['diff']<.01]
#all_data = all_data[all_data['Kp Std']<0.1]

#all_data['Kp Std'] = np.log(all_data['Kp Std'])

sns.lmplot(x='Mean SSE',y='Gauss',data=all_data,aspect=2)
#sns.barplot(x='Gauss',y='Mean SSE',data=all_data)