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
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
sns.set_theme(color_codes=True)

csvs = glob.glob('*.csv')

all_data = pd.read_csv(csvs[-1])
#all_data = all_data[all_data['theta Std']<.5]

columns = all_data.columns
'''
for csv in csvs[1:]:
    df = pd.read_csv(csv)
    all_data =  all_data.append(df)
'''
#all_data = all_data[all_data['Mean SSE']<1000]

keys = ['a','b','k']
vals = ['tau','Kp','theta']

d = dict(zip(keys,vals))

fig, axes = plt.subplots(1, 3,figsize=(15,5),dpi=400)  
for (idx,key) in enumerate(keys):

    all_data[key+' Estimate Error'] = abs(all_data[d[key]] - all_data[d[key]+' Pred'])
    all_data[key+' Estimated Uncertainy'] = all_data[d[key]+' Std']*2
    
    #all_data = all_data[all_data[key+' Estimate Error']<0.15]
    
    ax = axes[idx]
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    ax.plot(all_data[key+' Estimated Uncertainy'],all_data[key+' Estimate Error'],'.')
    r2 =("r\u00b2 = %.3f" % r2_score(all_data['Mean SSE'],all_data[key+' Estimate Error']))
    print(r2)
    #ax.legend()

for ax in axes.flat:
    ax.label_outer()
    ax.xaxis.grid(False)
    ax.yaxis.grid(False)
    
plt.ylabel("Predicted Parameter Value")
plt.xlabel("True Parameter Value")
            
'''   
for parameter in parameters: 
    all_data['Kp Prediction Error'] = abs(all_data['Kp'] - all_data['Kp Pred'])
all_data = all_data[all_data['diff']<.2]
#all_data = all_data[all_data['Kp Std']<0.1]

all_data['Kp Stdev']


#all_data['Kp Std'] = np.log(all_data['Kp Std'])
all_data['Average Parameter Uncertainty'] = (all_data['Kp Std']+all_data['tau Std']+all_data['theta Std'] / 3)**1
all_data['MSE of Prediction'] = np.sqrt(all_data['Mean SSE']/600)

#corr, _ = pearsonr(all_data['Average Parameter Uncertainty'],all_data['MSE of Prediction'])
sns.lmplot(x='Kp Std',y='diff',data=all_data,aspect=1.2)
plt.text(.2, .1, "text on plot")

#sns.barplot(x='Gauss',y='Mean SSE',data=all_data)
'''