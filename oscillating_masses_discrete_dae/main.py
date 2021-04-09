#
#   This file is part of do-mpc
#
#   do-mpc: An environment for the easy, modular and efficient implementation of
#        robust nonlinear model predictive control
#
#   Copyright (c) 2014-2019 Sergio Lucia, Alexandru Tatulea-Codrean
#                        TU Dortmund. All rights reserved
#
#   do-mpc is free software: you can redistribute it and/or modify
#   it under the terms of the GNU Lesser General Public License as
#   published by the Free Software Foundation, either version 3
#   of the License, or (at your option) any later version.
#
#   do-mpc is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU Lesser General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with do-mpc.  If not, see <http://www.gnu.org/licenses/>.
import random
import numpy as np
import matplotlib.pyplot as plt
from casadi import *
from casadi.tools import *
import pdb
import sys
import time
sys.path.append('../../')
import do_mpc
import seaborn as sns

from template_model import template_model
from template_mpc import template_mpc
from template_simulator import template_simulator
from Signal import *


""" User settings: """
show_animation = True
length = 300

"""
Get configured do-mpc modules:
"""

A = np.array([[-0.04,0.1],
              [0.3, -0.09],])

B = np.array([[0.09,  0],
              [0.04,  .05],])

PRBS_gen = Signal(1,1,1)
prbs = PRBS_gen.PRBS()

model = template_model(A,B)
mpc = template_mpc(model,prbs)
simulator = template_simulator(model)
estimator = do_mpc.estimator.StateFeedback(model)

"""
Set initial state
"""
np.random.seed(99)

e = np.full([model.n_x,1],prbs[0])
x0 = e # Values between +3 and +3 for all states
mpc.x0 = x0
simulator.x0 = x0
estimator.x0 = x0

# Use initial state to set the initial guess.
mpc.set_initial_guess()
"""
Run MPC main loop:
"""
u0s= []

for k in range(length):  
    u0 = mpc.make_step(x0)
    y_next = simulator.make_step(u0)
    x0 = estimator.make_step(y_next)
    
mpc_graphics = do_mpc.graphics.Graphics(mpc.data) 

timeData=mpc.data._time
stateData = mpc.data._x
inputData = mpc.data._u

sns.set()
plt.figure(dpi=300,figsize=(10,5))
plt.plot(timeData,stateData[:,1])
plt.plot(timeData,inputData[:,1],'--')

'''
from matplotlib import rcParams
rcParams['axes.grid'] = True
rcParams['font.size'] = 18

fig, ax = plt.subplots(3, sharex=True, figsize=(20,15))
# Configure plot:
mpc_graphics.add_line(var_type='_x', var_name='x', axis=ax[0])
mpc_graphics.add_line(var_type='_u', var_name='u', axis=ax[1])
mpc_graphics.add_line(var_type='_aux', var_name='cost', axis=ax[2])
ax[0].set_ylabel('Controlled Variable (x)')
ax[1].set_ylabel('Manipulated Variable (u)')
ax[2].set_ylabel('Set Point Tracking Error')

# Update properties for all prediction lines:
for line_i in mpc_graphics.pred_lines.full:
    line_i.set_linewidth(2)
# Highlight nominal case:
for line_i in np.sum(mpc_graphics.pred_lines['_x', :, :,0]):
    line_i.set_linewidth(5)
for line_i in np.sum(mpc_graphics.pred_lines['_u', :, :,0]):
    line_i.set_linewidth(5)
for line_i in np.sum(mpc_graphics.pred_lines['_aux', :, :,0]):
    line_i.set_linewidth(5)

# Add labels
label_lines = mpc_graphics.result_lines['_x', 'C_a']+mpc_graphics.result_lines['_x', 'C_b']
ax[0].legend(label_lines, ['C_a', 'C_b'])
label_lines = mpc_graphics.result_lines['_x', 'T_R']+mpc_graphics.result_lines['_x', 'T_K']
ax[1].legend(label_lines, ['T_R', 'T_K'])

fig.align_ylabels()

from matplotlib.animation import FuncAnimation

def update(t_ind):
    print('Writing frame: {}.'.format(t_ind), end='\r')
    mpc_graphics.plot_results(t_ind=t_ind)
    mpc_graphics.plot_predictions(t_ind=t_ind)
    mpc_graphics.reset_axes()
    lines = mpc_graphics.result_lines.full
    return lines

n_steps = mpc.data['_time'].shape[0]

anim = FuncAnimation(fig, update, frames=n_steps, blit=False)
anim.save('discrete_system.gif', writer="imagemagick")
'''