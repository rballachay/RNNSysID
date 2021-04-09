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

import numpy as np
from casadi import *
from casadi.tools import *
import pdb
import sys
sys.path.append('../../')
import do_mpc


def template_mpc(model,structure):
    """
    --------------------------------------------------------------------------
    template_mpc: tuning parameters
    --------------------------------------------------------------------------
    """
    mpc = do_mpc.controller.MPC(model)

    setup_mpc = {
        'n_robust': 0,
        'n_horizon': 7,
        't_step': 1,
        'state_discretization': 'discrete',
        'store_full_solution':True,
    }

    mpc.set_param(**setup_mpc)

    mterm = model.aux['cost']
    lterm = model.aux['cost'] # terminal cost

    mpc.set_objective(mterm=mterm, lterm=lterm)
    mpc.set_rterm(u=1e-4)


    # The timevarying paramters have no effect on the simulator (they are only part of the cost function).
    # We simply use the default values:
    tvp_template = mpc.get_tvp_template()
    
    def tvp_fun(t_now):
        for k in range(7+1):
            tvp_template['_tvp',k,'x_set_point'] = structure[int(t_now)]
        return tvp_template
    
    mpc.set_tvp_fun(tvp_fun)
    
    max_x = np.array([[100000], [100000]])

    mpc.bounds['lower','_x','x'] = -max_x[0]
    mpc.bounds['upper','_x','x'] =  max_x[0]

    mpc.bounds['lower','_u','u'] = -1000000
    mpc.bounds['upper','_u','u'] = 1000000


    mpc.setup()

    return mpc
