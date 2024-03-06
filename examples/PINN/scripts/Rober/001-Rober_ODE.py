# (C) Copyright IBM Corp. 2019, 2020, 2021, 2022.

#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at

#           http://www.apache.org/licenses/LICENSE-2.0

#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.


"""
Example Created by:
Prof. Nicolas Spogis, Ph.D.
Chemical Engineering Department
LinkTree: https://linktr.ee/spogis

Rober Stiffness 

The Rober problem revolves around a complex system of three nonlinear ordinary 
differential equations that describe the kinetics of an autocatalytic reaction 
as formulated by Robertson (1966). The presence of a substantial discrepancy 
among the reaction rate constants is the underlying cause of stiffness in this 
problem. As is typical for challenges encountered in chemical kinetics, this 
particular system exhibits a brief but intense initial transient phase. 

Following this, the components undergo a gradual and continuous variation, 
making it appropriate to employ a numerical method that can accommodate a 
larger step size.
"""

"""    Import Python Libraries    """

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

"""    Global Variables    """

k1 = 0.04
k2 = 3e7
k3 = 1e4

Simulated_Time = 500

"""    Differential Equations    """


def model(t, y):
    s1 = y[0]
    s2 = y[1]
    s3 = y[2]

    ds1dt = -k1 * s1 + k3 * s2 * s3
    ds2dt = k1 * s1 - k2 * (s2**2) - k3 * s2 * s3
    ds3dt = k2 * (s2**2)

    return [ds1dt, ds2dt, ds3dt]


"""     Initial Conditions    """
s1_0 = 1.0
s2_0 = 0.0
s3_0 = 0.0
Initial_Conditions = [s1_0, s2_0, s3_0]

# In order to use on Plots
Variable_Names = ["s1", "s2", "s3"]

"""     Solve ODE       """

t_eval = np.linspace(0, Simulated_Time, num=1000, endpoint=True)

sol = solve_ivp(
    model, [0, Simulated_Time], Initial_Conditions, method="LSODA", t_eval=t_eval
)

"""     Generating a More Usable Dataset    """

t = np.transpose(sol.t)
y = np.transpose(sol.y)

y = y * np.array([1, 1e4, 1])

df_time = pd.DataFrame({"Time": t})

df_num_sol = pd.DataFrame(y, columns=Variable_Names)

Results = pd.concat([df_time, df_num_sol], axis=1)

"""     Export Results for PINN Performance Evaluation    """

Results.to_csv("001-Rober_ODE_Dataset.csv", index=False)
