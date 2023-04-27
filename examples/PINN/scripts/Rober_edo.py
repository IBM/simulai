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

t_eval = np.linspace(0, Simulated_Time, num=50000, endpoint=True)

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

Results.to_csv("Rober_EDO.csv", index=False)

"""     Plot Results    """

df_PINN = pd.read_csv("Rober_PINN.csv")

plt.figure(2)
plt.plot(Results["Time"], Results["s1"], label="s1 EDO")
plt.plot(Results["Time"], Results["s2"], label="s2 EDO")
plt.plot(Results["Time"], Results["s3"], label="s3 EDO")
plt.plot(df_PINN["Time"], df_PINN["s1"], label="s1 PINN")
plt.plot(df_PINN["Time"], df_PINN["s2"], label="s2 PINN")
plt.plot(df_PINN["Time"], df_PINN["s3"], label="s3 PINN")
plt.legend()
plt.xlabel("Time")
