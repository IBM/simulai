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

u0 = 0.01

Simulated_Time = 2 / u0

"""    Differential Equations    """


def model(t, y):
    u = y[0]

    dudt = u**2 - u**3

    return [dudt]


"""     Initial Conditions    """

Initial_Conditions = [u0]

# In order to use on Plots
Variable_Names = ["u"]

"""     Solve ODE       """

t_eval = np.linspace(0, Simulated_Time, num=500, endpoint=True)

sol = solve_ivp(
    model, [0, Simulated_Time], Initial_Conditions, method="LSODA", t_eval=t_eval
)

"""     Generating a More Usable Dataset    """

t = np.transpose(sol.t)
y = np.transpose(sol.y)

df_time = pd.DataFrame({"Time": t})

df_num_sol = pd.DataFrame(y, columns=Variable_Names)

Results = pd.concat([df_time, df_num_sol], axis=1)

"""     Export Results for PINN Performance Evaluation    """

Results.to_csv("Fire.csv", index=False)

"""     Plot Results    """

i = 0
for Names in Variable_Names:
    plt.figure(i)
    plt.plot(Results["Time"], Results[Names])
    plt.title(Names)
    plt.xlabel("Time")
    plt.ylabel(Names)
    i = i + 1

"""     Calculate and Plot Derivatives   """

dudt = np.gradient(Results["u"], Results["Time"])

plt.figure(i)
plt.plot(Results["Time"], dudt)
plt.title("dudt")
plt.xlabel("Time")
plt.ylabel("dudt")
i = i + 1

d2udt2 = np.gradient(np.gradient(Results["u"], Results["Time"]), Results["Time"])

plt.figure(i)
plt.plot(Results["Time"], d2udt2)
plt.title("d2udt2")
plt.xlabel("Time")
plt.ylabel("d2udt2")
