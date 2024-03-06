""" Example Created by:
Prof. Nicolas Spogis, Ph.D.
Chemical Engineering Department
LinkTree: https://linktr.ee/spogis

Simple Bioreactor with Monod Model
"""

""" Python Libraries"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

""" Kinetics constants"""
mumax = 0.20  # 1/hour  - maximum specific growth rate
Ks = 1.00  # g/liter - half saturation constant

# Yxs = mass of new cells formed / mass of substrate consumed
Yxs = 0.5  # g/g

# Ypx = mass of product formed / mass of product formed
Ypx = 0.2  # g/g

""" Process Variables"""
Total_Fermentation_Time = 72.0  # h

Sf = 10.0  # g/liter - Fed substrate concentration
Flow_Rate = 0.05  # Feed flowrate - l/h


""" Set Reactor Type"""
Reactor_Type = "Semi Batch Bioreactor"

"""  Inlet Flowrate """


def F(t):
    if Reactor_Type == "Semi Batch Bioreactor":
        Fr = Flow_Rate
    else:
        Fr = 0.0
    return Fr


""" Reaction Rates"""


def mu(S):  # Monod Model
    return mumax * S / (Ks + S)


def Rg(X, S):  # Cell Specific Growth
    return mu(S) * X


def Rp(X, S):  # Product Specific Growth -> Assumed to be a by-product of cell growth
    return Ypx * Rg(X, S)


""" Differential Equations (ODEs)"""


def model(t, y):
    X = y[0]  # Cell Concentration
    P = y[1]  # Products Concentration
    S = y[2]  # Substrate Concentration
    V = y[3]  # Reator Volume

    dXdt = -F(t) * X / V + Rg(X, S)
    dPdt = -F(t) * P / V + Rp(X, S)
    dSdt = F(t) * (Sf - S) / V - Rg(X, S) / Yxs
    dVdt = F(t)

    return [dXdt, dPdt, dSdt, dVdt]


""" Initial Conditions"""
X0 = 0.05
P0 = 0.00
S0 = 10.00
V0 = 1.00

Initial_Conditions = [X0, P0, S0, V0]

""" Variable Names - In order to use on Plots"""
Variable_Names = ["Cell Conc.", "Product Conc.", "Substrate Conc.", "Volume [liter]"]

""" Solve ODEs"""
t_eval = np.linspace(0, Total_Fermentation_Time, num=1000, endpoint=True)
sol = solve_ivp(
    model,
    [0, Total_Fermentation_Time],
    Initial_Conditions,
    method="LSODA",
    t_eval=t_eval,
)

""" Generating a More Usable Dataset"""
t = np.transpose(sol.t)
y = np.transpose(sol.y)
df_time = pd.DataFrame({"Time": t})
df_num_sol = pd.DataFrame(y, columns=Variable_Names)
Results = pd.concat([df_time, df_num_sol], axis=1)

""" Plot Results"""
Charts_Dir = "./"

plt.figure(1)
Chart_File_Name = Charts_Dir + "Bioreactor_ODE_Concentrations.png"

plt.plot(Results["Time"], Results["Cell Conc."], label="Cell Conc.")
plt.plot(Results["Time"], Results["Product Conc."], label="Product Conc.")
plt.plot(Results["Time"], Results["Substrate Conc."], label="Substrate Conc.")
plt.legend()
plt.xlabel("Time [hr]")
plt.ylabel("Concentration [g/liter]")
plt.savefig(Chart_File_Name)
plt.show()

plt.figure(2)
Chart_File_Name = Charts_Dir + "Bioreactor_ODE_Volume.png"

plt.plot(Results["Time"], Results["Volume [liter]"], label="Volume [liter]")
plt.legend()
plt.xlabel("Time [hr]")
plt.ylabel("Volume [liter]")
plt.savefig(Chart_File_Name)
plt.show()

""" Export Results for PINN Performance Evaluation"""
Results.to_csv("001-Bioreactor_ODEs_Dataset.csv", index=False)
