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

Simple Bioreactor with Monod Model
"""

"""    Import Python Libraries    """
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from simulai.optimization import Optimizer
from simulai.residuals import SymbolicOperator

"""    Variables    """
N = 100
n = 100
Evaluation_Steps = 72
Plots_Steps = 10

def Delta_t(iteration_number):
    # if iteration_number<=10:
    #     Delta_t = 1e-7
    # elif iteration_number>10 and iteration_number<=20:
    #     Delta_t = 1e-6
    # elif iteration_number>20 and iteration_number<=200:
    #     Delta_t = 1e-2
    # elif iteration_number>200 and iteration_number<=300:
    #     Delta_t = 1e-1
    # else:
    #     Delta_t = 1e-0
    Delta_t = 1e-0
    print("\n Delta t:", Delta_t)
    return Delta_t

n_epochs_ini = 5_000    # Maximum number of iterations for ADAM
n_epochs_min = 500      # Minimum number of iterations for ADAM
Epoch_Tau = 3.0         # Number o Epochs Decay
lr = 1e-3               # Initial learning rate for the ADAM algorithm
def Epoch_Decay(iteration_number):
    if iteration_number<100:
        n_epochs_iter = n_epochs_ini*(np.exp(-iteration_number/Epoch_Tau))
        n_epochs = int(max(n_epochs_iter, n_epochs_min))
    else:
        n_epochs = n_epochs_min
    print("N Epochs:", n_epochs)
    print("Iteration:", iteration_number)
    return n_epochs

    
input_labels = ["t"]
output_labels = ["X_C", "P_C", "S_C", "Vol"]

n_inputs = len(input_labels)
n_outputs = len(output_labels)

"""    Initial_Conditions """ 
X0 = 0.05
P0 = 0.00
S0 = 10.00
V0 = 1.00

state_t = np.array([X0, P0, S0, V0])
state_t_old = np.array([state_t])

"""    Kinetics constants """ 
mumax = 0.20      # 1/hour  - maximum specific growth rate
Ks = 1.00         # g/liter - half saturation constant

# Yxs = mass of new cells formed / mass of substrate consumed
Yxs = 0.5         # g/g

# Ypx = mass of product formed / mass of product formed
Ypx = 0.2         # g/g

Sf = 10.0         # g/liter - Fed substrate concentration
Flow_Rate = 0.05  # Feed flowrate - l/h

"""    The expressions we aim at minimizing """ 
f_X_C = "D(X_C, t) - ((-Flow_Rate*X_C/Vol)     +         (mumax*S_C/(Ks + S_C))*X_C)"
f_P_C = "D(P_C, t) - ((-Flow_Rate*P_C/Vol)     +     Ypx*(mumax*S_C/(Ks + S_C))*X_C)"
f_S_C = "D(S_C, t) - ((Flow_Rate*(Sf-S_C)/Vol) - (1/Yxs)*(mumax*S_C/(Ks + S_C))*X_C)"
f_Vol = "D(Vol, t) - (Flow_Rate)"

depth = 3
width = 50
activations_funct = "tanh"
def model():
  from simulai.regression import SLFNN, ConvexDenseNetwork
  from simulai.models import ImprovedDenseNetwork

  scale_factors = np.array([1, 1, 1, 1])

  # Configuration for the fully-connected network
  config = {
      "layers_units": depth * [width],               # Hidden layers
      "activations": activations_funct,
      "input_size": n_inputs,
      "output_size": n_outputs,
      "name": "net"}
      
  #Instantiating and training the surrogate model
  densenet = ConvexDenseNetwork(**config)
  encoder_u = SLFNN(input_size=1, output_size=width, activation=activations_funct)
  encoder_v = SLFNN(input_size=1, output_size=width, activation=activations_funct)

  class ScaledImprovedDenseNetwork(ImprovedDenseNetwork):
    
    def __init__(self, network=None, encoder_u=None, encoder_v=None, devices="gpu", scale_factors=None):
        
        super(ScaledImprovedDenseNetwork, self).__init__(network=densenet, encoder_u=encoder_u, encoder_v=encoder_v, devices="gpu")
        self.scale_factors = torch.from_numpy(scale_factors.astype("float32")).to(self.device)
        
    
    def forward(self, input_data=None):
        
        return super().forward(input_data)*self.scale_factors
    
  net = ScaledImprovedDenseNetwork(network=densenet, encoder_u=encoder_u, encoder_v=encoder_v, devices="gpu", scale_factors=scale_factors)

  # It prints a summary of the network features
  net.summary()
      
  return net

net = model()

optimizer_config = {"lr": lr}

optimizer = Optimizer("adamw", params=optimizer_config)

residual = SymbolicOperator(
    expressions= [f_X_C, f_P_C, f_S_C, f_Vol],
    input_vars= input_labels,
    output_vars= output_labels,
    function=net,
    constants={"mumax": mumax,
               "Ks": Ks,
               "Yxs": Yxs,
               "Ypx": Ypx,
               "Sf": Sf,
               "Flow_Rate": Flow_Rate,
               },
    engine="torch",
    device="gpu",
)

time_plot = np.empty((0,1), dtype=float)
approximated_data_plot = np.empty((0,n_outputs), dtype=float)
time_eval_plot = np.empty((0,1), dtype=float)

Next_Time = 0 

"""    From here we run the time deltas     """    
for i in range (0, Evaluation_Steps, 1):
    get_Delta_t = Delta_t(i)
    time_train = np.linspace(0, get_Delta_t, n)[:, None]
    time_eval  = np.linspace(0, get_Delta_t, n)[:, None]
    
    initial_state = np.array([state_t])
    
    params = {
        "residual": residual,
        "initial_input": np.array([0])[:, None],
        "initial_state": initial_state,
        "weights_residual": [1, 1, 1, 1],
        "initial_penalty": 1e8,
    }
    
    get_n_epochs = Epoch_Decay(i)
    optimizer.fit(
        op=net,
        input_data= time_train,
        n_epochs= get_n_epochs,
        loss="pirmse",
        params=params,
        device="gpu",
    )
    
    
    from simulai.optimization import ScipyInterface
    from simulai.optimization import PIRMSELoss
    
    loss_instance = PIRMSELoss(operator=net)
    
    optimizer_lbfgs = ScipyInterface(
        fun=net,
        optimizer="L-BFGS-B",
        loss=loss_instance,
        loss_config=params,
        optimizer_config={
            "options": {
                "maxiter": 50000,
                "maxfun":  50000,
                "maxcor":  50,
                "maxls":   50,
                "ftol": 1.0 * np.finfo(float).eps,
                "eps": 1e-6,}
            },
        )
    
    optimizer_lbfgs.fit(input_data=time_train)
    
    #Evaluation in training dataset
    approximated_data = net.eval(input_data=time_eval)
    state_t = approximated_data[-1]
    
    time_eval_plot  = np.linspace(Next_Time, Next_Time + get_Delta_t, N)[:, None]
    time_plot = np.vstack((time_plot,time_eval_plot))
    approximated_data_plot = np.vstack((approximated_data_plot, approximated_data))
    Next_Time = Next_Time + get_Delta_t
    
    Dataframe_Labels = ["X_C", "P_C", "S_C", "Vol"]
    df = pd.DataFrame(approximated_data_plot , columns = Dataframe_Labels)
    
    delta_state_t =  state_t - state_t_old
    grad_state_t = delta_state_t/get_Delta_t
    state_t_old = state_t
    
    if i % Plots_Steps == 0:
        # Plot Result   
        plt.figure(1)
        plt.plot(time_plot, df['X_C'], label='PINN - Cell Conc.')
        plt.plot(time_plot, df['P_C'], label='PINN - Product Conc.')
        plt.plot(time_plot, df['S_C'], label='PINN - Substrate Conc.')
        
        plt.legend()
        plt.xlabel('Time [hr]')
        plt.ylabel('Concentration [g/liter]')
        plt.show()
        
        plt.figure(2)
        plt.plot(time_plot, df['Vol'], label='PINN - Volume [liter]')
        plt.legend()
        plt.xlabel('Time [hr]')
        plt.ylabel('Volume [liter]')
        plt.show()
     
# Plot Result
Charts_Dir= './'
 
# Compare PINN and EDO Results
ODE_Results = pd.read_csv("Bioreactor_ODEs.csv")
Filter_Scale = 50
ODE_Results_OnlyFewData = ODE_Results[::Filter_Scale]

plt.figure(1)
Chart_File_Name = Charts_Dir + 'Bioreactor_ODE_PINN_Concentration_Comparison.png'
 
plt.scatter(ODE_Results_OnlyFewData["Time"], ODE_Results_OnlyFewData["Cell Conc."], label='ODE - Cell Conc.')
plt.scatter(ODE_Results_OnlyFewData["Time"], ODE_Results_OnlyFewData["Product Conc."], label='ODE - Product Conc.')
plt.scatter(ODE_Results_OnlyFewData["Time"], ODE_Results_OnlyFewData["Substrate Conc."], label='ODE - Substrate Conc.')
plt.plot(time_plot, df['X_C'], label='PINN - Cell Conc.')
plt.plot(time_plot, df['P_C'], label='PINN - Product Conc.')
plt.plot(time_plot, df['S_C'], label='PINN - Substrate Conc.')
 
plt.legend()
plt.xlabel('Time [hr]')
plt.ylabel('Concentration [g/liter]')
plt.savefig(Chart_File_Name)
plt.show()
 
plt.figure(2)
Chart_File_Name = Charts_Dir + 'Bioreactor_ODE_PINN_Volume_Comparison.png'
 
plt.scatter(ODE_Results_OnlyFewData["Time"], ODE_Results_OnlyFewData["Volume [liter]"], label='ODE - Volume [liter]')
plt.plot(time_plot, df['Vol'], label='PINN - Volume [liter]')
 
plt.legend()
plt.xlabel('Time [hr]')
plt.ylabel('Volume [liter]')
plt.savefig(Chart_File_Name)
plt.show()
 
"""     Generating a More Usable Dataset    """
time_plot = time_plot.flatten()
df_time = pd.DataFrame({'Time': time_plot})
df_num_sol = pd.DataFrame(approximated_data_plot , columns = output_labels)
Results = pd.concat([df_time,df_num_sol], axis=1)

"""     Export Results for PINN Performance Evaluation    """
Results.to_csv("Bioreactor_PINN.csv", index=False)
