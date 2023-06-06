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

Rober Multifidelity Auto Time Step
"""

"""    Import Python Libraries    """
from argparse import ArgumentParser
from typing import List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import torch 
torch.set_default_dtype(torch.float64)

from simulai.optimization import Optimizer, PIRMSELoss, ScipyInterface
from simulai.residuals import SymbolicOperator
from simulai.templates import NetworkTemplate, guarantee_device
from simulai.file import SPFile
from simulai.optimization import AnnealingWeights, InverseDirichletWeights, PIInverseDirichlet

"""    Variables    """
# Bioreactor
N = 100
n = 100

t_max = 500

"""    Initial_Conditions """ 
s1_0 = 1.0
s2_0 = 0.0
s3_0 = 0.0

state_t=np.array([s1_0, s2_0, s3_0])

# Will we train from scratch or restore a 
# pre-trained model from disk ?
parser = ArgumentParser(description="Reading input parameters")
parser.add_argument("--train", type=str, help="Training new model or restoring from disk", default="yes")
args = parser.parse_args()

train = args.train

train = "yes"

if train == "yes":

    """    Kinetics constants """    
    k1 = 0.04
    k2 = 3e7
    k3 = 1e4

    """    The expressions we aim at minimizing    """    
    f_s1 = "D(s1, t) + k1*s1 - k3*s2*s3"
    f_s2 = "D(s2, t) - k1*s1 + k2*(s2**2) + k3*s2*s3"
    f_s3 = "D(s3, t) - k2*(s2**2)"

    input_labels = ["t"]
    output_labels = ["s1", "s2", "s3"]

    n_inputs = len(input_labels)
    n_outputs = len(output_labels)

    n_epochs_ini = 2_000    # Maximum number of iterations for ADAM
    n_epochs_min = 500      # Minimum number of iterations for ADAM
    Epoch_Tau = 5.0         # Number o Epochs Decay
    lr = 5e-4               # Initial learning rate for the ADAM algorithm

    def Epoch_Decay(iteration_number):
        if iteration_number<100:
            n_epochs_iter = n_epochs_ini*(np.exp(-iteration_number/Epoch_Tau))
            n_epochs = int(max(n_epochs_iter, n_epochs_min))
        else:
            n_epochs = n_epochs_min
        print("N Epochs:", n_epochs)
        print("Iteration:", iteration_number+1)
        return n_epochs
    
    
    """ Adaptive Time Step """
    tol = 1e-05                     # Truncation Error Tolerance  
    def Delta_t(i, last_delta_t):
        dt_init = 1e-03             # Initial Time Step Size
        dt_min = 5e-04              # Minimum Time Step Size
        dt_max = 2.0                # Maximum Time Step Size
        Safety_Factor = 0.9         # Safety Factor
        Max_SCF = 2.0               # Maximum Step Change Factor 
        Min_SCF = 0.3               # Minimum Step Change Factor 
        Change_Iteration = 5        # Fixed Time Steps
           
        i = i+1
        if i<Change_Iteration+1:
            dt_sug = dt_init
            print("Number of Fixed Time Steps:", Change_Iteration)
            print("Fixed Delta t:", dt_sug)
        else:
            LastLoss = optimizer.loss_states['pde']
            LastLoss = LastLoss[-1]
            Adams_Moulton_Step =((270/19)*(tol*last_delta_t)/(LastLoss))*(1/4)
            Adams_Ratio = Adams_Moulton_Step/last_delta_t
            dt_sug = Safety_Factor*last_delta_t*max(min(Adams_Ratio,Max_SCF),Min_SCF)
            print("Suggested new Delta t:", dt_sug)

        Delta_t = min(max(dt_sug,dt_min),dt_max)    
        print("Next Delta t (resp. Min and Max):", Delta_t)
        return Delta_t


    # Local model, which will be replicated for each sub-domain
    depth = 3
    width = 50
    activations_funct = "tanh"

    def model():
      from simulai.regression import SLFNN, ConvexDenseNetwork
      from simulai.models import ImprovedDenseNetwork

      scale_factors = np.array([1, 1e-4, 1])

      # Configuration for the fully-connected network
      config = {
          "layers_units": depth * [width],               # Hidden layers
          "activations": activations_funct,
          "input_size": 1,
          "output_size": 3,
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
      # net.summary()

      return net

    # Multifidelity network, a composite model with multiple sub-domain 
    # networks
    def model_():

        from typing import List

        import torch 
        torch.set_default_dtype(torch.float64)

        from simulai.templates import NetworkTemplate, guarantee_device
        import numpy as np
        from simulai.models import ImprovedDenseNetwork
        from simulai.regression import SLFNN, ConvexDenseNetwork

        depth = 3
        width = 50
        activations_funct = "tanh"
        n_intervals = 500 # overestimated

        # Model used for initialization
        def sub_model():
            from simulai.regression import SLFNN, ConvexDenseNetwork
            from simulai.models import ImprovedDenseNetwork

            scale_factors = np.array([1, 1e-4, 1])

            # Configuration for the fully-connected network
            config = {
                "layers_units": depth * [width],               # Hidden layers
                "activations": activations_funct,
                "input_size": 1,
                "output_size": 3,
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
            # net.summary()

            return net

        # Initialization for the multifidelity network
        models_list = [sub_model() for i in range(n_intervals)]

        # Prototype for the multifidelity network
        class MultiNetwork(NetworkTemplate):

            def __init__(self, models_list:List[NetworkTemplate]=None,
                         device:str='cpu') -> None:

                super(MultiNetwork, self).__init__()

                for i, model in enumerate(models_list):
                    self.set_network(net=model, index=i)

                self.delta_t = torch.nn.Parameter(data=torch.zeros(len(models_list)),
                                                  requires_grad=False).to(device)

                self.device = device

            def set_delta_t(self, delta_t:float, index:int=None) -> None:

                getattr(self, "delta_t").data[index] = delta_t

            def set_network(self, net:NetworkTemplate=None, index:int=None) -> None:

                setattr(self, f"worker_{index}", net)

                self.add_module(f"worker_{index}", net)

            def _eval_interval(self, index:int=None, input_data:torch.Tensor=None) -> torch.Tensor:

                input_data = input_data[:, None]
                return getattr(self, f"worker_{index}").eval(input_data=input_data)

            @property
            def time_intervals(self):

                cumulated_time = [0] + np.cumsum(self.delta_t).tolist()

                return np.array([ [ cumulated_time[i], cumulated_time[i+1] ] for i in
                                 range(len(cumulated_time) -1)])

            def _interval_delimiter(self, value:float, lower:float,
                                    upper:float) -> bool:

                if lower <= value[0] < upper:
                    return True
                else:
                    return False

            def eval(self, input_data:np.ndarray=None) -> np.ndarray:

                n_samples = input_data.shape[0]

                time_intervals = self.time_intervals

                eval_indices = list()
                for j in range(n_samples):
                    is_in_interval = [self._interval_delimiter(input_data[j], low, up) for low,
                     up in time_intervals.tolist()]
                    index = is_in_interval.index(True)
                    eval_indices.append(index)

                cumulated_time = np.array([0] + np.cumsum(self.delta_t).tolist())

                input_data = input_data - cumulated_time[np.array(eval_indices)][:, None]

                return np.vstack([self._eval_interval(index=i, input_data=idata) \
                                                     for i, idata in zip(eval_indices, input_data)])

            def summary(self):

                print(self)

        multi_net = MultiNetwork(models_list=models_list)

        return multi_net

    net_ = model()
    multi_net = model_()

    optimizer_config = {"lr": lr}
    optimizer = Optimizer("adam", params=optimizer_config)

    time_plot = np.empty((0, 1), dtype=float)
    approximated_data_plot = np.empty((0, 1), dtype=float)
    time_eval_plot = np.empty((0, 1), dtype=float)

    ### Run Multifidelity Model
    i = 0
    t_acu = 0
    Not_Acepted_Steps = 0
    last_delta_t = 0.01
    get_Delta_t = Delta_t(i, last_delta_t)
    
    while t_acu < t_max:
        last_delta_t = get_Delta_t
        get_Delta_t = Delta_t(i, last_delta_t)
        
        net = net_

        time_train = np.linspace(0, get_Delta_t, n)[:, None]
        time_eval = np.linspace(0, get_Delta_t, n)[:, None]

            # Simple model of flame growth
        initial_state = np.array([state_t])

        residual = SymbolicOperator(
            expressions=[f_s1, f_s2, f_s3],
            input_vars=["t"],
            output_vars=["s1", "s2", "s3"],
            function=net,
            constants={"k1": k1, "k2": k2, "k3": k3},
            engine="torch",
            device="gpu",
        )

        params = {
            "residual": residual,
            "initial_input": np.array([0])[:, None],
            "initial_state": initial_state,
            "weights_residual": [1, 1, 1],
            "weights":  [1, 1e6, 1],        # Maximum derivative magnitudes to be used as loss weights
            "residual_weights_estimator": PIInverseDirichlet(alpha=0.5,
                                                             n_residuals=3),
            #"global_weights_estimator": InverseDirichletWeights(alpha=0.9), #AnnealingWeights(alpha=0.5),
            "initial_penalty": 1,
        }

        # Reduce Epochs for sequential PINNs
        get_n_epochs = Epoch_Decay(i)

        # First Evaluation With ADAM Optimizer
        optimizer.fit(
            op=net,
            input_data=time_train,
            n_epochs=get_n_epochs,
            loss="pirmse",
            params=params,
            device="gpu",
        )

        # Seccond Evaluation With L-BFGS-B
        loss_instance = PIRMSELoss(operator=net)

        optimizer_lbfgs = ScipyInterface(
            fun=net,
            optimizer="L-BFGS-B",
            loss=loss_instance,
            loss_config=params,
            optimizer_config={
                "options": {
                    "maxiter": 50000,
                    "maxfun": 50000,
                    "maxcor": 50,
                    "maxls": 50,
                    "ftol": 1.0 * np.finfo(float).eps,
                    "eps": 1e-6,
                }
            },
        )

        optimizer_lbfgs.fit(input_data=time_train)

        # Evaluation in training dataset
        approximated_data = net.eval(input_data=time_eval)
                
        LastLoss = optimizer.loss_states['pde']
        LastLoss = LastLoss[-1]
        
        if LastLoss>tol:
            Not_Acepted_Steps += 1
            print("\nSolution Not Accepted")
            print("Last Loss:", LastLoss)
            print("Minimum Tol.:", tol)
            print("Run again with reduced time step")
        else:
            print("\nSolution Accepted")
            print("Run next time step")
            # Get Last PINN Value and Update as Initial Condition for the Next PINN
            state_t = approximated_data[-1]
    
            # Initializing the nest sub-model using the current one
            net_ = model()
            state_dict = net.state_dict()
            net_.load_state_dict(state_dict)
    
            # Storing the current network in the multifidelity
            # model
            multi_net.set_network(net=net, index=i)
            multi_net.set_delta_t(delta_t=get_Delta_t, index=i)
    
            i += 1
            t_acu += get_Delta_t
            print("Simulated Time:", t_acu)
    
    print("Number of Non Acepted Steps:", Not_Acepted_Steps)
    saver = SPFile(compact=False)
    saver.write(
        save_dir='./',
        name="adaptative_multifidelity_rober_pinn",
        model=multi_net,
        template=model_,
    )

# Restoring the model from disk and using it for making 
# evaluations
else:
    saver = SPFile(compact=False)
    multi_net = saver.read(model_path='./adaptative_multifidelity_rober_pinn', device='cpu')

    input_labels = ["t"]
    output_labels = ["s1", "s2", "s3"]

    multi_net.summary()

    compare = False

    time_plot = np.linspace(0, t_max, 1000)[:, None]

    approximated_data_plot = multi_net.eval(input_data=time_plot)
    
    # Scale data (s2*1e4) only to help visualization
    approximated_data_plot = approximated_data_plot * np.array([1, 1e4, 1])
    
    df = pd.DataFrame(approximated_data_plot, columns = output_labels)
    # Plot Result
    Charts_Dir= './'

    # Compare PINN and EDO Results
    if compare:
        ODE_Results = pd.read_csv("Rober_ODE.csv")
        Filter_Scale = 50
        ODE_Results_OnlyFewData = ODE_Results[::Filter_Scale]

    plt.figure(1)
    Chart_File_Name = Charts_Dir + 'Rober_ODE_Multifidelity_PINN_Comparison.png'
    if compare: 
        plt.scatter(ODE_Results_OnlyFewData["Time"], ODE_Results_OnlyFewData["s1"], s=20, label='ODE - s1')
        plt.scatter(ODE_Results_OnlyFewData["Time"], ODE_Results_OnlyFewData["s2"], s=20, label='ODE - s2 (*1e4)')
        plt.scatter(ODE_Results_OnlyFewData["Time"], ODE_Results_OnlyFewData["s3"], s=20, label='ODE - s3')

    plt.plot(time_plot, df['s1'], label='PINN - s1')
    plt.plot(time_plot, df['s2'], label='PINN - s2 (*1e4)')
    plt.plot(time_plot, df['s3'], label='PINN - s3')
    
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Concentration')
    plt.savefig(Chart_File_Name)
    plt.show()
