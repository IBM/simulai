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

Rober Reaction - Multifidelity Model
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

"""    Variables    """
N = 100
n = 100

t_max = 500
n_intervals = 500
delta_t = t_max/n_intervals

"""    Initial Condition    """    
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
      net.summary()
          
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
        
        t_max = 500
        n_intervals = 500
        delta_t = t_max/n_intervals
        
        depth = 3
        width = 50
        activations_funct = "tanh"
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
            net.summary()
                
            return net

        # Initialization for the multifidelity network
        models_list = [sub_model() for i in range(n_intervals)]

        # Prototype for the multifidelity network
        class MultiNetwork(NetworkTemplate):

            def __init__(self, models_list:List[NetworkTemplate]=None,
                         delta_t:float=None, device:str='cpu') -> None:

                super(MultiNetwork, self).__init__()

                for i, model in enumerate(models_list):
                    self.set_network(net=model, index=i)

                self.delta_t = delta_t
                self.device = device

            def set_network(self, net:NetworkTemplate=None, index:int=None) -> None:

                setattr(self, f"worker_{index}", net)

                self.add_module(f"worker_{index}", net)

            def _eval_interval(self, index:int=None, input_data:torch.Tensor=None) -> torch.Tensor:

                input_data = input_data[:, None]
                return getattr(self, f"worker_{index}").eval(input_data=input_data)

            def eval(self, input_data:np.ndarray=None) -> np.ndarray:
 
                eval_indices_float = input_data/delta_t
                eval_indices = np.where(eval_indices_float > 0,
                                        np.floor(eval_indices_float - 1e-13).astype(int),
                                        eval_indices_float.astype(int))

                eval_indices = eval_indices.flatten().tolist()

                input_data = input_data - self.delta_t*np.array(eval_indices)[:, None]

                return np.vstack([self._eval_interval(index=i, input_data=idata) \
                                                     for i, idata in zip(eval_indices, input_data)])

            def summary(self):

                print(self)

        multi_net = MultiNetwork(models_list=models_list, delta_t=delta_t)

        return multi_net

    net_ = model()
    multi_net = model_()

    optimizer_config = {"lr": lr}
    optimizer = Optimizer("adam", params=optimizer_config)

    time_plot = np.empty((0, 1), dtype=float)
    approximated_data_plot = np.empty((0, 1), dtype=float)
    time_eval_plot = np.empty((0, 1), dtype=float)

    ### Run Multifidelity Model
    for i in range(0, int(n_intervals), 1):
        net = net_

        time_train = np.linspace(0, delta_t, n)[:, None]
        time_eval = np.linspace(0, delta_t, n)[:, None]

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
            "weights_residual": [1, 1, 1, 1],
            "weights":  [1, 1e6, 1],        # Maximum derivative magnitudes to be used as loss weights
            "initial_penalty": 1e8,
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

        # Get Last PINN Value and Update as Initial Condition for the Next PINN
        state_t = approximated_data[-1]

        # Initializing the nest sub-model using the current one
        net_ = model()
        state_dict = net.state_dict()
        net_.load_state_dict(state_dict)

        # Storing the current network in the multifidelity
        # model
        multi_net.set_network(net=net, index=i)

    saver = SPFile(compact=False)
    saver.write(
        save_dir='./',
        name="multi_fidelity_rober_pinn",
        model=multi_net,
        template=model_,
    )

# Restoring the model from disk and using it for making 
# evaluations
else:
    saver = SPFile(compact=False)
    multi_net = saver.read(model_path='./multi_fidelity_rober_pinn', device='cpu')

    input_labels = ["t"]
    output_labels = ["s1", "s2", "s3"]

    multi_net.summary()

    time_plot = np.linspace(0, t_max, 1000)[:, None]

    approximated_data_plot = multi_net.eval(input_data=time_plot)
    
    # Scale data (s2*1e4) only to help visualization
    approximated_data_plot = approximated_data_plot * np.array([1, 1e4, 1])
    
    df = pd.DataFrame(approximated_data_plot, columns = output_labels)
    # Plot Result
    Charts_Dir= './'

    # Compare PINN and EDO Results
    ODE_Results = pd.read_csv("Rober_ODE.csv")
    Filter_Scale = 20
    ODE_Results_OnlyFewData = ODE_Results[::Filter_Scale]

    plt.figure(1)
    Chart_File_Name = Charts_Dir + 'Rober_ODE_Multifidelity_PINN_Comparison.png'
    
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