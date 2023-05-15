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

Simple model of flame growth
Consider the scenario where you ignite a match, leading to the formation of a fiery ball 
that swiftly increases in size until it hits a certain threshold. Once it reaches this 
critical point, the ball of flame stabilizes, as the oxygen consumed by the combustion 
inside the ball equals the amount of oxygen available through the surface.
 
A simplified model of this phenomenon can be described as follows:
    du/dt = u**2 − u**3

The given problem involves u, which denotes the radius of the fiery sphere with respect to 
the critical radius. 

The terms u^2 and u^3 arise from the surface area and volume of the sphere, respectively. 
The problem can be solved over a duration of time that is inversely proportional to u0.
"""

"""    Import Python Libraries    """
from argparse import ArgumentParser
from typing import List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch 
torch.set_default_dtype(torch.float64)

from simulai.optimization import Optimizer, PIRMSELoss, ScipyInterface
from simulai.residuals import SymbolicOperator
from simulai.templates import NetworkTemplate, guarantee_device
from simulai.file import SPFile

"""    Variables    """
N = 200  # Number of Evaluation steps
n = 200  # Number of Training steps
u0 = 0.01  # Initial Flame Size
t_max = 2 / u0  # Total Simulated TIme
intervals = 50  # Number of Steps (Sequential PINNs)
delta_t = t_max / intervals  # PINNs Evaluation Time

state_t = u0  # Initial Condition

# Will we train from scratch or restore a 
# pre-trained model from disk ?
parser = ArgumentParser(description="Reading input parameters")
parser.add_argument("--train", type=bool, help="Training new model or restoring from disk", default="yes")
args = parser.parse_args()

train = args.train

if train == "yes":

    """    The expression we aim at minimizing    """
    f = "D(u, t) - (u**2 - u**3)"

    input_labels = ["t"]
    output_labels = ["u"]

    n_inputs = len(input_labels)
    n_outputs = len(output_labels)

    lr = 1e-3  # Initial learning rate for the ADAM algorithm

    # Local model, which will be replicated for each sub-domain
    def model():

        from simulai.models import ImprovedDenseNetwork
        from simulai.regression import SLFNN, ConvexDenseNetwork

        # Configuration for the fully-connected network
        config = {
            "layers_units": [50, 50, 50],
            "activations": "tanh",
            "input_size": 1,
            "output_size": 1,
            "name": "net",
        }

        # Instantiating and training the surrogate model
        densenet = ConvexDenseNetwork(**config)
        encoder_u = SLFNN(input_size=1, output_size=50, activation="tanh")
        encoder_v = SLFNN(input_size=1, output_size=50, activation="tanh")

        net = ImprovedDenseNetwork(
            network=densenet,
            encoder_u=encoder_u,
            encoder_v=encoder_v,
            devices="gpu",
        )

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

        delta_t = 4
        n_intervals = 50

        # Model used for initialization
        def sub_model():

            # Configuration for the fully-connected network
            config = {
                "layers_units": [50, 50, 50],
                "activations": "tanh",
                "input_size": 1,
                "output_size": 1,
                "name": "net",
            }

            # Instantiating and training the surrogate model
            densenet = ConvexDenseNetwork(**config)
            encoder_u = SLFNN(input_size=1, output_size=50, activation="tanh")
            encoder_v = SLFNN(input_size=1, output_size=50, activation="tanh")

            net = ImprovedDenseNetwork(
                network=densenet,
                encoder_u=encoder_u,
                encoder_v=encoder_v,
                devices="gpu",
            )

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

                input_data = np.array([input_data])[:, None]
                return getattr(self, f"worker_{index}").eval(input_data=input_data)

            def eval(self, input_data:np.ndarray=None) -> np.ndarray:
 
                eval_indices_float = input_data/delta_t
                eval_indices = np.where(eval_indices_float > 0,
                                        np.floor(eval_indices_float - 1e-13).astype(int),
                                        eval_indices_float.astype(int))

                input_data = input_data - self.delta_t*eval_indices

                eval_interval = np.vectorize(self._eval_interval)

                return eval_interval(index=eval_indices, input_data=input_data)

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

    ### A partir daqui rodamos os deltas de tempo
    for i in range(0, int(intervals), 1):

        net = net_

        time_train = np.linspace(0, delta_t, n)[:, None]
        time_eval = np.linspace(0, delta_t, n)[:, None]

        # Simple model of flame growth
        initial_state = np.array([state_t])

        residual = SymbolicOperator(
            expressions=[f],
            input_vars=["t"],
            output_vars=["u"],
            function=net,
            engine="torch",
            device="gpu",
        )

        params = {
            "residual": residual,
            "initial_input": np.array([0])[:, None],
            "initial_state": initial_state,
            "weights_residual": [1],
            "initial_penalty": 1,
        }

        # Reduce Epochs for sequential PINNs
        n_epochs_ini = 5_000  # Maximum number of iterations for ADAM
        n_epochs_iter = n_epochs_ini / (i+1)
        n_epochs_min = 500
        n_epochs = int(max(n_epochs_iter, n_epochs_min))

        # First Evaluation With ADAM Optimizer
        optimizer.fit(
            op=net,
            input_data=time_train,
            n_epochs=n_epochs,
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

        time_eval_plot = np.linspace((i) * delta_t, i * delta_t, N)[:, None]
        time_plot = np.vstack((time_plot, time_eval_plot))
        #approximated_data_plot = np.vstack((approximated_data_plot, approximated_data))

        # Plot Results
        #plt.plot(time_plot, approximated_data_plot, label="Approximated")
        #plt.xlabel("t")
        #plt.legend()
        #plt.ylim([0.0, 1.1])
        #plt.show()

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
        name="multi_fidelity_fire_pinn",
        model=multi_net,
        template=model_,
    )

    time_plot = np.linspace(0, 200, 1000)[:, None]

    approximated_data_plot = multi_net.eval(input_data=time_plot)

    # Compare PINN and EDO Results
    #ODE_Results = pd.read_csv("Fire.csv")
    plt.plot(time_plot, approximated_data_plot, label="PINN")
    #plt.plot(ODE_Results["Time"], ODE_Results["u"], label="ODE")
    plt.xlabel("t")
    plt.legend()
    plt.ylim([0.0, 1.1])
    plt.show()

# Restoring the model from disk and using it for making 
# evaluations
else:

    saver = SPFile(compact=False)
    multi_net = saver.read(model_path='./multi_fidelity_fire_pinn', device='cpu')

    multi_net.summary()

    time_plot = np.linspace(0, 200, 1000)[:, None]

    approximated_data_plot = multi_net.eval(input_data=time_plot)

    # Compare PINN and EDO Results
    #ODE_Results = pd.read_csv("Fire.csv")
    plt.plot(time_plot, approximated_data_plot, label="PINN")
    #plt.plot(ODE_Results["Time"], ODE_Results["u"], label="ODE")
    plt.xlabel("t")
    plt.legend()
    plt.ylim([0.0, 1.1])
    plt.show()
