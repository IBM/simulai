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
import os
from argparse import ArgumentParser
from typing import List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import random
import torch

torch.set_default_dtype(torch.float64)
torch.set_default_tensor_type(torch.DoubleTensor)

from simulai import ARRAY_DTYPE
from simulai.optimization import Optimizer, PIRMSELoss, ScipyInterface
from simulai.residuals import SymbolicOperator
from simulai.templates import NetworkTemplate, guarantee_device
from simulai.file import SPFile

"""#########################################################################"""
"""REPRODUCIBILITY                                                          """
"""#########################################################################"""

Fixed_Seed = 42

torch.manual_seed(Fixed_Seed)
np.random.seed(Fixed_Seed)
random.seed(Fixed_Seed)

"""#########################################################################"""
""" Will we train from scratch or restore a                                 """
""" pre-trained model from disk?                                            """
"""#########################################################################"""

parser = ArgumentParser(description="Reading input parameters")
parser.add_argument(
    "--train", type=str, help="Training new model or restoring from disk", default="yes"
)
args = parser.parse_args()

train = args.train


def create_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created successfully.")
    else:
        print(f"Directory '{directory_path}' already exists.")


Charts_Dir = "./Charts/"
create_directory(Charts_Dir)

train = "yes"

"""#########################################################################"""
""" Adaptive Epoch Decay                                                             """
"""#########################################################################"""
n_epochs_ini = 2_000  # Maximum number of iterations for ADAM
n_epochs_min = 500  # Minimum number of iterations for ADAM
Epoch_Tau = 2  # Number o Epochs Decay
lr = 1e-3  # Initial learning rate for the ADAM algorithm


def Epoch_Decay(iteration_number):
    if iteration_number < 100:
        n_epochs_iter = n_epochs_ini * (np.exp(-iteration_number / Epoch_Tau))
        n_epochs = int(max(n_epochs_iter, n_epochs_min))
    else:
        n_epochs = n_epochs_min
    print("N Epochs:", n_epochs)
    print("Iteration:", iteration_number + 1)
    return n_epochs


"""#########################################################################"""
""" Optimizer Config                                                        """
"""#########################################################################"""


def SetOptimizerConfig(solver, lr, Decay):
    global optimizer, get_n_epochs
    optimizer_config = {"lr": lr}

    solver = solver.lower()

    if Decay == False or Decay == None:
        print("Without Decay")
        optimizer = Optimizer(
            solver,
            params=optimizer_config,
        )
    else:
        print("With Exponential Decay")
        lr_decay_scheduler_params = {
            "name": "ExponentialLR",
            "gamma": 0.9,
            "decay_frequency": get_n_epochs / 10,
        }
        optimizer = Optimizer(
            solver,
            params=optimizer_config,
            lr_decay_scheduler_params=lr_decay_scheduler_params,
        )


"""#########################################################################"""
""" Adaptive Time Step                                                      """
"""#########################################################################"""

tol = 1e-06  # Truncation Error Tolerance
dt_init = 1e-04  # Initial Time Step Size
Last_It_Acepted = 0


def Delta_t(i, last_delta_t, Accepted):
    global get_n_epochs, Last_It_Acepted
    global dt_init
    dt_min = 1e-04  # Minimum Time Step Size
    dt_max = 5e-01  # Maximum Time Step Size
    Change_Iteration = 5  # Fixed Time Steps
    Refine_Factor = 1.2
    Expand_Factor = 1.1

    i = i + 1

    if Accepted == True:
        # Reduce Epochs for sequential PINNs
        if i < Change_Iteration + 1:
            Last_It_Acepted = 5
            get_n_epochs = Epoch_Decay(i - 1)
            dt_sug = dt_init
            print("Number of Fixed Time Steps:", Change_Iteration)
            print("Fixed Delta t:", dt_sug)
        else:
            if Last_It_Acepted >= 5:
                get_n_epochs = Epoch_Decay(i - 1)
                dt_sug = last_delta_t * Expand_Factor
            else:
                dt_sug = last_delta_t
                Last_It_Acepted = Last_It_Acepted + 1
                print("N Epochs:", get_n_epochs)
                print("Iteration:", i - 1)
            print("Suggested new Delta t:", dt_sug)
    else:
        Last_It_Acepted = 0
        get_n_epochs = get_n_epochs * 1.2
        get_n_epochs = int(min(get_n_epochs, n_epochs_ini))
        dt_sug = last_delta_t / Refine_Factor
        print("N Epochs:", get_n_epochs)
        print("Iteration:", i - 1)
        print("Suggested new Delta t:", dt_sug)

    Delta_t = min(max(dt_sug, dt_min), dt_max)
    print("Next Delta t (resp. Min and Max):", Delta_t)
    return Delta_t


"""#########################################################################"""
""" Adaptive Loss Weights                                                   """
"""#########################################################################"""


def adaptative_weights_residual(alpha: float = None):
    global weights_Old, initial_weights_Old, weights_residual
    global scale_factors, initial_residual

    if not alpha:
        alpha = 0.5

    weights_reduction = 1e-1
    gain = 1 / (n_outputs + 1)
    num = abs(state_t)
    Nunber_Of_Data = len(state_t)

    num = np.where(num < 0.1, 0.1, num)
    weights = 1 / num

    for j in range(Nunber_Of_Data):
        # +1 represent the initial values loss
        weights[j] = min(weights[j], 1 / (Nunber_Of_Data + 1))

    weights = weights * weights_reduction * gain
    weights_new = weights

    if i < 1:
        weights_residual = weights_new
    else:
        weights_residual = weights_Old * alpha + (1 - alpha) * weights_new

    weights_Old = weights_residual
    initial_residual = gain

    return


"""#########################################################################"""
""" Adaptive Scale Factors                                                   """
"""#########################################################################"""


def adaptative_scale_factors(alpha: float = None):
    global state_t, Scalers_Old, Scalers_New
    global scale_factors

    if not alpha:
        alpha = 0.5

    num = abs(state_t)
    Scalers_New = np.where(num < 0.001, 0.001, num) * 2

    if i < 1:
        Delta = np.array([0.001, 0.001, 0.001])
        Scalers = Scalers_New + Delta
    else:
        Scalers = Scalers_Old * alpha + (1 - alpha) * Scalers_New

    Scalers_Old = Scalers
    scale_factors = Scalers

    return


"""    Variables    """
N = 100
n = 100

t_max = 500


"""#########################################################################"""
"""    Kinetics Constants                                                   """
"""#########################################################################"""

k1 = 0.04
k2 = 3e7
k3 = 1e4

"""#########################################################################"""
"""    Initial_Conditions                                                   """
"""#########################################################################"""

s1_0 = 1.0
s2_0 = 0.0
s3_0 = 0.0

state_t = np.array([s1_0, s2_0, s3_0])

"""#########################################################################"""
"""    The expressions we aim at minimizing                                 """
"""#########################################################################"""
f_s1 = "D(s1, t) + k1*s1 - k3*s2*s3"

f_s2 = "D(s2, t) - k1*s1 + k2*(s2**2) + k3*s2*s3"

f_s3 = "D(s3, t) - k2*(s2**2)"

input_labels = ["t"]
output_labels = ["s1", "s2", "s3"]

n_inputs = len(input_labels)
n_outputs = len(output_labels)

if train == "yes":
    # Local model, which will be replicated for each sub-domain
    depth = 3
    width = 64
    activations_funct = "tanh"

    def model():
        from simulai.regression import SLFNN, ConvexDenseNetwork
        from simulai.models import ImprovedDenseNetwork

        # Configuration for the fully-connected network
        config = {
            "layers_units": depth * [width],  # Hidden layers
            "activations": activations_funct,
            "last_activation": "identity",
            "input_size": 1,
            "output_size": 3,
            "name": "net",
        }

        # Instantiating and training the surrogate model
        densenet = ConvexDenseNetwork(**config)
        encoder_u = SLFNN(input_size=1, output_size=width, activation=activations_funct)
        encoder_v = SLFNN(input_size=1, output_size=width, activation=activations_funct)

        class ScaledImprovedDenseNetwork(ImprovedDenseNetwork):
            def __init__(
                self,
                network=None,
                encoder_u=None,
                encoder_v=None,
                devices="gpu",
                scale_factors=None,
            ):
                super(ScaledImprovedDenseNetwork, self).__init__(
                    network=densenet,
                    encoder_u=encoder_u,
                    encoder_v=encoder_v,
                    devices="gpu",
                )

                if scale_factors != None:
                    self.scale_factors = torch.from_numpy(
                        scale_factors.astype(ARRAY_DTYPE)
                    ).to(self.device)
                else:
                    self.scale_factors = torch.nn.Parameter(
                        data=torch.zeros(self.network.output_size), requires_grad=True
                    ).to(self.device)

            def set_scale_factors(self, scale_factors: np.ndarray = None) -> None:
                getattr(self, f"scale_factors").data = torch.from_numpy(
                    scale_factors.astype(ARRAY_DTYPE)
                )

            def forward(self, input_data=None):
                return super().forward(input_data) * self.scale_factors

        net = ScaledImprovedDenseNetwork(
            network=densenet, encoder_u=encoder_u, encoder_v=encoder_v, devices="gpu"
        )

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
        width = 64
        activations_funct = "tanh"
        n_intervals = 5000  # overestimated

        # Model used for initialization
        def sub_model():
            from simulai.regression import SLFNN, ConvexDenseNetwork
            from simulai.models import ImprovedDenseNetwork

            # Configuration for the fully-connected network
            config = {
                "layers_units": depth * [width],  # Hidden layers
                "activations": activations_funct,
                "last_activation": "identity",
                "input_size": 1,
                "output_size": 3,
                "name": "net",
            }

            # Instantiating and training the surrogate model
            densenet = ConvexDenseNetwork(**config)
            encoder_u = SLFNN(
                input_size=1, output_size=width, activation=activations_funct
            )
            encoder_v = SLFNN(
                input_size=1, output_size=width, activation=activations_funct
            )

            class ScaledImprovedDenseNetwork(ImprovedDenseNetwork):
                def __init__(
                    self,
                    network=None,
                    encoder_u=None,
                    encoder_v=None,
                    devices="gpu",
                    scale_factors=None,
                ):
                    super(ScaledImprovedDenseNetwork, self).__init__(
                        network=densenet,
                        encoder_u=encoder_u,
                        encoder_v=encoder_v,
                        devices="gpu",
                    )

                    if scale_factors != None:
                        self.scale_factors = torch.from_numpy(
                            scale_factors.astype(ARRAY_DTYPE)
                        ).to(self.device)
                    else:
                        self.scale_factors = torch.nn.Parameter(
                            data=torch.zeros(self.network.output_size),
                            requires_grad=False,
                        ).to(self.device)

                def set_scale_factors(self, scale_factors: np.ndarray = None) -> None:
                    getattr(self, f"scale_factors").data = torch.from_numpy(
                        scale_factors.astype(ARRAY_DTYPE)
                    )

                def forward(self, input_data=None):
                    return super().forward(input_data) * self.scale_factors

            net = ScaledImprovedDenseNetwork(
                network=densenet,
                encoder_u=encoder_u,
                encoder_v=encoder_v,
                devices="gpu",
            )

            # It prints a summary of the network features
            # net.summary()

            return net

        # Initialization for the multifidelity network
        models_list = [sub_model() for i in range(n_intervals)]

        # Prototype for the multifidelity network
        class MultiNetwork(NetworkTemplate):
            def __init__(
                self, models_list: List[NetworkTemplate] = None, device: str = "cpu"
            ) -> None:
                super(MultiNetwork, self).__init__()

                for i, model in enumerate(models_list):
                    self.set_network(net=model, index=i)

                self.delta_t = torch.nn.Parameter(
                    data=torch.zeros(len(models_list)), requires_grad=False
                ).to(device)

                self.device = device

            def set_delta_t(self, delta_t: float, index: int = None) -> None:
                getattr(self, "delta_t").data[index] = delta_t

            def set_network(
                self, net: NetworkTemplate = None, index: int = None
            ) -> None:
                setattr(self, f"worker_{index}", net)

                self.add_module(f"worker_{index}", net)

            def _eval_interval(
                self, index: int = None, input_data: torch.Tensor = None
            ) -> torch.Tensor:
                input_data = input_data[:, None]
                return getattr(self, f"worker_{index}").eval(input_data=input_data)

            @property
            def time_intervals(self):
                cumulated_time = [0] + np.cumsum(self.delta_t).tolist()

                return np.array(
                    [
                        [cumulated_time[i], cumulated_time[i + 1]]
                        for i in range(len(cumulated_time) - 1)
                    ]
                )

            def _interval_delimiter(
                self, value: float, lower: float, upper: float
            ) -> bool:
                if lower <= value[0] < upper:
                    return True
                else:
                    return False

            def eval(self, input_data: np.ndarray = None) -> np.ndarray:
                n_samples = input_data.shape[0]

                time_intervals = self.time_intervals

                eval_indices = list()
                for j in range(n_samples):
                    is_in_interval = [
                        self._interval_delimiter(input_data[j], low, up)
                        for low, up in time_intervals.tolist()
                    ]
                    index = is_in_interval.index(True)
                    eval_indices.append(index)

                cumulated_time = np.array([0] + np.cumsum(self.delta_t).tolist())

                input_data = (
                    input_data - cumulated_time[np.array(eval_indices)][:, None]
                )

                return np.vstack(
                    [
                        self._eval_interval(index=i, input_data=idata)
                        for i, idata in zip(eval_indices, input_data)
                    ]
                )

            def summary(self):
                print(self)

        multi_net = MultiNetwork(models_list=models_list)

        return multi_net

    net_ = model()
    multi_net = model_()

    time_plot = np.empty((0, 1), dtype=float)
    approximated_data_plot = np.empty((0, 1), dtype=float)
    time_eval_plot = np.empty((0, 1), dtype=float)

    ### Run Multifidelity Model
    i = 0
    t_acu = 0
    Not_Acepted_Steps = 0
    last_delta_t = dt_init
    Accepted = True
    get_Delta_t = Delta_t(i, last_delta_t, Accepted)

    adaptative_scale_factors(alpha=0.9)
    adaptative_weights_residual(alpha=0.9)
    net_.set_scale_factors(scale_factors=scale_factors)

    while t_acu < t_max:
        last_delta_t = get_Delta_t
        if i > 0:
            get_Delta_t = Delta_t(i, last_delta_t, Accepted)

        net = net_

        time_train = np.linspace(0, get_Delta_t, n)[:, None]
        time_eval = np.linspace(0, get_Delta_t, n)[:, None]

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
            "weights_residual": weights_residual,
            "initial_penalty": initial_residual,
        }

        print("\nRun Optimizer - Lower Learning Rate")
        SetOptimizerConfig("Adam", 1e-3, True)
        optimizer.fit(
            op=net,
            input_data=time_train,
            n_epochs=get_n_epochs,
            loss="pirmse",
            params=params,
            device="cpu",
        )

        # Evaluation in training dataset
        approximated_data = net.eval(input_data=time_eval)

        LastLoss = np.array(optimizer.loss_states["pde"]) + np.array(
            optimizer.loss_states["init"]
        )
        LastLoss = LastLoss[-1]

        if (LastLoss > tol) and i > 4:
            Accepted = False
            Not_Acepted_Steps += 1
            print("\nSolution Not Accepted")
            print("Last Loss:", LastLoss)
            print("Minimum Tol.:", tol)
            print("Run again with reduced time step")
        else:
            Accepted = True
            print("\nSolution Accepted")
            print("Last Loss:", LastLoss)
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

            adaptative_scale_factors(alpha=0.9)
            adaptative_weights_residual(alpha=0.9)
            net_.set_scale_factors(scale_factors=scale_factors)

            print("Simulated Time:", t_acu)

    print("Number of Non Acepted Steps:", Not_Acepted_Steps)
    print("Number of Steps:", i - 1)

    saver = SPFile(compact=False)
    saver.write(
        save_dir="./",
        name="adaptative_multifidelity_rober_pinn",
        model=multi_net,
        template=model_,
    )

# Restoring the model from disk and using it for making
# evaluations
else:
    from sklearn.metrics import r2_score

    saver = SPFile(compact=False)
    multi_net = saver.read(
        model_path="./adaptative_multifidelity_rober_pinn", device="cpu"
    )

    input_labels = ["t"]
    output_labels = ["s1", "s2", "s3"]

    multi_net.summary()

    # Compare PINN and EDO Results
    ODE_Results = pd.read_csv("001-Rober_ODE_Dataset.csv")
    Filter_Scale = 50
    ODE_Results_OnlyFewData = ODE_Results[::Filter_Scale]

    time_plot = ODE_Results["Time"].to_numpy()[:, None]

    approximated_data_plot = multi_net.eval(input_data=time_plot)

    # Scale data (s2*1e4) only to help visualization
    approximated_data_plot = approximated_data_plot * np.array([1, 1e4, 1])

    df1 = pd.DataFrame()
    df1 = pd.DataFrame(time_plot, columns=["Time"])
    df2 = pd.DataFrame(approximated_data_plot, columns=output_labels)
    df = pd.concat([df1, df2], axis=1)

    # Plot Results
    plt.figure(1)
    Chart_File_Name = Charts_Dir + "Rober_ODE_Multifidelity_PINN_Comparison.png"

    plt.scatter(
        ODE_Results_OnlyFewData["Time"],
        ODE_Results_OnlyFewData["s1"],
        s=20,
        label="ODE - s1",
    )
    plt.scatter(
        ODE_Results_OnlyFewData["Time"],
        ODE_Results_OnlyFewData["s2"],
        s=20,
        label="ODE - s2 (*1e4)",
    )
    plt.scatter(
        ODE_Results_OnlyFewData["Time"],
        ODE_Results_OnlyFewData["s3"],
        s=20,
        label="ODE - s3",
    )
    plt.plot(time_plot, df["s1"], label="PINN - s1")
    plt.plot(time_plot, df["s2"], label="PINN - s2 (*1e4)")
    plt.plot(time_plot, df["s3"], label="PINN - s3")

    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Concentration")
    plt.ylim(ymin=0.0)
    plt.ylim(ymax=1.1)
    plt.savefig(Chart_File_Name)
    plt.show()

    # Calculate R² score
    r2 = r2_score(ODE_Results["s1"], df["s1"])
    print("\ns1 R²:", r2)

    # Calculate R² score
    r2 = r2_score(ODE_Results["s2"], df["s2"])
    print("\ns2 R²:", r2)

    # Calculate R² score
    r2 = r2_score(ODE_Results["s3"], df["s3"])
    print("\ns3 R²:", r2)

    df.to_csv("002-Rober_PINN_Dataset.csv", index=False)
