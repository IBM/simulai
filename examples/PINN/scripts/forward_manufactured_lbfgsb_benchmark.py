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

import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from simulai.metrics import L2Norm
from simulai.optimization import Optimizer
from simulai.regression import DenseNetwork
from simulai.residuals import SymbolicOperator

# This is a very basic script for exploring the PDE solving via
# feedforward fully-connected neural networks

N = 10_000
n = 1_000
T_max = 1
omega = 40
mu = 0.25
pi = np.pi

time_train = (np.random.rand(n) * T_max)[:, None]
time_eval = np.linspace(0, T_max, N)[:, None]
time_ext = np.linspace(T_max, T_max + 0.5, N)[:, None]


def dataset(t: np.ndarray = None) -> np.ndarray:
    return (t - mu) ** 2 * np.cos(omega * np.pi * t)

def dataset_2(t: np.ndarray = None) -> np.ndarray:
    return np.sin(omega * np.pi * t)

# Datasets used for comparison
u_data = dataset_2(t=time_eval)
u_data_ext = dataset_2(t=time_ext)

def k1(t: torch.Tensor) -> torch.Tensor:
    return 2 * (t - mu) * torch.cos(omega * pi * t)

def k2(t: torch.Tensor) -> torch.Tensor:
    return torch.sin(omega * pi * t)

# The expression we aim at minimizing
#f = "D(u, t) - k1(t) + omega*pi*((t - mu)**2)*sin(omega*pi*t)"
f = "u - k2(t)"

input_labels = ["t"]
output_labels = ["u"]

n_inputs = len(input_labels)
n_outputs = len(output_labels)

n_epochs = 2_000  # Maximum number of iterations for ADAM
lr = 1e-3  # Initial learning rate for the ADAM algorithm


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


net = model()

optimizer_config = {"lr": lr}
optimizer = Optimizer("adam", params=optimizer_config)

residual = SymbolicOperator(
    expressions=[f],
    input_vars=["t"],
    output_vars=["u"],
    function=net,
    constants={"omega": omega, "mu": mu},
    external_functions={"k1": k1, "k2": k2},
    engine="torch",
    device="gpu",
)

params = {
    "residual": residual,
    "initial_input": np.array([0])[:, None],
    "initial_state": u_data[0],
    "weights_residual": [100],
    "initial_penalty": 1,
}

optimizer.fit(
    op=net,
    input_data=time_train,
    n_epochs=n_epochs,
    loss="pirmse",
    params=params,
    device="gpu",
)


from simulai.optimization import ScipyInterface
from simulai.optimization import PIRMSELoss

loss_instance = PIRMSELoss(operator=net)

optimizer_lbfgs = ScipyInterface(fun=net, optimizer='L-BFGS-B',
                                 loss=loss_instance, loss_config=params) 

optimizer_lbfgs.fit(input_data=time_train)

# Evaluation in training dataset
approximated_data = net.eval(input_data=time_eval)

l2_norm = L2Norm()

error = 100 * l2_norm(data=approximated_data, reference_data=u_data, relative_norm=True)

print(f"Approximation error: {error} %")

for ii in range(n_outputs):
    plt.plot(time_eval, approximated_data, label="Approximated")
    plt.plot(time_eval, u_data, label="Exact")
    plt.xlabel("t")
    plt.ylabel(f"{output_labels[ii]}")
    plt.legend()
    plt.show()
