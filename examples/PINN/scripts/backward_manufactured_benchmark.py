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
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.nn.parameter import Parameter

# In order to execute this script, it is necessary to
# set the environment variable engine as "pytorch" before initializing
# simulai
os.environ["engine"] = "pytorch"

from simulai.metrics import L2Norm
from simulai.optimization import Optimizer
from simulai.regression import DenseNetwork
from simulai.residuals import SymbolicOperator, diff

# This is a very basic script for exploring the PDE solving via
# feedforward fully-connected neural networks

N = 10_000
n = 1_000
T_max = 0.5
omega = 40
mu = 0.25
pi = np.pi

time_train = (np.random.rand(n) * T_max)[:, None]
time_eval = np.linspace(0, T_max, N)[:, None]
time_ext = np.linspace(T_max, T_max + 0.5, N)[:, None]
indices = np.random.choice(N, n, replace=False)

def dataset(t: np.ndarray = None) -> np.ndarray:
    return (t - mu) ** 2 * np.cos(omega * np.pi * t)

# Datasets used for comparison
u_data = dataset(t=time_eval)
u_data_ext = dataset(t=time_ext)

time_extra_train = time_eval[indices]
u_extra_train = u_data[indices]

def k1(t:torch.Tensor, mu) -> torch.Tensor:

    return 2*(t-mu)*torch.cos(omega*pi*t)

class MyExpression(torch.nn.Module):

    def __init__(self):

        super(MyExpression, self).__init__()

        self.mu = Parameter(torch.tensor(0.5)) 

    def forward(self, u, t):

        # The expression we aim at minimizing
        du_dt = diff(u, t)

        f = du_dt - k1(t, self.mu) + omega*pi*((t - self.mu)**2)*torch.sin(omega*pi*t)

        return f

expression = MyExpression()

input_labels = ["t"]
output_labels = ["u"]

n_inputs = len(input_labels)
n_outputs = len(output_labels)

n_epochs = 10_000  # Maximum number of iterations for ADAM
lr = 1e-3  # Initial learning rate for the ADAM algorithm

def model():

    from simulai.regression import SLFNN, ConvexDenseNetwork
    from simulai.models import ImprovedDenseNetwork

    # Configuration for the fully-connected network
    config = {
        "layers_units": [50, 50, 50],
        "activations": "tanh",
        "input_size": 1,
        "output_size": 1,
        "name": "net",
    }

    #Instantiating and training the surrogate model
    densenet = ConvexDenseNetwork(**config)
    encoder_u = SLFNN(input_size=1, output_size=50, activation="tanh")
    encoder_v = SLFNN(input_size=1, output_size=50, activation="tanh")

    net = ImprovedDenseNetwork(
        network=densenet, encoder_u=encoder_u, encoder_v=encoder_v
    )

   # It prints a summary of the network features
    net.summary()

    return net

net = model()

optimizer_config = {"lr": lr}
optimizer = Optimizer("adam", params=optimizer_config)

residual = SymbolicOperator(
    expressions=[expression],
    input_vars=["t"],
    output_vars=["u"],
    function=net,
    constants={"omega": omega},
    trainable_parameters={'mu': expression.mu},
    external_functions={"k1": k1},
    engine="torch",
)

params = {
    "residual": residual,
    "initial_input": np.array([0])[:, None],
    "initial_state": u_data[0],
    "extra_input_data": time_extra_train[:, None],
    "extra_target_data": u_extra_train[:, None],
    "weights_residual": [1],
    "initial_penalty": 1,
}

optimizer.fit(
    op=net,
    input_data=time_train,
    n_epochs=n_epochs,
    loss="pirmse",
    params=params,
)

# Evaluation in training dataset
approximated_data = net.eval(input_data=time_eval)

l2_norm = L2Norm()

error = 100 * l2_norm(
    data=approximated_data, reference_data=u_data, relative_norm=True
)

print(f"he parameter 'mu' was estimated as {expression.mu}")

for ii in range(n_outputs):
    plt.plot(time_eval, approximated_data, label="Approximated")
    plt.plot(time_eval, u_data, label="Exact")
    plt.xlabel("t")
    plt.ylabel(f"{output_labels[ii]}")
    plt.legend()
    plt.show()

    print(f"Approximation error for the derivatives: {error} %")
