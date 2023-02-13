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
import random

import matplotlib.pyplot as plt
import numpy as np

os.environ["engine"] = "pytorch"

from examples.utils.oscillator_solver import oscillator_solver
from simulai.metrics import L2Norm
from simulai.models import DeepONet
from simulai.optimization import Optimizer
from simulai.regression import DenseNetwork

n_steps = 1000
T = 50
dt = T / n_steps
initial_state = np.array([1, 0])[None, :]
time = np.arange(1, n_steps + 1) * dt

n_cases = 500
train_fraction = 0.8
n_cases_plot = 10
n_cases_train = int(train_fraction * n_cases)
n_cases_test = n_cases - n_cases_train

parameters_keys = ["alpha1", "alpha2", "beta1", "beta2"]

alpha1_interval = [-0.5, -0.1]
alpha2_interval = [-1, -0.5]
beta1_interval = [1, 2]
beta2_interval = [-0.5, -0.1]

list_of_cases = list()
input_parameters = list()

for j in range(n_cases):
    extra_params = {
        "alpha1": random.uniform(*alpha1_interval),
        "alpha2": random.uniform(*alpha2_interval),
        "beta1": random.uniform(*beta1_interval),
        "beta2": random.uniform(*beta2_interval),
    }

    list_of_cases.append(extra_params)
    input_parameters.append([extra_params[key] for key in parameters_keys])

input_parameters = np.vstack(input_parameters)

cases_train_indices = np.random.choice(n_cases, n_cases_train, replace=False).tolist()
cases_test_indices = [i for i in range(n_cases) if not i in cases_train_indices]
cases_plot_indices = np.random.choice(
    cases_test_indices, n_cases_plot, replace=False
).tolist()

input_parameters_train = input_parameters[cases_train_indices]
input_parameters_test = input_parameters[cases_test_indices]

field_data = list()

for j in range(n_cases):
    oscillator_data, _ = oscillator_solver(
        T, dt, initial_state, extra_params=list_of_cases[j]
    )
    field_data.append(oscillator_data[None, ...])

field_data_train = np.vstack(field_data)[cases_train_indices]
field_data_test = np.vstack(field_data)[cases_test_indices]

# Constructing datasets
Time_train = np.tile(time[:, None], (n_cases_train, 1))

Field_data_train = field_data_train.reshape(n_cases_train * n_steps, -1)
Input_parameters_train = np.tile(
    input_parameters_train[:, None, :], (1, n_steps, 1)
).reshape(n_steps * n_cases_train, -1)

n_inputs = 4
n_outputs = 2

lambda_1 = 1e-3  # Penalty for the L¹ regularization (Lasso)
lambda_2 = 1e-3  # Penalty factor for the L² regularization
n_epochs = int(1e3)  # Maximum number of iterations for ADAM
lr = 1e-3  # Initial learning rate for the ADAM algorithm
n_latent = 50

# Configuration for the fully-connected trunk network
trunk_config = {
    "layers_units": [50, 50, 50],  # Hidden layers
    "activations": "elu",
    "input_size": 1,
    "output_size": n_latent * n_outputs,
    "name": "trunk_net",
}

# Configuration for the fully-connected branch network
branch_config = {
    "layers_units": [50, 50, 50],  # Hidden layers
    "activations": "elu",
    "input_size": n_inputs,
    "output_size": n_latent * n_outputs,
    "name": "branch_net",
}

# Instantiating and training the surrogate model
trunk_net = DenseNetwork(**trunk_config)
branch_net = DenseNetwork(**branch_config)

optimizer_config = {"lr": lr}

# Maximum derivative magnitudes to be used as loss weights
# (1/np.linalg.norm(Field_data_train, 1, axis=0)).tolist()
maximum_values = [1, 1]

params = {"lambda_1": lambda_1, "lambda_2": lambda_2, "weights": maximum_values}

# It prints a summary of the network features
trunk_net.summary()
branch_net.summary()

input_data = {"input_branch": Input_parameters_train, "input_trunk": Time_train}

oscillator_net = DeepONet(
    trunk_network=trunk_net,
    branch_network=branch_net,
    var_dim=n_outputs,
    model_id="oscillator_net",
)

optimizer = Optimizer("adam", params=optimizer_config)

optimizer.fit(
    oscillator_net,
    input_data=input_data,
    target_data=Field_data_train,
    n_epochs=n_epochs,
    loss="wrmse",
    params=params,
)

for c in range(n_cases_test):
    Time_test = time[:, None]
    Field_data_test = field_data_test[c]
    Input_parameters_test = np.tile(input_parameters_test[c], (n_steps, 1))

    approximated_data = oscillator_net.eval(
        trunk_data=Time_test, branch_data=Input_parameters_test
    )

    l2_norm = L2Norm()

    error = 100 * l2_norm(
        data=approximated_data, reference_data=Field_data_test, relative_norm=True
    )

    if c in cases_plot_indices:
        for ii in range(n_outputs):
            plt.plot(time, approximated_data[:, ii], label="Approximated")
            plt.plot(time, Field_data_test[:, ii], label="Exact")
            plt.grid(True)
            plt.legend()
            plt.show()

    print(f"Approximation error for the derivatives: {error} %")
