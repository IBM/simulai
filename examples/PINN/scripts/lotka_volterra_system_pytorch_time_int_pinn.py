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
from scipy.integrate import odeint

os.environ["engine"] = "pytorch"

from simulai.io import IntersectingBatches
from simulai.metrics import L2Norm
from simulai.models import DeepONet
from simulai.optimization import Optimizer
from simulai.regression import DenseNetwork, ResDenseNetwork
from simulai.residuals import SymbolicOperator


# Projection to interval
def project_to_interval(interval, data):
    return interval[1] * (data - data.min()) / (data.max() - data.min()) + interval[0]


# Lotka-Volterra numerical solver
class LotkaVolterra:
    def __init__(self, alpha=None, beta=None, gamma=None, delta=None):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta

    def eval(self, state: np.ndarray = None, t: float = None) -> np.ndarray:
        x = state[0]
        y = state[1]

        x_residual = self.alpha * x - self.beta * x * y
        y_residual = self.delta * x * y - self.gamma * y

        return np.array([x_residual, y_residual])

    def run(self, initial_state, t):
        solution = odeint(self.eval, initial_state, t)

        return np.vstack(solution)


# Basic configurations
alpha = 1.1
beta = 0.4
gamma = 0.4
delta = 0.1
dt = 0.01
T_max = 300
n_samples = int(T_max / dt)
train_fraction = 0.8
n_samples_train = int(train_fraction * n_samples)
delta_t = 1
n_chunk_samples = 1

# The expression we aim at minimizing
f_x = "D(x, t) - ( alpha * x  - betha * x * y )"
f_y = "D(y, t) - ( delta * x * y - gammma * y )"

input_labels = ["t"]
output_labels = ["x", "y"]

# Generating data
t = np.arange(0, T_max, dt)
t_test = t[n_samples_train:]

initial_state = np.array([20, 5])

solver = LotkaVolterra(alpha=alpha, beta=beta, gamma=gamma, delta=delta)

data = solver.run(initial_state, t)

# Pre-processing data
batcher = IntersectingBatches(skip_size=1, batch_size=int(delta_t / dt))

time_chunks_ = batcher(input_data=t[:n_samples_train])
data_chunks = batcher(input_data=data[:n_samples_train])

T_max_train = n_samples_train * dt

time_aux = [
    t[(t >= i) & (t < i + delta_t)] for i in np.arange(T_max_train, T_max, delta_t)
]
data_aux = [
    data[(t >= i) & (t < i + delta_t)] for i in np.arange(T_max_train, T_max, delta_t)
]

initial_states = [chunk[0] for chunk in data_chunks]

time_chunks = [
    project_to_interval([0, delta_t], chunk)[:, None] for chunk in time_chunks_
]

time_chunks_train = list()
data_chunks_train = list()

for i in range(len(time_chunks)):
    indices = sorted(np.random.choice(time_chunks[i].shape[0], n_chunk_samples))
    time_chunks_train.append(time_chunks[i][indices])
    data_chunks_train.append(data_chunks[i][indices])

initial_states_train = initial_states

time_chunks_test = [project_to_interval([0, 1], chunk)[:, None] for chunk in time_aux]
data_chunks_test = data_aux
initial_states_test = [chunk[0] for chunk in data_aux]

branch_input_train = np.vstack(
    [
        np.tile(init[None, :], (time_chunk.shape[0], 1))
        for init, time_chunk in zip(initial_states_train, time_chunks_train)
    ]
)

branch_input_test = np.vstack(
    [
        np.tile(init, (time_chunk.shape[0], 1))
        for init, time_chunk in zip(initial_states_test, time_chunks_test)
    ]
)

trunk_input_train = np.vstack(time_chunks_train)
trunk_input_test = np.vstack(time_chunks_test)

output_train = np.vstack(data_chunks_train)
output_test = np.vstack(data_chunks_test)

# Configuring models
n_inputs = 1  # time, in the trunk network
n_outputs = 2  # x and y

lambda_1 = 0.0  # Penalty for the L¹ regularization (Lasso)
lambda_2 = 0.0  # Penalty factor for the L² regularization
n_epochs = 2_000  # 200_000  # Maximum number of iterations for ADAM
lr = 1e-3  # Initial learning rate for the ADAM algorithm
n_latent = 100

# Configuration for the fully-connected trunk network
trunk_config = {
    "layers_units": 9 * [100],  # Hidden layers
    "activations": "tanh",
    "input_size": 1,
    "output_size": n_latent * n_outputs,
    "residual_size": 2,
    "name": "trunk_net",
}

# Configuration for the fully-connected branch network
branch_config = {
    "layers_units": 9 * [100],  # Hidden layers
    "activations": "tanh",
    "input_size": n_outputs,
    "output_size": n_latent * n_outputs,
    "name": "branch_net",
}

# Instantiating and training the surrogate model
trunk_net = ResDenseNetwork(**trunk_config)
branch_net = DenseNetwork(**branch_config)

optimizer_config = {"lr": lr}

# Maximum derivative magnitudes to be used as loss weights
maximum_values = (1 / np.linalg.norm(output_train, 2, axis=0)).tolist()

# It prints a summary of the network features
trunk_net.summary()
branch_net.summary()

input_data = {"input_branch": branch_input_train, "input_trunk": trunk_input_train}

# Instantiating the DeepONet
lotka_volterra_net = DeepONet(
    trunk_network=trunk_net,
    branch_network=branch_net,
    var_dim=n_outputs,
    model_id="lotka_volterra_net",
    devices="gpu",
)

residual = SymbolicOperator(
    expressions=[f_x, f_y],
    input_vars=input_labels,
    output_vars=output_labels,
    function=lotka_volterra_net,
    inputs_key="input_trunk",
    constants={"alpha": alpha, "betha": beta, "gammma": gamma, "delta": delta},
    engine="torch",
)

# Instantiating the optimizer and training
optimizer = Optimizer("adam", params=optimizer_config)

params = {
    "lambda_1": lambda_1,
    "lambda_2": lambda_2,
    "residual": residual,
    "initial_input": input_data,
    "initial_state": output_train,
    "weights": maximum_values,
    "initial_penalty": 100,
}

# Evaluating approximation error in the test dataset
optimizer.fit(
    op=lotka_volterra_net,
    input_data=input_data,
    target_data=output_train,
    n_epochs=n_epochs,
    loss="opirmse",
    params=params,
    device="gpu",
)

approximated_data = lotka_volterra_net.eval(
    trunk_data=trunk_input_test, branch_data=branch_input_test
)

l2_norm = L2Norm()

error = 100 * l2_norm(
    data=approximated_data, reference_data=output_test, relative_norm=True
)

# Post-processing and visualizing
for ii in range(n_outputs):
    plt.plot(t_test, approximated_data[:, ii], label="Approximated")
    plt.plot(t_test, output_test[:, ii], label="Exact")
    plt.legend()
    plt.savefig(f"lorenz_deeponet_time_int_{ii}.png")
    plt.show()
    plt.close()

print(f"Approximation error for the variables: {error} %")
