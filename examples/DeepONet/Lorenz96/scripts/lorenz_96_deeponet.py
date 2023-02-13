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


import numpy as np
from scipy.integrate import odeint

from simulai.file import SPFile
from simulai.io import IntersectingBatches
from simulai.metrics import L2Norm, LyapunovUnits
from simulai.optimization import Optimizer

save_path = "/tmp"
tol = 0.5
# These are our constants
N = 40  # Number of variables
F = 8  # Forcing
raw_init = False
differentiation = "spline"

label = f"n_{N}_F_{F}"

if F == 8:
    lambda_1 = 1 / 1.68
else:
    lambda_1 = 1 / 2.27


def Lorenz96(x, t):
    """Lorenz 96 model with constant forcing"""
    # Setting up vector
    d = np.zeros(N)
    # Loops over indices (with operations and Python underflow indexing handling edge cases)
    for i in range(N):
        d[i] = (x[(i + 1) % N] - x[i - 2]) * x[i - 1] - x[i] + F
    return d


x0 = F * np.ones(N)  # Initial state (equilibrium)
x0 += 0.01 * np.random.rand(N)  # Add small perturbation to the first variable

# Global parameters
dt = 0.5
discard = 1000
T_max = 2000
skip_size = 1
Q = 100
delta_t = 1
n_epochs = 10_000
lr = 1e-3
save_dir = "/tmp"
model_name = "L96"

# Generating data
t = np.arange(0.0, T_max, dt)
lorenz_data = odeint(Lorenz96, x0, t)

# Separating train and test datasets
n_steps = t[t >= discard].shape[0]
nt = int(0.5 * n_steps)
nt_test = n_steps - nt

t_train = t[t >= discard][:nt]
t_test = t[t >= discard][nt:]

n_field = N

train_field_ = lorenz_data[t >= discard][:nt]
test_field_ = lorenz_data[t >= discard][nt:]

max_value = train_field_.max(0)
min_value = train_field_.min(0)

train_field = 2 * (train_field_ - min_value) / (max_value - min_value) - 1
test_field = 2 * (test_field_ - min_value) / (max_value - min_value) - 1

time = np.linspace(0, delta_t, Q)

# Training dataset
batcher = IntersectingBatches(skip_size=skip_size, batch_size=Q)
train_data_splits = batcher(input_data=train_field)
initial_states = [chunk[0:1] for chunk in train_data_splits]
n_chunks = len(train_data_splits)

trunk_train_input = np.vstack(n_chunks * [time[:, None]])
branch_train_input = np.vstack(
    [np.tile(init_state, (Q, 1)) for init_state in initial_states]
)

# Testing dataset
data_splits = np.array_split(test_field, int(test_field.shape[0] / Q), axis=0)
test_initial_states = [chunk[0:1] for chunk in data_splits]
n_chunks_test = len(data_splits)

trunk_test_input = np.vstack(n_chunks * [time[:, None]])
branch_test_input = np.vstack(
    [np.tile(init_state, (Q, 1)) for init_state in initial_states]
)

train_output = np.vstack(train_data_splits)

input_data = {"input_branch": branch_train_input, "input_trunk": trunk_train_input}


# Model template
def model():
    from simulai.models import ImprovedDeepONet as DeepONet
    from simulai.regression import SLFNN, ConvexDenseNetwork

    n_latent = 200
    n_inputs_t = 1
    n_inputs_b = 40
    n_outputs = 40

    activation_t = "sin"
    activation_b = "sin"
    activation_et = "sin"
    activation_eb = "sin"

    # Configuration for the fully-connected trunk network
    trunk_config = {
        "layers_units": 7 * [100],
        "activations": activation_t,
        "input_size": n_inputs_t,
        "output_size": n_latent * n_outputs,
        "last_activation": "identity",  # activation,
        "name": "trunk_net",
    }

    # Configuration for the fully-connected branch network
    branch_config = {
        "layers_units": 7 * [100],
        "activations": activation_b,
        "input_size": n_inputs_b,
        "output_size": n_latent * n_outputs,
        "last_activation": "identity",  # activation,
        "name": "branch_net",
    }

    # Instantiating and training the surrogate model
    trunk_net = ConvexDenseNetwork(**trunk_config)

    branch_net = ConvexDenseNetwork(**branch_config)

    encoder_trunk = SLFNN(
        input_size=n_inputs_t, output_size=100, activation=activation_et
    )
    encoder_branch = SLFNN(
        input_size=n_inputs_b, output_size=100, activation=activation_eb
    )

    l96_net = DeepONet(
        trunk_network=trunk_net,
        branch_network=branch_net,
        encoder_trunk=encoder_trunk,
        encoder_branch=encoder_branch,
        var_dim=n_outputs,
        multiply_by_trunk=False,
        devices="gpu",
        model_id="l96_net",
    )

    # It prints a summary of the network features
    l96_net.summary()

    print(f"This network has: {l96_net.n_parameters} parameters.")

    return l96_net


# Instantiating model
l96_net = model()

# Executing optimization
batch_size = 1_000
weights_ = (
    np.ones(train_output.shape[-1]) / 40
).tolist()  # np.max(np.abs(train_output), axis=0).tolist()
optimizer_config = {"lr": lr}
params = {
    "lambda_1": 0.0,
    "lambda_2": 1e-10,
    "weights": weights_,
    "use_mean": True,
    "relative": True,
}

optimizer = Optimizer(
    "adam",
    params=optimizer_config,
    lr_decay_scheduler_params={
        "name": "ExponentialLR",
        "gamma": 0.9,
        "decay_frequency": 2_500,
    },
)

optimizer.fit(
    op=l96_net,
    input_data=input_data,
    target_data=train_output,
    n_epochs=n_epochs,
    loss="wrmse",
    params=params,
    device="gpu",
    batch_size=batch_size,
)

# Saving model to disk
saver = SPFile(compact=False)
saver.write(save_dir=save_dir, name=model_name, model=l96_net, template=model)

state = test_initial_states[0]

# Composed time-extrapolation

evaluations = list()

for j in range(n_chunks_test):
    evaluation = l96_net.eval(
        trunk_data=time[:, None], branch_data=np.tile(state, (Q, 1))
    )
    state = evaluation[-1]
    evaluations.append(evaluation)

approximated = np.vstack(evaluations)

l2_norm = L2Norm()

error = 100 * l2_norm(data=approximated, reference_data=test_field, relative_norm=True)

print(f"Approximation error: {error} %")
