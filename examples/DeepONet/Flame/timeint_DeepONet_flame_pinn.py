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
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np

from simulai.file import SPFile
from simulai.optimization import Optimizer
from simulai.residuals import SymbolicOperator

# Reading command line arguments.
parser = ArgumentParser(description="Reading input parameters")

parser.add_argument("--save_path", type=str, help="Save path", default="/tmp")
parser.add_argument(
    "--model_name", type=str, help="Model name", default="flame_deeponet"
)
args = parser.parse_args()

save_path = args.save_path
model_name = args.model_name

Q = 1_000
N = int(5e4)
Delta_t = 0.01

t_intv = [0, Delta_t]
u_intv = np.stack([[0], [1]], axis=0)

# The expression we aim at minimizing
f_u = "D(u, t) - (2/u0)*(u**2 + u**3)"

U_t = np.random.uniform(low=t_intv[0], high=t_intv[1], size=Q)
U_u = np.random.uniform(low=u_intv[0], high=u_intv[1], size=(N, 1))

branch_input_train = np.tile(U_u[:, None, :], (1, Q, 1)).reshape(N * Q, -1)
trunk_input_train = np.tile(U_t[:, None], (N, 1))

initial_states = U_u

input_labels = ["t", "u0"]
output_labels = ["u"]

n_inputs = len(input_labels)
n_outputs = len(output_labels)

lambda_1 = 0.0  # Penalty for the L¹ regularization (Lasso)
lambda_2 = 0.0  # Penalty factor for the L² regularization
n_epochs = 200_000  # Maximum number of iterations for ADAM
lr = 1e-3  # Initial learning rate for the ADAM algorithm


def model():
    import numpy as np

    from simulai.models import DeepONet, ImprovedDenseNetwork
    from simulai.regression import SLFNN, ConvexDenseNetwork

    n_latent = 100
    n_inputs_b = 1
    n_inputs_t = 1
    n_outputs = 1

    # Configuration for the fully-connected trunk network
    trunk_config = {
        "layers_units": 7 * [100],  # Hidden layers
        "activations": "tanh",
        "input_size": n_inputs_t,
        "output_size": n_latent * n_outputs,
        "name": "trunk_net",
    }

    # Configuration for the fully-connected branch network
    branch_config = {
        "layers_units": 7 * [100],  # Hidden layers
        "activations": "tanh",
        "input_size": n_inputs_b,
        "output_size": n_latent * n_outputs,
        "name": "branch_net",
    }

    encoder_u_trunk = SLFNN(input_size=n_inputs_t, output_size=100, activation="tanh")
    encoder_v_trunk = SLFNN(input_size=n_inputs_t, output_size=100, activation="tanh")
    encoder_u_branch = SLFNN(input_size=n_inputs_b, output_size=100, activation="tanh")
    encoder_v_branch = SLFNN(input_size=n_inputs_b, output_size=100, activation="tanh")

    # Instantiating and training the surrogate model
    trunk_net_dense = ConvexDenseNetwork(**trunk_config)
    branch_net_dense = ConvexDenseNetwork(**branch_config)

    trunk_net = ImprovedDenseNetwork(
        network=trunk_net_dense, encoder_u=encoder_u_trunk, encoder_v=encoder_v_trunk
    )

    branch_net = ImprovedDenseNetwork(
        network=branch_net_dense, encoder_u=encoder_u_branch, encoder_v=encoder_v_branch
    )

    # It prints a summary of the network features
    trunk_net.summary()
    branch_net.summary()

    flame_net = DeepONet(
        trunk_network=trunk_net,
        branch_network=branch_net,
        var_dim=n_outputs,
        rescale_factors=np.array([1]),
        devices="gpu",
        model_id="flame_net",
    )

    return flame_net


flame_net = model()

residual = SymbolicOperator(
    expressions=[f_u],
    input_vars=input_labels,
    output_vars=output_labels,
    function=flame_net,
    inputs_key="input_trunk|input_branch:0",
    device="gpu",
    engine="torch",
)

penalties = [1]
batch_size = 10_000

optimizer_config = {"lr": lr}

input_data = {"input_branch": branch_input_train, "input_trunk": trunk_input_train}

optimizer = Optimizer(
    "adam",
    params=optimizer_config,
    lr_decay_scheduler_params={
        "name": "ExponentialLR",
        "gamma": 0.9,
        "decay_frequency": 5_000,
    },
    checkpoint_params={
        "save_dir": save_path,
        "name": model_name,
        "template": model,
        "checkpoint_frequency": 10_000,
        "overwrite": False,
    },
    summary_writer=True,
)

params = {
    "lambda_1": lambda_1,
    "lambda_2": lambda_2,
    "residual": residual,
    "initial_input": {"input_trunk": np.zeros((N, 1)), "input_branch": initial_states},
    "initial_state": initial_states,
    "weights_residual": [1, 1, 1],
    "weights": penalties,
}

optimizer.fit(
    op=flame_net,
    input_data=input_data,
    n_epochs=n_epochs,
    loss="opirmse",
    params=params,
    device="gpu",
    batch_size=batch_size,
    use_jit=True,
)

# Saving model
print("Saving model.")
saver = SPFile(compact=False)
saver.write(save_dir=save_path, name=model_name, model=flame_net, template=model)

initial_state_test = np.array([1e-3])
n_outputs = 1
n_times = 1 #int(2 / (initial_state_test[0] * Delta_t))
Q = 1000

branch_input_test = np.tile(initial_state_test[None, :], (Q, 1))
trunk_input_test = np.linspace(0, Delta_t, Q)[:, None]

eval_list = list()

for i in range(0, n_times):
    branch_input_test = np.tile(initial_state_test[None, :], (Q, 1))

    approximated_data = flame_net.eval(
        trunk_data=trunk_input_test, branch_data=branch_input_test
    )
    initial_state_test = approximated_data[-1]

    eval_list.append(approximated_data[0])

evaluation = np.vstack(eval_list)
time = np.linspace(0, n_times * Delta_t, evaluation.shape[0])

np.save("evaluation.npy", evaluation)
plt.plot(time, evaluation, label="Approximated")
plt.xlabel("t (s)")
plt.savefig("flame_approximation.png")
plt.close()

plt.figure(figsize=(15, 6))

for i in range(n_outputs):
    plt.plot(time, evaluation[:, i], label=f"u")
    plt.xlabel("t (s)")

plt.yticks(np.linspace(0, 1, 5))
plt.legend()
plt.grid(True)
plt.savefig(f"flame_approximation_custom.png")
