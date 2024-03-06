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

import matplotlib.pyplot as plt
import numpy as np

from simulai.file import SPFile
from simulai.io import Tokenizer
from simulai.models import Transformer
from simulai.optimization import Optimizer
from simulai.regression import DenseNetwork, ModalRBFNetwork
from simulai.residuals import SymbolicOperator

# Our PDE
# Allen-cahn equation

f = "D(u, t) - mu*D(D(u, x), x) + alpha*(u**3) + beta*u"

g_u = "u"
g_ux = "D(u, x)"

input_labels = ["x", "t"]
output_labels = ["u"]

# Some fixed values
X_DIM = 50  # 256
T_DIM = 20  # 100

L = 1
x_0 = -1
T = 1

## Global parameters
n_epochs = 5_000
DEVICE = "gpu"
num_step = 10
# """
step = T / T_DIM
# """

# Generating the training grid

x_interval = [x_0, L]
t_interval = [0, T]

intervals = [x_interval, t_interval]

intv_array = np.vstack(intervals).T

# Regular grid
x_0, x_L = x_interval
t_0, t_L = t_interval
dx = (x_L - x_0) / X_DIM
dt = (t_L - t_0) / T_DIM

grid = np.mgrid[t_0 + dt : t_L + dt : dt, x_0:x_L:dx]

data = np.hstack([grid[1].flatten()[:, None], grid[0].flatten()[:, None]])

data_init = np.linspace(*x_interval, X_DIM)
u_init = ((data_init**2) * np.cos(np.pi * data_init))[:, None]
print(u_init.shape)

# Boundary grids
data_boundary_x0 = np.hstack(
    [
        x_interval[0] * np.ones((T_DIM, 1)),
        np.linspace(*t_interval, T_DIM)[:, None],
    ]
)

data_boundary_xL = np.hstack(
    [
        x_interval[-1] * np.ones((T_DIM, 1)),
        np.linspace(*t_interval, T_DIM)[:, None],
    ]
)

data_boundary_t0 = np.hstack(
    [
        np.linspace(*x_interval, X_DIM)[:, None],
        t_interval[0] * np.ones((X_DIM, 1)),
    ]
)

# Visualizing the training mesh
# plt.scatter(*np.split(data, 2, axis=1))
# plt.scatter(*np.split(data_boundary_x0, 2, axis=1))
# plt.scatter(*np.split(data_boundary_xL, 2, axis=1))
# plt.scatter(*np.split(data_boundary_t0, 2, axis=1))

# plt.show()
# plt.close()

n_epochs = 50_000  # Maximum number of iterations for ADAM
lr = 1e-3  # Initial learning rate for the ADAM algorithm

# Preparing datasets
tokenizer = Tokenizer(kind="spatiotemporal_indexer")
input_data = tokenizer.generate_input_tokens(
    input_data=data, num_step=num_step, step=step
)
data_boundary_x0 = tokenizer.generate_input_tokens(
    input_data=data_boundary_x0, num_step=num_step, step=step
)
data_boundary_xL = tokenizer.generate_input_tokens(
    input_data=data_boundary_xL, num_step=num_step, step=step
)
data_boundary_t0 = tokenizer.generate_input_tokens(
    input_data=data_boundary_t0, num_step=num_step, step=step
)
u_init = np.repeat(np.expand_dims(u_init, axis=1), num_step, axis=1)


def model():
    from simulai.regression import DenseNetwork

    input_labels = ["x", "t"]
    output_labels = ["u"]

    n_inputs = len(input_labels)
    n_outputs = len(output_labels)

    # Configuration for the fully-connected network
    config = {
        "layers_units": [128, 128, 128, 128],
        "activations": "tanh",
        "input_size": n_inputs,
        "output_size": n_outputs,
        "name": "allen_cahn_net",
    }

    # Instantiating and training the surrogate model
    net = DenseNetwork(**config)

    return net


def model_transformer():
    num_heads = 2
    embed_dim = 2
    embed_dim_out = 1
    hidden_dim = 8
    number_of_encoders = 2
    number_of_decoders = 2
    output_dim = embed_dim_out
    n_samples = 100

    input_data = np.random.rand(n_samples, embed_dim)

    # Configuration for the fully-connected branch network
    encoder_mlp_config = {
        "layers_units": [hidden_dim, hidden_dim, hidden_dim],  # Hidden layers
        "activations": "Wavelet",
        "input_size": embed_dim,
        "output_size": embed_dim,
        "name": "mlp_layer",
        "devices": "gpu",
    }

    decoder_mlp_config = {
        "layers_units": [hidden_dim, hidden_dim, hidden_dim],  # Hidden layers
        "activations": "Wavelet",
        "input_size": embed_dim,
        "output_size": embed_dim,
        "name": "mlp_layer",
        "devices": "gpu",
    }

    # Instantiating and training the surrogate model
    transformer = Transformer(
        num_heads_encoder=num_heads,
        num_heads_decoder=num_heads,
        embed_dim_encoder=embed_dim,
        embed_dim_decoder=embed_dim,
        output_dim=output_dim,
        encoder_activation="Wavelet",
        decoder_activation="Wavelet",
        encoder_mlp_layer_config=encoder_mlp_config,
        decoder_mlp_layer_config=decoder_mlp_config,
        number_of_encoders=number_of_encoders,
        number_of_decoders=number_of_decoders,
        devices="gpu",
    )

    transformer.summary()

    return transformer


optimizer_config = {"lr": lr}

net = model_transformer()

print(f"Number of coefficients: {net.n_parameters}")

residual = SymbolicOperator(
    expressions=[f],
    input_vars=input_labels,
    auxiliary_expressions={"periodic_u": g_u, "periodic_du": g_ux},
    constants={"mu": 1e-4, "alpha": 5, "beta": -5},
    output_vars=output_labels,
    function=net,
    engine="torch",
    device="gpu",
)

# It prints a summary of the network features
net.summary()

optimizer = Optimizer(
    "adam",
    params=optimizer_config,
    lr_decay_scheduler_params={
        "name": "ExponentialLR",
        "gamma": 0.9,
        "decay_frequency": 5_000,
    },
    shuffle=False,
    summary_writer=True,
)

params = {
    "residual": residual,
    "initial_input": data_boundary_t0,
    "initial_state": u_init,
    "boundary_input": {
        "periodic_u": [data_boundary_xL, data_boundary_x0],
        "periodic_du": [data_boundary_xL, data_boundary_x0],
    },
    "boundary_penalties": [1, 1],
    "weights_residual": [1],
    "initial_penalty": 100,
}

optimizer.fit(
    op=net,
    input_data=input_data,
    n_epochs=n_epochs,
    loss="pirmse",
    params=params,
    device="gpu",
)

saver = SPFile(compact=False)
saver.write(save_dir="/tmp", name="allen_cahn_net", model=net, template=model)

# Evaluation and post-processing
X_DIM_F = 5 * X_DIM
T_DIM_F = 5 * T_DIM

x_f = np.linspace(*x_interval, X_DIM_F)
t_f = np.linspace(*t_interval, T_DIM_F)

T_f, X_f = np.meshgrid(t_f, x_f, indexing="ij")

data_f = np.hstack([X_f.flatten()[:, None], T_f.flatten()[:, None]])

# Evaluation in training dataset
approximated_data = net.cpu().eval(input_data=data_f)

U_f = approximated_data.reshape(T_DIM_F, X_DIM_F)

fig, ax = plt.subplots()
ax.set_aspect("auto")
gf = ax.pcolormesh(X_f, T_f, U_f, cmap="jet")
fig.colorbar(gf)

plt.savefig("allen_cahn.png")
