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

from argparse import ArgumentParser

import numpy as np

from simulai.file import SPFile
from simulai.metrics import L2Norm
from simulai.optimization import Optimizer

# Reading command line arguments.
parser = ArgumentParser(description="Reading input parameters")

parser.add_argument("--data_path", type=str, help="Path to the dataset.")

args = parser.parse_args()

data_path = args.data_path

manufactured = True


# Model template
def model():
    from simulai.activations import Siren
    from simulai.regression import ResDenseNetwork as DenseNetwork

    # Configuration for the fully-connected branch network
    encoder_config = {
        "layers_units": [50, 50, 50],  # Hidden layers
        "activations": Siren(omega_0=30, c=6),
        "input_size": 2,
        "output_size": 1,
        "name": "net",
    }

    # Instantiating and training the surrogate model
    net = DenseNetwork(**encoder_config)

    return net


net = model()

net.summary()

lr = 5e-5
n_epochs = 2_000
n_train = 20_000

# Data preparing
if manufactured is True:

    def u(t, x, L: float = None, t_max: float = None) -> np.ndarray:
        return np.sin(
            4 * np.pi * t * np.cos(5 * np.pi * (t / t_max)) * (x / L - 1 / 2) ** 2
        ) * np.cos(5 * np.pi * (t / t_max - 1 / 2) ** 2)

    t_max = 10
    L = 5
    K = 512
    N = 100_000

    x_interval = [0, L]
    time_interval = [0, t_max]

    x = np.linspace(*x_interval, K)
    t = np.linspace(*time_interval, N)

    T, X = np.meshgrid(t, x, indexing="ij")
    output_data = u(T, X, L=L, t_max=t_max)

    positions = np.stack([X[::100].flatten(), T[::100].flatten()], axis=1)
    positions = 2 * positions / np.array([L, t_max]) - 1
else:
    output_data = np.load(data_path)[::10]

optimizer_config = {"lr": lr}


n_t, n_x = output_data.shape

x_i = np.random.randint(0, n_x, size=(n_train, 1))
t_i = np.random.randint(0, n_t, size=(n_train, 1))

input_train = 2 * np.hstack([x[x_i], t[t_i]]) / np.array([L, t_max]) - 1
output_train = output_data[t_i, x_i]

# Configuring Optimizer
params = {"lambda_1": 0.0, "lambda_2": 1e-14}

optimizer = Optimizer("adam", params=optimizer_config)

optimizer.fit(
    op=net,
    input_data=input_train,
    target_data=output_train,
    n_epochs=n_epochs,
    loss="rmse",
    params=params,
    batch_size=1_000,
)

# First evaluation
approximated_data = net.eval(input_data=positions)

l2_norm = L2Norm()

projection_error = 100 * l2_norm(
    data=approximated_data, reference_data=output_data[::100], relative_norm=True
)

print(f"Projection error: {projection_error} %")

# Saving model
print("Saving model.")
saver = SPFile(compact=False)
saver.write(save_dir="/tmp", name="data_representation", model=net, template=model)

# Testing to reload from disk
saver = SPFile(compact=False)
net_reload = saver.read(model_path="/tmp/data_representation")

# Post-processing
approximated_data = net_reload.eval(input_data=positions)
approximated_data = approximated_data.reshape(-1, K)

l2_norm = L2Norm()

projection_error = 100 * l2_norm(
    data=approximated_data, reference_data=output_data[::100], relative_norm=True
)

print(f"Projection error: {projection_error} %")

import matplotlib.pyplot as plt

plt.imshow(approximated_data)
plt.colorbar()

plt.show()

plt.imshow(output_data[::100])
plt.colorbar()

plt.show()
