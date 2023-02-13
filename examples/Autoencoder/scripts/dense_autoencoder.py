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
import torch

from simulai.file import SPFile
from simulai.metrics import L2Norm
from simulai.optimization import Optimizer


def u(t, x, L: float = None, t_max: float = None) -> np.ndarray:
    return np.sin(4 * np.pi * t * (x / L - 1 / 2) ** 2) * np.cos(
        5 * np.pi * (t / t_max - 1 / 2) ** 2
    )


t_max = 10
L = 5
K = 512
N = 10_000

x = np.linspace(0, L, K)
t = np.linspace(0, t_max, N)
T, X = np.meshgrid(t, x, indexing="ij")


# Model template
def model():
    from simulai.models import AutoencoderMLP
    from simulai.regression import DenseNetwork

    K = 512
    N = 10_000

    n_samples, n_inputs = N, K
    n_latent = 50

    # Configuration for the fully-connected branch network
    encoder_config = {
        "layers_units": [256, 128, 64],  # Hidden layers
        "activations": "tanh",
        "input_size": n_inputs,
        "output_size": n_latent,
        "name": "encoder_net",
    }

    # Instantiating and training the surrogate model
    encoder_net = DenseNetwork(**encoder_config)

    # Configuration for the fully-connected branch network
    decoder_config = {
        "layers_units": [64, 128, 256],  # Hidden layers
        "activations": "tanh",
        "input_size": n_latent,
        "output_size": n_inputs,
        "name": "decoder_net",
    }

    # Instantiating and training the surrogate model
    decoder_net = DenseNetwork(**decoder_config)

    autoencoder = AutoencoderMLP(encoder=encoder_net, decoder=decoder_net)

    return autoencoder


autoencoder = model()
data = u(T, X, L=L, t_max=t_max)

data_tensor = torch.from_numpy(data.astype("float32"))

autoencoder.summary()

lr = 1e-3
n_epochs = 100

optimizer_config = {"lr": lr}
params = {"lambda_1": 0.0, "lambda_2": 1e-4}

optimizer = Optimizer("adam", params=optimizer_config)

data_train = data
data_test = data

optimizer.fit(
    op=autoencoder,
    input_data=data_train,
    target_data=data_train,
    n_epochs=n_epochs,
    loss="rmse",
    params=params,
    batch_size=5_00,
)

# First evaluation
approximated_data = autoencoder.eval(input_data=data_test)

l2_norm = L2Norm()

projection_error = 100 * l2_norm(
    data=approximated_data, reference_data=data_test, relative_norm=True
)

print(f"Projection error: {projection_error} %")

autoencoder.save(save_dir="/tmp", name="autoencoder_mlp")

saver = SPFile(compact=False)
saver.write(save_dir="/tmp", name="autoencoder_mlp", model=autoencoder, template=model)

print("Restoring from disk.")

autoencoder_reload = saver.read(model_path="/tmp/autoencoder_mlp")

approximated_data = autoencoder_reload.eval(input_data=data_test)

l2_norm = L2Norm()

projection_error = 100 * l2_norm(
    data=approximated_data, reference_data=data_test, relative_norm=True
)

print(f"Projection error: {projection_error} %")
