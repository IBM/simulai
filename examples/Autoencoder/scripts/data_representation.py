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
import torch

from simulai.file import SPFile
from simulai.metrics import L2Norm
from simulai.optimization import Optimizer

# Reading command line arguments.
parser = ArgumentParser(description="Reading input parameters")

parser.add_argument("--data_path", type=str, help="Path to the dataset.")

args = parser.parse_args()

data_path = args.data_path


# Model template
def model():
    from simulai.models import AutoencoderMLP
    from simulai.regression import DenseNetwork

    n_inputs = 512

    n_latent = 20

    # Configuration for the fully-connected branch network
    encoder_config = {
        "layers_units": [256, 128, 64],  # Hidden layers
        "activations": "sin",
        "input_size": n_inputs,
        "output_size": n_latent,
        "name": "encoder_net",
    }

    # Instantiating and training the surrogate model
    encoder_net = DenseNetwork(**encoder_config)

    # Configuration for the fully-connected branch network
    decoder_config = {
        "layers_units": [64, 128, 256],  # Hidden layers
        "activations": "sin",
        "input_size": n_latent,
        "output_size": n_inputs,
        "name": "decoder_net",
    }

    # Instantiating and training the surrogate model
    decoder_net = DenseNetwork(**decoder_config)

    autoencoder = AutoencoderMLP(encoder=encoder_net, decoder=decoder_net)

    return autoencoder


autoencoder = model()
data = np.load(data_path)[::10]

data_tensor = torch.from_numpy(data.astype("float32"))

autoencoder.summary()

lr = 1e-4
n_epochs = 100

optimizer_config = {"lr": lr}
params = {"lambda_1": 0.0, "lambda_2": 1e-10}

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

autoencoder_reload = saver.read(model_path="/tmp/autoencoder_mlp")

approximated_data = autoencoder_reload.eval(input_data=data_test)

l2_norm = L2Norm()

projection_error = 100 * l2_norm(
    data=approximated_data, reference_data=data_test, relative_norm=True
)

print(f"Projection error: {projection_error} %")
