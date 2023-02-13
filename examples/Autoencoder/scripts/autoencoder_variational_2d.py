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

import h5py
import numpy as np

# In order to execute this script, it is necessary to
# set the environment variable engine as "pytorch" before initializing
# simulai
os.environ["engine"] = "pytorch"

from simulai.batching import BatchwiseSampler
from simulai.file import SPFile
from simulai.optimization import Optimizer

# Reading command line arguments.
parser = ArgumentParser(description="Reading input parameters")

parser.add_argument("--data_path", type=str, help="Path to the HDF5 dataset.")
args = parser.parse_args()
data_path = args.data_path

save_path = os.path.dirname(data_path)

fp = h5py.File(data_path, "r")
dataset = fp.get("tasks")

N = dataset["u"].shape[0]
lr = 1e-3
train_fraction = 0.9
n_inputs = 1
n_outputs = 1
n_samples_train = int(train_fraction * N)
n_epochs = 100

data = BatchwiseSampler(
    dataset=dataset, input_variables=["u"], target_variables=["u"], channels_first=True
)
minimum, maximum = data.minmax(batch_size=1_00, data_interval=[0, n_samples_train])
max_value = np.maximum(np.abs(minimum), np.abs(maximum))


def model():
    from simulai.models import AutoencoderVariational
    from simulai.regression import SLFNN, ConvolutionalNetwork, Linear

    transpose = False

    ### Layers Configurations ####
    ### BEGIN
    encoder_layers = [
        {
            "in_channels": n_inputs,
            "out_channels": 16,
            "kernel_size": 3,
            "stride": 1,
            "padding": 1,
            "after_conv": {"type": "maxpool2d", "kernel_size": 2, "stride": 2},
        },
        {
            "in_channels": 16,
            "out_channels": 32,
            "kernel_size": 3,
            "stride": 1,
            "padding": 1,
            "after_conv": {"type": "maxpool2d", "kernel_size": 2, "stride": 2},
        },
        {
            "in_channels": 32,
            "out_channels": 64,
            "kernel_size": 3,
            "stride": 1,
            "padding": 1,
            "after_conv": {"type": "maxpool2d", "kernel_size": 2, "stride": 2},
        },
        {
            "in_channels": 64,
            "out_channels": 128,
            "kernel_size": 3,
            "stride": 1,
            "padding": 1,
            "after_conv": {"type": "maxpool2d", "kernel_size": 2, "stride": 2},
        },
    ]

    bottleneck_encoder_layers = {
        "input_size": 3200,
        "output_size": 20,
        "activation": "identity",
        "name": "bottleneck_encoder",
    }

    bottleneck_decoder_layers = {
        "input_size": 20,
        "output_size": 3200,
        "activation": "identity",
        "name": "bottleneck_decoder",
    }

    if transpose == False:
        decoder_layers = [
            {
                "in_channels": 128,
                "out_channels": 64,
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
                "before_conv": {
                    "type": "upsample",
                    "scale_factor": 2,
                    "mode": "bicubic",
                },
            },
            {
                "in_channels": 64,
                "out_channels": 32,
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
                "before_conv": {
                    "type": "upsample",
                    "scale_factor": 2,
                    "mode": "bicubic",
                },
            },
            {
                "in_channels": 32,
                "out_channels": 16,
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
                "before_conv": {
                    "type": "upsample",
                    "scale_factor": 2,
                    "mode": "bicubic",
                },
            },
            {
                "in_channels": 16,
                "out_channels": n_outputs,
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
                "before_conv": {
                    "type": "upsample",
                    "scale_factor": 2,
                    "mode": "bicubic",
                },
            },
        ]

    else:
        decoder_layers = [
            {"in_channels": 4, "out_channels": 8, "kernel_size": 2, "stride": 2},
            {"in_channels": 8, "out_channels": 16, "kernel_size": 2, "stride": 2},
            {
                "in_channels": 16,
                "out_channels": n_outputs,
                "kernel_size": 2,
                "stride": 2,
            },
        ]
    ### END
    ### Layers Configurations ####

    # Instantiating network
    encoder = ConvolutionalNetwork(
        layers=encoder_layers, activations="tanh", case="2d", name="encoder"
    )
    bottleneck_encoder = SLFNN(**bottleneck_encoder_layers)
    bottleneck_decoder = SLFNN(**bottleneck_decoder_layers)
    decoder = ConvolutionalNetwork(
        layers=decoder_layers,
        activations="tanh",
        case="2d",
        transpose=transpose,
        name="decoder",
    )

    autoencoder = AutoencoderVariational(
        encoder=encoder,
        bottleneck_encoder=bottleneck_encoder,
        bottleneck_decoder=bottleneck_decoder,
        decoder=decoder,
        encoder_activation="tanh",
    )

    return autoencoder


autoencoder = model()

autoencoder.summary(input_shape=data.input_shape)

optimizer_config = {"lr": lr, "n_samples": n_samples_train}
params = {"lambda_1": 0.0, "lambda_2": 0.0, "use_mean": False, "relative": True}

optimizer = Optimizer("adam", params=optimizer_config)

optimizer.fit(
    op=autoencoder,
    input_data=data.input_data,
    target_data=data.target_data,
    n_epochs=n_epochs,
    loss="vaermse",
    params=params,
    batch_size=5_00,
)

saver = SPFile(compact=False)
saver.write(
    save_dir=save_path,
    name="autoencoder_rb_just_test",
    model=autoencoder,
    template=model,
)

print(optimizer.loss_states)
