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
from unittest import TestCase

import numpy as np
from tests.config import configure_dtype
torch = configure_dtype()

from utils import configure_device

from simulai import ARRAY_DTYPE
from simulai.file import SPFile
from simulai.optimization import Optimizer

DEVICE = configure_device()


def generate_data(
    n_samples: int = None,
    image_size: tuple = None,
    n_inputs: int = None,
    n_outputs: int = None,
) -> (torch.Tensor, torch.Tensor):

    input_data = np.random.rand(n_samples, n_inputs, *image_size)
    output_data = np.random.rand(n_samples, n_outputs, *image_size)

    return torch.from_numpy(input_data.astype(ARRAY_DTYPE)), torch.from_numpy(
        output_data.astype(ARRAY_DTYPE)
    )

# Model template
def model_2d():
    from simulai.models import UNet

    # Configuring model
    n_inputs = 3
    n_outputs = 1
    n_ch_0 = 2

    layers = {
        "encoder": {
            "type": "cnn", 
            "architecture" :[

                {
                    "in_channels": n_inputs,
                    "out_channels": n_ch_0,
                    "kernel_size": 3,
                    "stride": 1,
                    "padding": 1,
                },
               
                {
                    "in_channels": n_ch_0,
                    "out_channels": n_ch_0,
                    "kernel_size": 3,
                    "stride": 1,
                    "padding": 1,
                    "after_conv": {"type": "maxpool2d", "kernel_size": 2, "stride": 2},
                },

                {
                    "in_channels": n_ch_0,
                    "out_channels": 2*n_ch_0,
                    "kernel_size": 3,
                    "stride": 1,
                    "padding": 1,
                },
              
                {
                    "in_channels": 2*n_ch_0,
                    "out_channels": 2*n_ch_0,
                    "kernel_size": 3,
                    "stride": 1,
                    "padding": 1,
                    "after_conv": {"type": "maxpool2d", "kernel_size": 2, "stride": 2},
                },

                {
                    "in_channels": 2*n_ch_0,
                    "out_channels": 4*n_ch_0,
                    "kernel_size": 3,
                    "stride": 1,
                    "padding": 1,
                },
              
                {
                    "in_channels": 4*n_ch_0,
                    "out_channels": 4*n_ch_0,
                    "kernel_size": 3,
                    "stride": 1,
                    "padding": 1,
                    "after_conv": {"type": "maxpool2d", "kernel_size": 2, "stride": 2},
                },

                {
                    "in_channels": 4*n_ch_0,
                    "out_channels": 8*n_ch_0,
                    "kernel_size": 3,
                    "stride": 1,
                    "padding": 1,
                },
              
                {
                    "in_channels": 8*n_ch_0,
                    "out_channels": 8*n_ch_0,
                    "kernel_size": 3,
                    "stride": 1,
                    "padding": 1,
                    "after_conv": {"type": "maxpool2d", "kernel_size": 2, "stride": 2},
                },

                {
                    "in_channels": 8*n_ch_0,
                    "out_channels": 16*n_ch_0,
                    "kernel_size": 3,
                    "stride": 1,
                    "padding": 1,
                },

            ]
        },
        "decoder": {
            "type": "cnn", 
            "architecture" :[

                {
                    "in_channels": 16*n_ch_0,
                    "out_channels": 16*n_ch_0,
                    "kernel_size": 3,
                    "stride": 1,
                    "padding": 1,

                },

                {
                    "in_channels": 16*n_ch_0,
                    "out_channels": 8*n_ch_0,
                    "kernel_size": 3,
                    "stride": 1,
                    "padding": 1,
                    "before_conv": {"type": "upsample", "scale_factor": 2, "mode": "bicubic"},
                },

                {
                    "in_channels": 16*n_ch_0,
                    "out_channels": 8*n_ch_0,
                    "kernel_size": 3,
                    "stride": 1,
                    "padding": 1,

                },

                {
                    "in_channels": 8*n_ch_0,
                    "out_channels": 8*n_ch_0,
                    "kernel_size": 3,
                    "stride": 1,
                    "padding": 1,

                },

                {
                    "in_channels": 8*n_ch_0,
                    "out_channels": 4*n_ch_0,
                    "kernel_size": 3,
                    "stride": 1,
                    "padding": 1,
                    "before_conv": {"type": "upsample", "scale_factor": 2, "mode": "bicubic"},
                },

                {
                    "in_channels": 8*n_ch_0,
                    "out_channels": 4*n_ch_0,
                    "kernel_size": 3,
                    "stride": 1,
                    "padding": 1,

                },

                {
                    "in_channels": 4*n_ch_0,
                    "out_channels": 4*n_ch_0,
                    "kernel_size": 3,
                    "stride": 1,
                    "padding": 1,

                },

                {
                    "in_channels": 4*n_ch_0,
                    "out_channels": 2*n_ch_0,
                    "kernel_size": 3,
                    "stride": 1,
                    "padding": 1,
                    "before_conv": {"type": "upsample", "scale_factor": 2, "mode": "bicubic"},
                },

                {
                    "in_channels": 4*n_ch_0,
                    "out_channels": 2*n_ch_0,
                    "kernel_size": 3,
                    "stride": 1,
                    "padding": 1,
                },

                {
                    "in_channels": 2*n_ch_0,
                    "out_channels": 2*n_ch_0,
                    "kernel_size": 3,
                    "stride": 1,
                    "padding": 1,
                },

                {
                    "in_channels": 2*n_ch_0,
                    "out_channels": 1*n_ch_0,
                    "kernel_size": 3,
                    "stride": 1,
                    "padding": 1,
                    "before_conv": {"type": "upsample", "scale_factor": 2, "mode": "bicubic"},
                },

                {
                    "in_channels": 2*n_ch_0,
                    "out_channels": 1*n_ch_0,
                    "kernel_size": 3,
                    "stride": 1,
                    "padding": 1,
                },

                {
                    "in_channels": 1*n_ch_0,
                    "out_channels": 1*n_ch_0,
                    "kernel_size": 3,
                    "stride": 1,
                    "padding": 1,
                },

                {
                    "in_channels": 1*n_ch_0,
                    "out_channels": n_outputs,
                    "kernel_size": 3,
                    "stride": 1,
                    "padding": 1,
                },

            ]
        },
        "encoder_activations": ["relu", "relu", "relu", "relu", "relu",
                                "relu", "relu", "relu", "relu", "relu",
                                "relu"
                                ],

        "decoder_activations": ["relu", "identity", "relu", "relu", "identity",
                                "relu", "relu", "identity", "relu", "relu",
                                "identity", "relu", "relu", "identity", 
                                ],
    }

    unet = UNet(layers_config=layers,
                intermediary_outputs_indices=[4, 9, 14, 19],
                intermediary_inputs_indices=[4, 10, 16, 22],
                )

    return unet

class TestConvNet2D(TestCase):
    def setUp(self) -> None:
        pass

    def test_convnet_2d_n_parameters(self):
        convnet = model_2d()

        assert type(convnet.n_parameters) == int

    def test_convnet_2d_eval(self):
        input_data, output_data = generate_data(
            n_samples=100, image_size=(16, 16), n_inputs=3, n_outputs=1
        )

        unet = model_2d()
        unet.summary()

        estimated_output_data = unet.eval(input_data=input_data)

        assert estimated_output_data.shape == output_data.shape, (
            "The output of eval is not correct."
            f" Expected {output_data.shape},"
            f" but received {estimated_output_data.shape}."
        )


