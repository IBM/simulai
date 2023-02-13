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
import sys
from typing import Union
from unittest import TestCase

import matplotlib.pyplot as plt
import numpy as np
import torch

# In order to execute this script, it is necessary to
# set the environment variable engine as "pytorch" before initializing
# simulai
os.environ["engine"] = "pytorch"

from simulai.optimization import Optimizer
from simulai.regression import ConvolutionalNetwork, Linear
from simulai.templates import NetworkTemplate


class Autoencoder1D(NetworkTemplate):
    def __init__(
        self,
        encoder: ConvolutionalNetwork,
        bottleneck_encoder: Linear,
        bottleneck_decoder: Linear,
        decoder: ConvolutionalNetwork,
    ) -> None:
        super(Autoencoder1D, self).__init__()

        self.weights = list()

        self.encoder = encoder
        self.bottleneck_encoder = bottleneck_encoder
        self.botleneck_decoder = bottleneck_decoder
        self.decoder = decoder

        self.add_module("encoder", self.encoder)
        self.add_module("bottleneck_encoder", self.bottleneck_encoder)
        self.add_module("bottleneck_decoder", self.botleneck_decoder)
        self.add_module("decoder", self.decoder)

        self.weights += self.encoder.weights
        self.weights += self.bottleneck_encoder.weights
        self.weights += self.bottleneck_decoder.weights
        self.weights += self.decoder.weights

        self.last_encoder_channels = None

    def summary(
        self, input_data: Union[np.ndarray, torch.Tensor] = None
    ) -> torch.Tensor:
        btnk_input = self.encoder.forward(input_data=input_data)

        self.encoder.summary(input_data=input_data)

        self.last_encoder_channels = btnk_input.shape[1]

        btnk_input = btnk_input.reshape((-1, np.prod(btnk_input.shape[1:])))

        latent = self.bottleneck_encoder.forward(input_data=btnk_input)

        self.bottleneck_encoder.summary()
        self.bottleneck_decoder.summary()

        bottleneck_output = torch.nn.ReLU()(
            self.bottleneck_decoder.forward(input_data=latent)
        )

        dimension = int(bottleneck_output.shape[-1] / self.last_encoder_channels)
        bottleneck_output = bottleneck_output.reshape(
            (-1, self.last_encoder_channels, dimension)
        )

        self.decoder.summary(input_data=bottleneck_output)

    def projection(self, input_data: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        btnk_input = self.encoder.forward(input_data=input_data)
        self.last_encoder_channels = btnk_input.shape[1]

        btnk_input = btnk_input.reshape((-1, np.prod(btnk_input.shape[1:])))

        latent = self.bottleneck_encoder.forward(input_data=btnk_input)
        return latent

    def reconstruction(
        self, input_data: Union[torch.Tensor, np.ndarray]
    ) -> torch.Tensor:
        bottleneck_output = torch.nn.ReLU()(
            self.bottleneck_decoder.forward(input_data=input_data)
        )

        dimension = int(bottleneck_output.shape[-1] / self.last_encoder_channels)
        bottleneck_output = bottleneck_output.reshape(
            (-1, self.last_encoder_channels, dimension)
        )

        reconstructed = self.decoder.forward(input_data=bottleneck_output)

        return reconstructed

    def forward(self, input_data: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        latent = self.projection(input_data=input_data)
        reconstructed = self.reconstruction(input_data=latent)

        return reconstructed


class TestAutoencoder(TestCase):
    def setUp(self) -> None:
        pass

    def test_autoencoder(self):
        K = 512
        N = 10_000
        n_inputs = 1
        n_outputs = 1
        lr = 1e-3
        train_fraction = 0.9
        n_samples_train = int(train_fraction * N)
        n_epochs = 30
        t_max = 10
        L = 5
        x = np.linspace(0, L, K)
        t = np.linspace(0, t_max, N)
        T, X = np.meshgrid(t, x, indexing="ij")

        # data = np.sin(4*np.pi*T*(X/L - 1/2)**2)*np.cos(5*np.pi*(T/t_max - 1/2)**2)
        data = np.sin(4 * np.pi * T) * np.cos(5 * np.pi * X)

        data_train = data[:n_samples_train]
        data_test = data[n_samples_train:]

        encoder_layers = [
            {
                "in_channels": n_inputs,
                "out_channels": 16,
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
                "after_conv": {"type": "maxpool1d", "kernel_size": 2, "stride": 2},
            },
            {
                "in_channels": 16,
                "out_channels": 8,
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
                "after_conv": {"type": "maxpool1d", "kernel_size": 2, "stride": 2},
            },
            {
                "in_channels": 8,
                "out_channels": 4,
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
                "after_conv": {"type": "maxpool1d", "kernel_size": 2, "stride": 2},
            },
            {
                "in_channels": 4,
                "out_channels": 2,
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
                "after_conv": {"type": "maxpool1d", "kernel_size": 2, "stride": 2},
            },
        ]

        bottleneck_encoder_layers = {
            "input_size": 64,
            "output_size": 16,
            "name": "bottleneck_encoder",
        }

        bottleneck_decoder_layers = {
            "input_size": 16,
            "output_size": 64,
            "name": "bottleneck_decoder",
        }

        decoder_layers = [
            {"in_channels": 2, "out_channels": 4, "kernel_size": 2, "stride": 2},
            {"in_channels": 4, "out_channels": 8, "kernel_size": 2, "stride": 2},
            {"in_channels": 8, "out_channels": 16, "kernel_size": 2, "stride": 2},
            {"in_channels": 16, "out_channels": 8, "kernel_size": 2, "stride": 2},
            {
                "in_channels": 8,
                "out_channels": n_outputs,
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
            },
        ]

        # Instantiating network
        encoder = ConvolutionalNetwork(
            layers=encoder_layers, activations="relu", case="1d", name="encoder"
        )
        bottleneck_encoder = Linear(**bottleneck_encoder_layers)
        bottleneck_decoder = Linear(**bottleneck_decoder_layers)
        decoder = ConvolutionalNetwork(
            layers=decoder_layers,
            activations="relu",
            case="1d",
            transpose=True,
            name="decoder",
        )

        autoencoder = Autoencoder1D(
            encoder=encoder,
            bottleneck_encoder=bottleneck_encoder,
            bottleneck_decoder=bottleneck_decoder,
            decoder=decoder,
        )

        autoencoder.summary(input_data=data_train)

        optimizer_config = {"lr": lr}
        params = {"lambda_1": 0.0, "lambda_2": 0.0}

        optimizer = Optimizer("adam", params=optimizer_config)

        optimizer.fit(
            op=autoencoder,
            input_data=data_train,
            target_data=data_train,
            n_epochs=n_epochs,
            loss="rmse",
            params=params,
            batch_size=5_00,
        )

        approximated_data = autoencoder.eval(input_data=data_test)

        plt.pcolormesh(
            X[n_samples_train:], T[n_samples_train:], approximated_data[:, 0, ...]
        )
        plt.show()

        plt.pcolormesh(X[n_samples_train:], T[n_samples_train:], data_test)
        plt.show()
