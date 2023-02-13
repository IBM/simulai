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
from typing import List, Union
from unittest import TestCase

import matplotlib.pyplot as plt
import numpy as np
import torch

torch.autograd.set_detect_anomaly(True)

# In order to execute this script, it is necessary to
# set the environment variable engine as "pytorch" before initializing
# simulai
os.environ["engine"] = "pytorch"

from simulai.optimization import Optimizer
from simulai.regression import ConvolutionalNetwork, ResConvolutionalNetwork
from simulai.templates import ConvNetworkTemplate


class ResNet(ConvNetworkTemplate):
    def __init__(self, networks_list: List[ConvNetworkTemplate]) -> None:
        super(ResNet, self).__init__()

        self.networks_list = networks_list
        self.weights = list()

        for net in networks_list:
            self.add_module(f"subnetwork_{net.name}", net)
            self.weights += net.weights

    def forward(self, input_data: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        input_data_ = input_data
        output_data = None

        for net in self.networks_list:
            output_data = net.forward(input_data=input_data_)

            input_data_ = output_data

        return output_data


class TestResidualConv(TestCase):
    def setUp(self) -> None:
        pass

    def u(self, t, x, L: float = None, t_max: float = None) -> np.ndarray:
        return np.sin(4 * np.pi * t * (x / L - 1 / 2) ** 2) * np.cos(
            5 * np.pi * (t / t_max - 1 / 2) ** 2
        )

    def test_residual(self):
        K_l = 32
        K = 256
        N = 10_000
        n_inputs = 1
        n_outputs = 1
        lr = 1e-3
        train_fraction = 0.9
        n_samples_train = int(train_fraction * N)
        n_epochs = 30
        t_max = 10
        L = 5

        x_l = np.linspace(0, L, K_l)
        x = np.linspace(0, L, K)
        t = np.linspace(0, t_max, N)

        T_l, X_l = np.meshgrid(t, x_l, indexing="ij")
        T, X = np.meshgrid(t, x, indexing="ij")

        data = self.u(T, X, L=L, t_max=t_max)
        data_l = self.u(T_l, X_l, L=L, t_max=t_max)

        data_l_train = data_l[:n_samples_train]
        data_train = data[:n_samples_train]

        data_l_test = data_l[n_samples_train:]
        data_test = data[n_samples_train:]

        data_l_max, data_l_min = data_l_train.max(), data_l_train.min()
        data_max, data_min = data_train.max(), data_train.min()

        # Normalizing data
        data_l_train = (data_l_train - data_l_min) / (data_l_max - data_l_min)
        data_l_test = (data_l_test - data_l_min) / (data_l_max - data_l_min)
        data_train = (data_train - data_l_min) / (data_l_max - data_l_min)
        data_test = (data_test - data_l_min) / (data_l_max - data_l_min)

        # Subnetworks configurations
        input_convolution_layers = [
            {
                "in_channels": n_inputs,
                "out_channels": 16,
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
                "after_conv": {"type": "upsample", "scale_factor": 2},
            }
        ]

        residual_net_1_layers = [
            [
                {
                    "in_channels": 16,
                    "out_channels": 16,
                    "kernel_size": 3,
                    "stride": 1,
                    "padding": 1,
                },
                {
                    "in_channels": 16,
                    "out_channels": 16,
                    "kernel_size": 3,
                    "stride": 1,
                    "padding": 1,
                },
            ],
            [
                {
                    "in_channels": 16,
                    "out_channels": 16,
                    "kernel_size": 3,
                    "stride": 1,
                    "padding": 1,
                },
                {
                    "in_channels": 16,
                    "out_channels": 16,
                    "kernel_size": 3,
                    "stride": 1,
                    "padding": 1,
                },
            ],
        ]

        transition_convolution_layers_1 = [
            {
                "in_channels": 16,
                "out_channels": 4,
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
                "after_conv": {"type": "upsample", "scale_factor": 2},
            }
        ]

        residual_net_2_layers = [
            [
                {
                    "in_channels": 4,
                    "out_channels": 4,
                    "kernel_size": 3,
                    "stride": 1,
                    "padding": 1,
                },
                {
                    "in_channels": 4,
                    "out_channels": 4,
                    "kernel_size": 3,
                    "stride": 1,
                    "padding": 1,
                },
            ],
            [
                {
                    "in_channels": 4,
                    "out_channels": 4,
                    "kernel_size": 3,
                    "stride": 1,
                    "padding": 1,
                },
                {
                    "in_channels": 4,
                    "out_channels": 4,
                    "kernel_size": 3,
                    "stride": 1,
                    "padding": 1,
                },
            ],
        ]

        transition_convolution_layers_2 = [
            {
                "in_channels": 4,
                "out_channels": 1,
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
                "after_conv": {"type": "upsample", "scale_factor": 2},
            }
        ]

        input_convnet = ConvolutionalNetwork(
            layers=input_convolution_layers,
            activations="relu",
            case="1d",
            name="conv_1",
        )

        residual_net_1 = ResConvolutionalNetwork(
            stages=residual_net_1_layers,
            activations=2 * ["relu"],
            case="1d",
            name="resconv_1",
            last_activation="relu",
        )

        intermediary_convnet = ConvolutionalNetwork(
            layers=transition_convolution_layers_1,
            activations="relu",
            case="1d",
            name="conv_2",
        )

        residual_net_2 = ResConvolutionalNetwork(
            stages=residual_net_2_layers,
            activations=2 * ["relu"],
            case="1d",
            name="resconv_2",
            last_activation="relu",
        )

        output_convnet = ConvolutionalNetwork(
            layers=transition_convolution_layers_2,
            activations="relu",
            case="1d",
            name="conv_3",
        )

        networks_list = [
            input_convnet,
            residual_net_1,
            intermediary_convnet,
            residual_net_2,
            output_convnet,
        ]

        # Configuring model
        n_inputs = 1
        lr = 1e-3  # Initial learning rate for the ADAM algorithm
        n_epochs = 10

        optimizer_config = {"lr": lr}

        resnet = ResNet(networks_list=networks_list)

        resnet.forward(input_data=data_l_train)

        params = {"lambda_1": 0.0, "lambda_2": 0.0}
        optimizer = Optimizer("adam", params=optimizer_config)

        ### Training
        optimizer.fit(
            op=resnet,
            input_data=data_l_train,
            target_data=data_train,
            n_epochs=n_epochs,
            loss="rmse",
            params=params,
            batch_size=100,
            device="gpu",
        )

        ### Evaluating
        data_test_eval = resnet.eval(input_data=data_l_test)
