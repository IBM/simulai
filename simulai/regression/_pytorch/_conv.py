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

import importlib
from typing import List, Union

import numpy as np
import torch

from simulai.templates import ConvNetworkTemplate, as_tensor


def channels_dim(method):
    def inside(self, input_data=None):
        if len(input_data.shape) < self.n_dimensions:
            return method(self, input_data=input_data[:, None, ...])
        else:
            return method(self, input_data=input_data)

    return inside


# High-level class for assembling different kinds of convolutional networks
class ConvolutionalNetwork(ConvNetworkTemplate):
    name = "conv"
    engine = "torch"

    def __init__(
        self,
        layers: list = None,
        activations: list = None,
        case: str = "2d",
        last_activation: str = "identity",
        transpose: bool = False,
        flatten: bool = False,
        name: str = None,
    ) -> None:
        super(ConvolutionalNetwork, self).__init__(name=name, flatten=flatten)

        self.args = ["in_channels", "out_channels", "kernel_size"]

        # The operation coming in the sequence of each convolution layer can be
        # a pooling ou a sampling
        self.before_conv_tag = "before_conv"
        self.after_conv_tag = "after_conv"

        self.default_last_activation = last_activation
        self.layers_config = layers
        self.n_layers = len(layers)
        # The restriction is: (n_samples, n_channels, *dims), with len(dims) = case - 'd'
        self.n_dimensions = int(case.replace("d", "")) + 2
        self.activations, self.activations_str = self._setup_activations(
            activation=activations
        )

        if transpose is True:
            self.convolution_case = "ConvTranspose"
        else:
            self.convolution_case = "Conv"

        self.case = case
        self.type = self.convolution_case + case
        self.nn_module_name = "torch.nn"
        self.nn_engine = importlib.import_module(self.nn_module_name)

        self.layer_template = getattr(self.nn_engine, self.type)

        (
            self.before_conv_layers,
            self.conv_layers,
            self.after_conv_layers,
            self.weights,
        ) = self._setup_layers(layers_config=self.layers_config)

        self.list_of_layers = self._merge(
            before_conv=self.before_conv_layers,
            conv=self.conv_layers,
            act=self.activations,
            after_conv=self.after_conv_layers,
        )

        self.pipeline = torch.nn.Sequential(*self.list_of_layers)

    @as_tensor
    @channels_dim
    def forward(
        self, input_data: Union[torch.Tensor, np.ndarray] = None
    ) -> torch.Tensor:
        return self.flattener(input_data=self.pipeline(input_data))


# Residual Version of the Convolution Networks considering constant dimensions
class ResConvolutionalNetwork(ConvNetworkTemplate):
    def __init__(
        self,
        stages: List[list] = None,
        activations: List[list] = None,
        case: str = "2d",
        last_activation: str = "identity",
        transpose: bool = False,
        name: str = None,
    ) -> None:
        super(ResConvolutionalNetwork, self).__init__(name=name)

        self.args = ["in_channels", "out_channels", "kernel_size"]

        # The operation coming in the sequence of each convolution layer can be
        # a pooling ou a sampling
        self.after_conv_tag = "after_conv"

        self.default_last_activation = last_activation

        # The restriction is: (n_samples, n_channels, *dims), with len(dims) = case - 'd'
        self.n_dimensions = int(case.replace("d", "")) + 2

        if transpose is True:
            self.convolution_case = "ConvTranspose"
        else:
            self.convolution_case = "Conv"

        self.type = self.convolution_case + case
        self.nn_module_name = "torch.nn"
        self.nn_engine = importlib.import_module(self.nn_module_name)

        self.layer_template = getattr(self.nn_engine, self.type)

        self.stages = stages
        self.activations = activations
        self.case = case
        self.last_activation = last_activation
        self.transpose = transpose

        self.weights = list()
        self.nn_layers = list()

        self.blocks = list()

        for stage, activations in zip(self.stages, self.activations):
            (
                before_conv_layers,
                conv_layers,
                after_conv_layers,
                weights,
            ) = self._setup_layers(layers_config=stage)
            activations_, activations_str = self._setup_activations(
                activation=activations, n_layers=len(stage)
            )

            layers_sequence = self._merge(
                before_conv=before_conv_layers,
                conv=conv_layers,
                act=activations_,
                after_conv=after_conv_layers,
            )

            sequence = torch.nn.Sequential(*layers_sequence)
            self.blocks.append(sequence)

            self.weights += weights

    @as_tensor
    @channels_dim
    def forward(
        self, input_data: Union[torch.Tensor, np.ndarray] = None
    ) -> torch.Tensor:
        input_tensor_ = input_data

        for block in self.blocks:
            output_tensor = input_tensor_ + block(input_tensor_)

            input_tensor_ = output_tensor

        return self.flattener(input_data=input_tensor_)
