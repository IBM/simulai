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

import warnings
from typing import Union

import numpy as np
import torch

from simulai.regression._pytorch import LinearNumpy
from simulai.templates import NetworkTemplate, as_tensor

warnings.filterwarnings("ignore", category=UserWarning)

class Linear(NetworkTemplate):

    name = "linear"
    engine = "torch"

    def __init__(
        self,
        input_size: int = None,
        output_size: int = None,
        bias: bool = True,
        name: str = None,
    ) -> None:

        """
        Linear operator F(u) = Au + b

        Parameters
        ----------
        input_size : int
            Dimension of the input.
        output_size : int
            Dimension of the output.
        bias : bool
            Using bias tensor or not.
        name : str
            A name for identifying the model.
        """

        super(Linear, self).__init__(name=name)

        self.input_size = input_size
        self.output_size = output_size

        self.activations_str = None

        self.layers = [torch.nn.Linear(input_size, output_size, bias=bias)]

        self.add_module(self.name + "_" + "linear_op", self.layers[0])

        self.weights = [item.weight for item in self.layers]

        self.bias = [item.bias for item in self.layers]

        self.name = name

    @as_tensor
    def forward(
        self, input_data: Union[torch.Tensor, np.ndarray] = None
    ) -> torch.Tensor:

        """
        Applying the operator Linear.

        Parameters
        ----------
        input_data : Union[torch.Tensor, np.ndarray]
            Data to be processed using Linear.

        Returns
        -------
            The output after the application of Linear.
        """

        return self.layers[0](input_data)

    def to_numpy(self):

        """
        It converts the tensors in Linear to numpy.ndarray. 

        Returns
        -------
            An object LinearNumpy, which reproduces Linear. 
        """

        return LinearNumpy(layer=self.layers[0], name=self.name)


class SLFNN(Linear):
    def __init__(
        self,
        input_size: int = None,
        output_size: int = None,
        bias: bool = True,
        name: str = None,
        activation: str = "tanh",
    ) -> None:
        """
        Single layer fully-connected (dense) neural network

        Parameters
        ----------
        input_size : int
            Dimension of the input.
        output_size : int
            Dimension of the output.
        bias : bool
            Using bias tensor or not. 
        name : str
            A name for identifying the model.
        activation : str
            Activation function.
        """

        super(SLFNN, self).__init__(
            input_size=input_size, output_size=output_size, bias=bias, name=name
        )

        self.activation = self._get_operation(operation=activation)

    def forward(
        self, input_data: Union[torch.Tensor, np.ndarray] = None
    ) -> torch.Tensor:

        """
        Applying the operator Linear.

        Parameters
        ----------
        input_data : Union[torch.Tensor, np.ndarray]
            Data to be processed using SLFNN.

        Returns
        -------
            The output after the application of SLFNN.
        """


        return self.activation(super().forward(input_data=input_data))


class ShallowNetwork(SLFNN):
    def __init__(
        self,
        input_size: int = None,
        hidden_size: int = None,
        output_size: int = None,
        bias: bool = True,
        name: str = None,
        activation: str = "tanh",
    ) -> None:

        """
        ELM-like (Extreme Learning Machine) shallow network

        Parameters
        ----------
        input_size : int
            Dimension of the input.
        hidden_size : int
            Dimension of the hidden (intermediary) state.
        output_size : int
            Dimension of the output.
        bias : bool
            Using bias or not for the last layer.
        name : str
            A name for identifying the model. 
        activation : str
            Activation function.
        """

        super(ShallowNetwork, self).__init__(
            input_size=input_size, output_size=hidden_size, bias=bias, name=name
        )

        self.output_layer = Linear(
            input_size=hidden_size, output_size=output_size, bias=False, name="output"
        )

        self.output_size = output_size

    def forward(
        self, input_data: Union[torch.Tensor, np.ndarray] = None
    ) -> torch.Tensor:
        """

        Parameters
        ----------
        input_data : Union[torch.Tensor, np.ndarray]
            Input data to be processed using ShallowNetwork.
        Returns
        -------
            The output of ShallowNetwork.
        """

        hidden_state = self.activation(super().forward(input_data=input_data))

        return self.output_layer.forward(input_data=hidden_state)


# Dense (fully-connected) neural network written in PyTorch
class DenseNetwork(NetworkTemplate):
    name = "dense"
    engine = "torch"

    def __init__(
        self,
        layers_units: list = None,
        activations: Union[list, str] = None,
        input_size: int = None,
        output_size: int = None,
        normalization: str = "bypass",
        name: str = "",
        last_bias: bool = True,
        last_activation: str = "identity",
        **kwargs,
    ) -> None:

        """
        Dense (fully-connected) neural network written in PyTorch

        Parameters
        ----------
        layers_units : list
            List with the number of neurons for each layer.
        activations : Union[list, str]
            List of activations for each layer or a single string 
            informing the activation used for all of them.
        input_size : int
            Dimension of the input.
        output_size : int
            Dimension of the output.
        normalization : str
            Kind of normalization used between two layers.
        name : str
            A name for identifying the model.
        last_bias : bool
            Using bias in the last layer or not.
        last_activation : str
            Activation for the last layer (default is 'identity').
        """

        super(DenseNetwork, self).__init__()

        assert layers_units, "Please, set a list of units for each layer"

        assert activations, (
            "Please, set a list of activation functions" "or a string for all of them."
        )

        # These activations support gain evaluation for the initial state
        self.gain_supported_activations = ["sigmoid", "tanh", "relu", "leaky_relu"]

        # Default attributes
        self.layers_units = layers_units
        self.input_size = input_size
        self.output_size = output_size
        self.normalization = normalization
        self.name = name
        self.last_bias = last_bias

        # For extra and not ever required parameters
        for k, v in kwargs.items():
            setattr(self, k, v)

        # Getting up parameters from host
        self._get_from_guest(activation=activations)

        self.weights = list()

        # The total number of layers includes the output layer
        self.n_layers = len(self.layers_units) + 1

        self.default_last_activation = last_activation

        self.activations, self.activations_str = self._setup_activations(
            activation=activations
        )

        self.initializations = [
            self._determine_initialization(activation)
            for activation in self.activations_str
        ]

        self.layers = self._setup_hidden_layers(last_bias=last_bias)

        array_layers = self._numpy_layers()
        n_layers = len(self.layers)

        self.shapes = [item.shape for item in list(sum(array_layers, []))]

        self.stitch_idx = self._make_stitch_idx()

        self.layers_map = [[ll, ll + 1] for ll in range(0, 2 * n_layers, 2)]

    def _calculate_gain(self, activation: str = "Tanh") -> float:

        """
        It evaluates a multiplier coefficient, named as `gain`, 
        which is used to enhance the funcionality of each kind of activation
        function.

        Parameters
        ----------
        activation : str
            Activation function.
        Returns
        -------
            The value of the gain associated to that activation.
        """

        if type(activation) is not str:
            assert hasattr(
                activation, "name"
            ), f"Activation object {type(activation)} must have attribute ´name´."
            name = getattr(activation, "name")
        else:
            name = activation

        if name.lower() in self.gain_supported_activations:
            return torch.nn.init.calculate_gain(name.lower())
        else:
            return 1

    @staticmethod
    def _determine_initialization(activation: str = "Tanh") -> str:

        """
        It determines the most proper initialization method for each 
        activation function.

        Parameters
        ----------
        activation : str
            Activation function.

        Returns
        -------
            The initialization method most indicated for that activation.
        """

        if type(activation) is not str:
            assert hasattr(
                activation, "name"
            ), f"Activation object {type(activation)} must have attribute ´name´."
            name = getattr(activation, "name")
        else:
            name = activation

        if name in ["ReLU"]:
            return "kaiming"
        elif name == "Siren":
            return "siren"
        else:
            return "xavier"

    def _setup_layer(
        self,
        input_size: int = 0,
        output_size: int = 0,
        initialization: str = None,
        bias: bool = True,
        first_layer: bool = False,
    ) -> torch.nn.Linear:

        """

        Parameters
        ----------
        input_size : int
            Dimension of the input.
        output_size : int
            Dimension of the output.
        initialization : str
            Initialization method.
        bias : bool
            Using bias tensor or not.
        first_layer : bool
            Is this layer the first layer or not.

        Returns
        -------
            An instance of torch.nn.Linear.
        """

        # It instantiates a linear operation
        # f: y^l = f(x^(l-1)) = (W^l).dot(x^(l-1)) + b^l
        layer = torch.nn.Linear(input_size, output_size, bias=bias)

        if initialization == "xavier":
            torch.nn.init.xavier_normal_(
                layer.weight, gain=self._calculate_gain(self.activations_str[0])
            )
            return layer

        # The Siren initialization requires some special consideration
        elif initialization == "siren":
            assert (
                self.c is not None
            ), "When using siren, the parameter c must be defined."
            assert (
                self.omega_0 is not None
            ), "When using siren, the parameter omega_0 must be defined."

            if first_layer == True:
                m = 1 / input_size
            else:
                m = np.sqrt(self.c / input_size) / self.omega_0

            torch.nn.init.trunc_normal_(layer.weight, a=-m, b=m)
            b = np.sqrt(1 / input_size)
            torch.nn.init.trunc_normal_(layer.bias, a=-b, b=b)
            return layer

        elif initialization == "kaiming":
            return layer  # Kaiming is the default initialization in PyTorch

        else:
            print(
                "Initialization method still not implemented.\
                  Using Kaiming instead"
            )

            return layer

    # The forward step of the network
    @as_tensor
    def forward(
        self, input_data: Union[torch.Tensor, np.ndarray] = None
    ) -> torch.Tensor:

        """
        It executes the forward step for the DenseNetwork.

        Parameters
        ----------
        input_data : Union[torch.Tensor, np.ndarray]
            The input tensor to be processed by DenseNetwork.

        Returns
        -------
            A torch.Tensor object representing the output of the 
            network. 
        """

        input_tensor_ = input_data

        # TODO It can be done using the PyTorch Sequential object
        for layer_id in range(len(self.layers)):
            output_tensor_ = self.layers[layer_id](input_tensor_)
            _output_tensor_ = self.activations[layer_id](output_tensor_)
            input_tensor_ = _output_tensor_

        output_tensor = input_tensor_

        return output_tensor


class ResDenseNetwork(DenseNetwork):
    name = "residualdense"
    engine = "torch"

    def __init__(
        self,
        layers_units: list = None,
        activations: Union[list, str] = None,
        input_size: int = None,
        output_size: int = None,
        normalization: str = "bypass",
        name: str = "",
        last_bias: bool = True,
        last_activation: str = "identity",
        residual_size: int = 1,
        **kwargs,
    ) -> None:

        """
        Residual Dense (fully-connected) neural network written in PyTorch

        Parameters
        ----------
        layers_units : list
            List with the number of neurons for each layer.
        activations : Union[list, str]
            List of activations for each layer or a single string 
            informing the activation used for all of them.
        input_size : int
            Dimension of the input.
        output_size : int
            Dimension of the output.
        normalization : str
            Kind of normalization used between two layers.
        name : str
            A name for identifying the model.
        last_bias : bool
            Using bias in the last layer or not.
        last_activation : str
            Activation for the last layer (default is 'identity').
        residual_size : int
            Size of the residual block. 
        """

        super().__init__(
            layers_units=layers_units,
            activations=activations,
            input_size=input_size,
            output_size=output_size,
            normalization=normalization,
            name=name,
            last_bias=last_bias,
            last_activation=last_activation,
            **kwargs,
        )

        # Considering the activations layers
        self.residual_size = 2 * residual_size
        self.ratio = 0.5

        # Excluding the input and output layers
        merged_layers = self._merge(layer=self.layers, act=self.activations)

        assert len(merged_layers[2:-2]) % self.residual_size == 0, (
            "The number of layers must be divisible"
            " by the residual block size,"
            f" but received {len(merged_layers)} and {residual_size}"
        )

        self.n_residual_blocks = int(len(merged_layers[2:-2]) / self.residual_size)

        sub_layers = [
            item.tolist()
            for item in np.split(np.array(merged_layers[2:-2]), self.n_residual_blocks)
        ]

        self.input_block = torch.nn.Sequential(*merged_layers[:2])
        self.hidden_blocks = [torch.nn.Sequential(*item) for item in sub_layers]
        self.output_block = torch.nn.Sequential(*merged_layers[-2:])

    # Merging the layers into a reasonable sequence
    def _merge(self, layer: list = None, act: list = None) -> list:

        """
        It merges the dense layers and the activations into a single block.

        Parameters
        ----------
        layer : list
            List of dense layers.
        act : list
            List of activation functions.

        Returns
        -------
            The merged list of layers and activations.
        """

        merged_list = list()

        for i, j in zip(layer, act):
            merged_list.append(i)
            merged_list.append(j)

        return merged_list

    def summary(self):

        """
        It prints a summary of the network.
        """

        super().summary()

        print("Residual Blocks:\n")

        print(self.input_block)
        print(self.hidden_blocks)
        print(self.output_block)

    @as_tensor
    def forward(
        self, input_data: Union[torch.Tensor, np.ndarray] = None
    ) -> torch.Tensor:

        """
        Parameters
        ----------
        input_data : Union[torch.Tensor, np.ndarray]
            Input data to be processed using ResDenseNetwork.
        Returns
        -------
            The output of ResDenseNetwork.
        """

        input_tensor_ = input_data

        input_tensor_ = self.input_block(input_tensor_)

        for block in self.hidden_blocks:
            output_tensor_ = self.ratio * (input_tensor_ + block(input_tensor_))

            input_tensor_ = output_tensor_

        output_tensor = self.output_block(input_tensor_)

        return output_tensor


class ConvexDenseNetwork(DenseNetwork):
    name = "convexdense"
    engine = "torch"

    def __init__(
        self,
        layers_units: list = None,
        activations: Union[list, str] = None,
        input_size: int = None,
        output_size: int = None,
        normalization: str = "bypass",
        name: str = "",
        last_bias: bool = True,
        last_activation: str = "identity",
        **kwargs,
    ) -> None:

        """
        Dense network with convex combinations in the hidden layers.
        This architecture is useful when combined to the Improved Version of DeepONets

        Parameters
        ----------
        layers_units : list
            List with the number of neurons for each layer.
        activations : Union[list, str]
            List of activations for each layer or a single string 
            informing the activation used for all of them.
        input_size : int
            Dimension of the input.
        output_size : int
            Dimension of the output.
        normalization : str
            Kind of normalization used between two layers.
        name : str
            A name for identifying the model.
        last_bias : bool
            Using bias in the last layer or not.
        last_activation : str
            Activation for the last layer (default is 'identity').
        """

        self.hidden_size = None
        assert self._check_regular_net(layers_units=layers_units), (
            "All the hidden layers must be equal in" "a Convex Dense Network."
        )

        super().__init__(
            layers_units=layers_units,
            activations=activations,
            input_size=input_size,
            output_size=output_size,
            normalization=normalization,
            name=name,
            last_bias=last_bias,
            last_activation=last_activation,
            **kwargs,
        )

    def _check_regular_net(self, layers_units: list) -> bool:

        """
        It checks if all the layers has the same number of neurons.

        Parameters
        ----------
        layers_units : list
            List of layers sizes.
        Returns
        -------
            If the network is regular or not.
        """

        mean = int(sum(layers_units) / len(layers_units))
        self.hidden_size = mean

        if len([True for j in layers_units if j == mean]) == len(layers_units):
            return True
        else:
            return False

    @as_tensor
    def forward(
        self,
        input_data: Union[torch.Tensor, np.ndarray] = None,
        u: Union[torch.Tensor, np.ndarray] = None,
        v: Union[torch.Tensor, np.ndarray] = None,
    ) -> torch.Tensor:

        """

        Parameters
        ----------
        input_data : Union[torch.Tensor, np.ndarray]
            Input data to be processed using ConvexDenseNetwork.
        u : Union[torch.Tensor, np.ndarray]
            Input generated by the first auxiliar encoder (external model).
        v : Union[torch.Tensor, np.ndarray]
            Input generated by the second auxiliar encoder (external model).

        Returns
        -------
            The output of ConvexDenseNetwork.
        """

        input_tensor_ = input_data

        # The first layer operation has no difference from the Vanilla one
        first_output = self.activations[0](self.layers[0](input_tensor_))

        input_tensor_ = first_output

        layers_hidden = self.layers[1:-1]
        activations_hidden = self.activations[1:-1]

        for layer_id in range(len(layers_hidden)):
            output_tensor_ = layers_hidden[layer_id](input_tensor_)
            z = activations_hidden[layer_id](output_tensor_)
            _output_tensor_ = (1 - z) * u + z * v

            input_tensor_ = _output_tensor_

        # The last layer operation too
        last_output = self.activations[-1](self.layers[-1](input_tensor_))
        output_tensor = last_output

        return output_tensor
