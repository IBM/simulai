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

from typing import Optional, Tuple, Union, List

import numpy as np
import torch

from simulai import ARRAY_DTYPE
from simulai.regression import ConvolutionalNetwork, DenseNetwork, Linear
from simulai.templates import (
    NetworkTemplate,
    as_tensor,
    autoencoder_auto,
    cnn_autoencoder_auto,
    mlp_autoencoder_auto,
)


########################################
### Some usual AutoEncoder architectures
########################################
class AutoencoderMLP(NetworkTemplate):
    """
    This is an implementation of a Fully-connected AutoEncoder as
    Reduced Order Model;

    A MLP autoencoder architecture consists of two stages:
    --> Fully-connected encoder
    --> Fully connected decoder

    SCHEME:
            |         |
            |  |   |  |
    Z ->    |  | | |  |  -> Z_til
            |  |   |  |
            |         |

    ENCODER       DECODER
    """

    def __init__(
        self,
        encoder: DenseNetwork = None,
        decoder: DenseNetwork = None,
        input_dim: Optional[int] = None,
        output_dim: Optional[int] = None,
        latent_dim: Optional[int] = None,
        activation: Optional[Union[list, str]] = None,
        shallow: Optional[bool] = False,
        devices: Union[str, list] = "cpu",
        name: str = None,
    ) -> None:
        """
        Initialize the AutoencoderMLP network

        Parameters
        ----------
        encoder : DenseNetwork
            The encoder network architecture.
        decoder : DenseNetwork
            The decoder network architecture.
        input_dim : int, optional
            The input dimensions of the data, by default None.
        output_dim : int, optional
            The output dimensions of the data, by default None.
        latent_dim : int, optional
            The dimensions of the latent space, by default None.
        activation : Union[list, str], optional
            The activation functions used by the network, by default None.
        shallow : bool, optional
            Whether the network should be shallow or not, by default False.
        devices : Union[str, list], optional
            The device(s) to be used for allocating subnetworks, by default "cpu".
        name : str, optional
            The name of the network, by default None.
        """

        super(AutoencoderMLP, self).__init__(name=name)

        self.weights = list()

        # This option is used when no network is provided
        # and it uses default choices for the architectures
        if encoder == None and decoder == None:
            encoder, decoder = mlp_autoencoder_auto(
                input_dim=input_dim,
                latent_dim=latent_dim,
                output_dim=output_dim,
                activation=activation,
                shallow=shallow,
            )

        # Determining the kind of device to be used for allocating the
        # subnetworks used in the DeepONet model
        self.device = self._set_device(devices=devices)

        self.encoder = self.to_wrap(entity=encoder, device=self.device)
        self.decoder = self.to_wrap(entity=decoder, device=self.device)

        self.add_module("encoder", self.encoder)
        self.add_module("decoder", self.decoder)

        self.weights += self.encoder.weights
        self.weights += self.decoder.weights

        self.last_encoder_channels = None

        self.shapes_dict = dict()

    def summary(self) -> None:
        """
        Prints the summary of the network architecture
        """
        self.encoder.summary()
        self.decoder.summary()

    def projection(
        self, input_data: Union[np.ndarray, torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Project the input dataset into the latent space.

        Parameters
        ----------
        input_data : Union[np.ndarray, torch.Tensor], optional
            The dataset to be projected, by default None.

        Returns
        -------
        torch.Tensor
            The dataset projected over the latent space.

        """
        latent = self.encoder.forward(input_data=input_data)

        return latent

    def reconstruction(
        self, input_data: Union[torch.Tensor, np.ndarray] = None
    ) -> torch.Tensor:
        """
        Reconstruct the latent dataset to the original one.

        Parameters
        ----------
        input_data : Union[np.ndarray, torch.Tensor], optional
            The dataset to be reconstructed, by default None.

        Returns
        -------
        torch.Tensor
            The dataset reconstructed.

        """
        reconstructed = self.decoder.forward(input_data=input_data)

        return reconstructed

    def forward(
        self, input_data: Union[np.ndarray, torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Execute the complete projection/reconstruction pipeline.

        Parameters
        ----------
        input_data : Union[np.ndarray, torch.Tensor], optional
            The input dataset, by default None.

        Returns
        -------
        torch.Tensor
            The dataset reconstructed.

        """
        latent = self.projection(input_data=input_data)
        reconstructed = self.reconstruction(input_data=latent)

        return reconstructed

    def eval_projection(
        self, input_data: Union[np.ndarray, torch.Tensor] = None
    ) -> np.ndarray:
        """
        Evaluate the projection of the input dataset into the latent space.

        Parameters
        ----------
        input_data : Union[np.ndarray, torch.Tensor], optional
            The dataset to be projected, by default None.

        Returns
        -------
        np.ndarray
            The dataset projected over the latent space.

        """
        return self.projection(input_data=input_data).detach().numpy()


class AutoencoderCNN(NetworkTemplate):
    """
    This is an implementation of a convolutional autoencoder as Reduced Order Model.
    An autoencoder architecture consists of three stages:

    * The convolutional encoder

    The bottleneck stage, subdivided in:
        * Fully-connected encoder
        * Fully connected decoder
        * The convolutional decoder

    SCHEME:

    Z -> [Conv] -> [Conv] -> ... [Conv] -> |  | | |  | -> [Conv.T] -> [Conv.T] -> ... [Conv.T] -> Z_til


                    ENCODER               DENSE BOTTLENECK           DECODER
    """

    def __init__(
        self,
        encoder: ConvolutionalNetwork = None,
        bottleneck_encoder: Linear = None,
        bottleneck_decoder: Linear = None,
        decoder: ConvolutionalNetwork = None,
        encoder_activation: str = "relu",
        input_dim: Optional[Tuple[int, ...]] = None,
        output_dim: Optional[Tuple[int, ...]] = None,
        latent_dim: Optional[int] = None,
        kernel_size: Optional[int] = None,
        activation: Optional[Union[list, str]] = None,
        channels: Optional[int] = None,
        case: Optional[str] = None,
        shallow: Optional[bool] = False,
        devices: Union[str, list] = "cpu",
        name: str = None,
        **kwargs,
    ) -> None:
        """
        Initialize the AutoencoderCNN network.

        Parameters
        ----------
        encoder : ConvolutionalNetwork, optional
            The encoder network architecture, by default None.
        bottleneck_encoder : Linear, optional
            The bottleneck encoder network architecture, by default None.
        bottleneck_decoder : Linear, optional
            The bottleneck decoder network architecture, by default None.
        decoder : ConvolutionalNetwork, optional
            The decoder network architecture, by default None.
        encoder_activation : str, optional
            The activation function used by the encoder network, by default 'relu'.
        input_dim : Tuple[int, ...], optional
            The input dimensions of the data, by default None.
        output_dim : Tuple[int, ...], optional
            The output dimensions of the data, by default None.
        latent_dim : int, optional
            The dimensions of the latent space, by default None.
        activation : Union[list, str], optional
            The activation functions used by the network, by default None.
        channels : int, optional
            The number of channels of the convolutional layers, by default None.
        case : str, optional
            The type of convolutional encoder and decoder to be used, by default None.
        shallow : bool, optional
            Whether the network should be shallow or not, by default False.
        devices : Union[str, list], optional
            The device(s) to be used for allocating subnetworks, by default 'cpu'.
        name : str, optional
            The name of the network, by default None.
        """

        super(AutoencoderCNN, self).__init__(name=name)

        self.weights = list()

        # Determining the kind of device to be used for allocating the
        # subnetworks
        self.device = self._set_device(devices=devices)

        self.input_dim = None

        # If not network is provided, the automatic generation
        # pipeline is activated.
        if all(
            [
                isn == None
                for isn in [encoder, decoder, bottleneck_encoder, bottleneck_decoder]
            ]
        ):
            self.input_dim = input_dim

            (
                encoder,
                decoder,
                bottleneck_encoder,
                bottleneck_decoder,
            ) = cnn_autoencoder_auto(
                input_dim=input_dim,
                latent_dim=latent_dim,
                output_dim=output_dim,
                activation=activation,
                kernel_size=kernel_size,
                channels=channels,
                case=case,
                shallow=shallow,
            )

        self.encoder = self.to_wrap(entity=encoder, device=self.device)
        self.bottleneck_encoder = self.to_wrap(entity=bottleneck_encoder, device=self.device)
        self.bottleneck_decoder = self.to_wrap(entity=bottleneck_decoder, device=self.device)
        self.decoder = self.to_wrap(entity=decoder, device=self.device)

        self.add_module("encoder", self.encoder)
        self.add_module("bottleneck_encoder", self.bottleneck_encoder)
        self.add_module("bottleneck_decoder", self.bottleneck_decoder)
        self.add_module("decoder", self.decoder)

        self.weights += self.encoder.weights
        self.weights += self.bottleneck_encoder.weights
        self.weights += self.bottleneck_decoder.weights
        self.weights += self.decoder.weights

        self.last_encoder_channels = None
        self.before_flatten_dimension = None

        self.encoder_activation = self._get_operation(operation=encoder_activation)

        self.shapes_dict = dict()

    def summary(
        self,
        input_data: Union[np.ndarray, torch.Tensor] = None,
        input_shape: list = None,
        verbose: bool = True,
    ) -> torch.Tensor:
        """
        Prints the summary of the network architecture.

        Parameters
        ----------
        input_data : np.ndarray or torch.Tensor
            The input dataset.
        input_shape : list, optional
            The shape of the input data.

        Returns
        -------
        torch.Tensor
            The dataset projected over the latent space.
        """

        if verbose == True:
            if self.input_dim != None:
                input_shape = self.input_dim
            else:
                pass

            self.encoder.summary(
                input_data=input_data, input_shape=input_shape, device=self.device
            )

            if isinstance(input_data, np.ndarray):
                btnk_input = self.encoder.forward(input_data=input_data)
            else:
                assert (
                    input_shape
                ), "It is necessary to have input_shape when input_data is None."
                input_shape = self.encoder.input_size
                input_shape[0] = 1

                input_data = self.to_wrap(entity=torch.ones(input_shape), device=self.device)

                btnk_input = self.encoder.forward(input_data=input_data)

            before_flatten_dimension = tuple(btnk_input.shape[1:])
            btnk_input = btnk_input.reshape((-1, np.prod(btnk_input.shape[1:])))

            latent = self.bottleneck_encoder.forward(input_data=btnk_input)

            self.bottleneck_encoder.summary()
            self.bottleneck_decoder.summary()

            bottleneck_output = self.encoder_activation(
                self.bottleneck_decoder.forward(input_data=latent)
            )

            bottleneck_output = bottleneck_output.reshape(
                (-1, *before_flatten_dimension)
            )

            self.decoder.summary(input_data=bottleneck_output, device=self.device)

            # Saving the content of the subnetworks to the overall architecture dictionary
            self.shapes_dict.update({"encoder": self.encoder.shapes_dict})
            self.shapes_dict.update(
                {"bottleneck_encoder": self.bottleneck_encoder.shapes_dict}
            )
            self.shapes_dict.update(
                {"bottleneck_decoder": self.bottleneck_decoder.shapes_dict}
            )
            self.shapes_dict.update({"decoder": self.decoder.shapes_dict})

        else:
            print(self)

    @as_tensor
    def projection(self, input_data: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Project input dataset into the latent space.

        Parameters
        ----------
        input_data : Union[np.ndarray, torch.Tensor]
            The dataset to be projected.

        Returns
        -------
        torch.Tensor
            The dataset projected over the latent space.

        """

        btnk_input = self.encoder.forward(input_data=input_data)

        self.before_flatten_dimension = tuple(btnk_input.shape[1:])

        btnk_input = btnk_input.reshape((-1, np.prod(self.before_flatten_dimension)))

        latent = self.bottleneck_encoder.forward(input_data=btnk_input)

        return latent

    @as_tensor
    def reconstruction(
        self, input_data: Union[torch.Tensor, np.ndarray]
    ) -> torch.Tensor:
        """
        Reconstruct the latent dataset to the original one.

        Parameters
        ----------
        input_data : Union[np.ndarray, torch.Tensor]
            The dataset to be reconstructed.

        Returns
        -------
        torch.Tensor
            The reconstructed dataset.

        """

        bottleneck_output = self.encoder_activation(
            self.bottleneck_decoder.forward(input_data=input_data)
        )

        bottleneck_output = bottleneck_output.reshape(
            (-1,) + self.before_flatten_dimension
        )

        reconstructed = self.decoder.forward(input_data=bottleneck_output)

        return reconstructed

    def forward(self, input_data: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Execute the complete projection/reconstruction pipeline.

        Parameters
        ----------
        input_data : Union[np.ndarray, torch.Tensor]
            The input dataset.

        Returns
        -------
        torch.Tensor
            The reconstructed dataset.
        """

        latent = self.projection(input_data=input_data)
        reconstructed = self.reconstruction(input_data=latent)

        return reconstructed

    def eval(self, input_data: Union[np.ndarray, torch.Tensor] = None) -> np.ndarray:
        """
        Evaluate the autoencoder on the given dataset.

        Parameters
        ----------
        input_data : Union[np.ndarray, torch.Tensor], optional
            The dataset to be evaluated, by default None.

        Returns
        -------
        np.ndarray
            The dataset projected over the latent space.
        """

        if isinstance(input_data, np.ndarray):
            input_data = torch.from_numpy(input_data.astype(ARRAY_DTYPE))

        input_data = input_data.to(self.device)

        return super().eval(input_data=input_data)

    def project(self, input_data: Union[np.ndarray, torch.Tensor] = None) -> np.ndarray:
        """
        Project the input dataset into the latent space.

        Parameters
        ----------
        input_data : Union[np.ndarray, torch.Tensor], optional
            The dataset to be projected, by default None.

        Returns
        -------
        np.ndarray
            The dataset projected over the latent space.
        """

        projected_data = self.projection(input_data=input_data)

        return projected_data.cpu().detach().numpy()

    def reconstruct(
        self, input_data: Union[np.ndarray, torch.Tensor] = None
    ) -> np.ndarray:
        """
        Reconstructs the latent dataset to the original one.

        Parameters
        ----------
        input_data : Union[np.ndarray, torch.Tensor], optional
            The dataset to be reconstructed. If not provided, uses the original input data, by default None.

        Returns
        -------
        np.ndarray
            The reconstructed dataset.

        """
        reconstructed_data = self.reconstruction(input_data=input_data)
        return reconstructed_data.cpu().detach().numpy()


class AutoencoderKoopman(NetworkTemplate):
    """
    This is an implementation of a Koopman autoencoder as a Reduced Order Model.

    A Koopman autoencoder architecture consists of five stages:

    * The convolutional encoder [Optional]
    * Fully-connected encoder
    * Koopman operator
    * Fully connected decoder
    * The convolutional decoder [Optional]

    SCHEME:
                                    (Koopman OPERATOR)
                                             ^
                                      |      |      |
                                      |  |   |   |  |
    Z -> [Conv] -> [Conv] -> ... [Conv] -> |  | | - | |  | -> [Conv.T] -> [Conv.T] -> ... [Conv.T] -> Z_til
                                      |  |       |  |
                                      |             |

                    ENCODER          DENSE BOTTLENECK        DECODER
    """

    def __init__(
        self,
        encoder: Union[ConvolutionalNetwork, DenseNetwork] = None,
        bottleneck_encoder: Optional[Union[Linear, DenseNetwork]] = None,
        bottleneck_decoder: Optional[Union[Linear, DenseNetwork]] = None,
        decoder: Union[ConvolutionalNetwork, DenseNetwork] = None,
        input_dim: Optional[Tuple[int, ...]] = None,
        output_dim: Optional[Tuple[int, ...]] = None,
        latent_dim: Optional[int] = None,
        activation: Optional[Union[list, str]] = None,
        channels: Optional[int] = None,
        case: Optional[str] = None,
        architecture: Optional[str] = None,
        shallow: Optional[bool] = False,
        use_batch_norm: Optional[bool] = False,
        encoder_activation: str = "relu",
        devices: Union[str, list] = "cpu",
        name: str = None,
    ) -> None:
        """
        Constructs a new instance of the Autoencoder

        Parameters
        ----------
        encoder : Union[ConvolutionalNetwork, DenseNetwork], optional
            The encoder network. Defaults to None.
        bottleneck_encoder : Optional[Union[Linear, DenseNetwork]], optional
            The bottleneck encoder network. Defaults to None.
        bottleneck_decoder : Optional[Union[Linear, DenseNetwork]], optional
            The bottleneck decoder network. Defaults to None.
        decoder : Union[ConvolutionalNetwork, DenseNetwork], optional
            The decoder network. Defaults to None.
        input_dim : Optional[Tuple[int, ...]], optional
            The input dimensions. Used for automatic network generation. Defaults to None.
        output_dim : Optional[Tuple[int, ...]], optional
            The output dimensions. Used for automatic network generation. Defaults to None.
        latent_dim : Optional[int], optional
            The latent dimensions. Used for automatic network generation. Defaults to None.
        activation : Optional[Union[list, str]], optional
            The activation functions for each layer. Used for automatic network generation. Defaults to None.
        channels : Optional[int], optional
            The number of channels. Used for automatic network generation. Defaults to None.
        case : Optional[str], optional
            The type of problem. Used for automatic network generation. Defaults to None.
        architecture : Optional[str], optional
            The network architecture. Used for automatic network generation. Defaults to None.
        shallow : Optional[bool], optional
            Whether to use shallow or deep network. Used for automatic network generation. Defaults to False.
        encoder_activation : str, optional
            The activation function for the encoder. Defaults to "relu".
        devices : Union[str, list], optional
            The devices to use. Defaults to "cpu".
        name : str, optional
            The name of the autoencoder. Defaults to None.
        """
        super(AutoencoderKoopman, self).__init__(name=name)

        self.weights = list()

        # Determining the kind of device to be used for allocating the
        # subnetworks
        self.device = self._set_device(devices=devices)

        self.input_dim = None

        # If not network is provided, the automatic generation
        # pipeline is activated.
        if all(
            [
                isn == None
                for isn in [encoder, decoder, bottleneck_encoder, bottleneck_decoder]
            ]
        ):
            self.input_dim = input_dim

            encoder, decoder, bottleneck_encoder, bottleneck_decoder = autoencoder_auto(
                input_dim=input_dim,
                latent_dim=latent_dim,
                output_dim=output_dim,
                activation=activation,
                channels=channels,
                architecture=architecture,
                case=case,
                shallow=shallow,
                use_batch_norm=use_batch_norm,
            )

        self.encoder = encoder.to(self.device)
        self.decoder = decoder.to(self.device)

        self.add_module("encoder", self.encoder)
        self.add_module("decoder", self.decoder)

        self.weights += self.encoder.weights
        self.weights += self.decoder.weights

        # These subnetworks are optional
        if bottleneck_encoder is not None and bottleneck_decoder is not None:
            self.bottleneck_encoder = self.to_wrap(entity=bottleneck_encoder, device=self.device)
            self.bottleneck_decoder = self.to_wrap(entity=bottleneck_decoder, device=self.device)

            self.add_module("bottleneck_encoder", self.bottleneck_encoder)
            self.add_module("bottleneck_decoder", self.bottleneck_decoder)

            self.weights += self.bottleneck_encoder.weights
            self.weights += self.bottleneck_decoder.weights

        # These subnetworks are optional
        if bottleneck_encoder is not None and bottleneck_decoder is not None:
            self.bottleneck_encoder = self.to_wrap(entity=bottleneck_encoder, device=self.device)
            self.bottleneck_decoder = self.to_wrap(entity=bottleneck_decoder, device=self.device)

            self.add_module("bottleneck_encoder", self.bottleneck_encoder)
            self.add_module("bottleneck_decoder", self.bottleneck_decoder)

            self.weights += self.bottleneck_encoder.weights
            self.weights += self.bottleneck_decoder.weights

        if bottleneck_encoder is not None and bottleneck_decoder is not None:
            self.projection = self._projection_with_bottleneck
            self.reconstruction = self._reconstruction_with_bottleneck
        else:
            self.projection = self._projection
            self.reconstruction = self._reconstruction

        self.last_encoder_channels = None
        self.before_flatten_dimension = None

        self.latent_dimension = None

        if bottleneck_encoder is not None:
            self.latent_dimension = bottleneck_encoder.output_size
        else:
            self.latent_dimension = self.encoder.output_size

        self.K_op = self.to_wrap(entity=torch.nn.Linear(
            self.latent_dimension, self.latent_dimension, bias=False
        ).weight, device=self.device)

        self.encoder_activation = self._get_operation(operation=encoder_activation)

        self.shapes_dict = dict()

    def summary(
        self,
        input_data: Union[np.ndarray, torch.Tensor] = None,
        input_shape: list = None,
        verbose: bool = True,
    ) -> torch.Tensor:
        if verbose == True:
            if self.input_dim != None:
                input_shape = list(self.input_dim)
            else:
                pass

            self.encoder.summary(
                input_data=input_data, input_shape=input_shape, device=self.device
            )

            self.before_flatten_dimension = tuple(self.encoder.output_size[1:])

            if isinstance(input_data, np.ndarray):
                btnk_input = self.encoder.forward(input_data=input_data)
            else:
                assert (
                    input_shape
                ), "It is necessary to have input_shape when input_data is None."
                input_shape = self.encoder.input_size
                input_shape[0] = 1

                input_data = self.to_wrap(entity=torch.ones(input_shape), device=self.device)

                btnk_input = self.encoder.forward(input_data=input_data)

            before_flatten_dimension = tuple(btnk_input.shape[1:])
            btnk_input = btnk_input.reshape((-1, np.prod(btnk_input.shape[1:])))

            latent = self.bottleneck_encoder.forward(input_data=btnk_input)

            self.bottleneck_encoder.summary()

            print(f"The Koopman Operator has shape: {self.K_op.shape} ")

            self.bottleneck_decoder.summary()

            bottleneck_output = self.encoder_activation(
                self.bottleneck_decoder.forward(input_data=latent)
            )

            bottleneck_output = bottleneck_output.reshape(
                (-1, *before_flatten_dimension)
            )

            self.decoder.summary(input_data=bottleneck_output, device=self.device)

            # Saving the content of the subnetworks to the overall architecture dictionary
            self.shapes_dict.update({"encoder": self.encoder.shapes_dict})
            self.shapes_dict.update(
                {"bottleneck_encoder": self.bottleneck_encoder.shapes_dict}
            )
            self.shapes_dict.update(
                {"bottleneck_decoder": self.bottleneck_decoder.shapes_dict}
            )
            self.shapes_dict.update({"decoder": self.decoder.shapes_dict})

        else:
            print(self)

    @as_tensor
    def _projection_with_bottleneck(
        self, input_data: Union[np.ndarray, torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Computes the projection of the input data onto the bottleneck encoder.

        Parameters
        ----------
        input_data : Union[np.ndarray, torch.Tensor], optional
            The input data. Defaults to None.

        Returns
        -------
        torch.Tensor
            The projected latent representation.

        """
        btnk_input = self.encoder.forward(input_data=input_data)

        self.before_flatten_dimension = tuple(btnk_input.shape[1:])

        btnk_input = btnk_input.reshape((-1, np.prod(self.before_flatten_dimension)))

        latent = self.bottleneck_encoder.forward(input_data=btnk_input)

        return latent

    @as_tensor
    def _projection(
        self, input_data: Union[np.ndarray, torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Computes the projection of the input data onto the encoder.

        Parameters
        ----------
        input_data : Union[np.ndarray, torch.Tensor], optional
            The input data. Defaults to None.

        Returns
        -------
        torch.Tensor
            The projected latent representation.

        """
        latent = self.encoder.forward(input_data=input_data)

        return latent

    @as_tensor
    def _reconstruction_with_bottleneck(
        self, input_data: Union[torch.Tensor, np.ndarray] = None
    ) -> torch.Tensor:
        """
        Reconstructs the input data using the bottleneck decoder.

        Parameters
        ----------
        input_data : Union[torch.Tensor, np.ndarray], optional
            The input data. Defaults to None.

        Returns
        -------
        torch.Tensor
            The reconstructed data.

        """
        bottleneck_output = self.encoder_activation(
            self.bottleneck_decoder.forward(input_data=input_data)
        )

        bottleneck_output = bottleneck_output.reshape(
            (-1,) + self.before_flatten_dimension
        )

        reconstructed = self.decoder.forward(input_data=bottleneck_output)

        return reconstructed

    @as_tensor
    def _reconstruction(
        self, input_data: Union[torch.Tensor, np.ndarray] = None
    ) -> torch.Tensor:
        """
        Reconstructs the input data using the decoder.

        Parameters
        ----------
        input_data : Union[torch.Tensor, np.ndarray], optional
            The input data. Defaults to None.

        Returns
        -------
        torch.Tensor
            The reconstructed data.

        """
        reconstructed = self.decoder.forward(input_data=input_data)

        return reconstructed

    def latent_forward_m(
        self, input_data: Union[np.ndarray, torch.Tensor] = None, m: int = 1
    ) -> torch.Tensor:
        """
        Evaluates the operation u^{u+m} = K^m u^{i}

        Parameters
        ----------
        input_data : Union[np.ndarray, torch.Tensor], optional
            The input data. Defaults to None.
        m : int, optional
            The number of Koopman iterations. Defaults to 1.

        Returns
        -------
        torch.Tensor
            The computed latent representation.

        """
        return torch.matmul(input_data, torch.pow(self.K_op.T, m))

    def latent_forward(
        self, input_data: Union[np.ndarray, torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Evaluates the operation u^{u+1} = K u^{i}

        Parameters
        ----------
        input_data : Union[np.ndarray, torch.Tensor], optional
            The input data. Defaults to None.

        Returns
        -------
        torch.Tensor
            The computed latent representation.

        """
        return torch.matmul(input_data, self.K_op.T)

    def reconstruction_forward(
        self, input_data: Union[np.ndarray, torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Evaluates the operation Ũ = D(E(U))

        Parameters
        ----------
        input_data : Union[np.ndarray, torch.Tensor], optional
            The input data. Defaults to None.

        Returns
        -------
        torch.Tensor
            The reconstructed data.

        """
        latent = self.projection(input_data=input_data)
        reconstructed = self.reconstruction(input_data=latent)

        return reconstructed

    def reconstruction_forward_m(
        self, input_data: Union[np.ndarray, torch.Tensor] = None, m: int = 1
    ) -> torch.Tensor:
        """
        Evaluates the operation Ũ_m = D(K^m E(U))

        Parameters
        ----------
        input_data : Union[np.ndarray, torch.Tensor], optional
            The input data. Defaults to None.
        m : int, optional
            The number of Koopman iterations. Defaults to 1.

        Returns
        -------
        torch.Tensor
            The reconstructed data.

        """
        latent = self.projection(input_data=input_data)
        latent_m = self.latent_forward_m(input_data=latent, m=m)
        reconstructed_m = self.reconstruction(input_data=latent_m)

        return reconstructed_m

    def predict(
        self, input_data: Union[np.ndarray, torch.Tensor] = None, n_steps: int = 1
    ) -> np.ndarray:
        """
        Predicts the reconstructed data for the input data after n_steps extrapolation in the latent space.

        Parameters
        ----------
        input_data : Union[np.ndarray, torch.Tensor], optional
            The input data. Defaults to None.
        n_steps : int, optional
            The number of extrapolations to perform. Defaults to 1.

        Returns
        -------
        np.ndarray
            The predicted reconstructed data.

        """
        if isinstance(input_data, np.ndarray):
            input_data = torch.from_numpy(input_data.astype(ARRAY_DTYPE))

        predictions = list()
        latent = self.projection(input_data=input_data)
        init_latent = latent

        # Extrapolating in the latent space over n_steps steps
        for s in range(n_steps):
            latent_s = self.latent_forward(input_data=init_latent)
            init_latent = latent_s
            predictions.append(latent_s)

        predictions = torch.vstack(predictions)

        reconstructed_predictions = self.reconstruction(input_data=predictions)

        return reconstructed_predictions.detach().numpy()

    def project(self, input_data: Union[np.ndarray, torch.Tensor] = None) -> np.ndarray:
        """
        Projects the input data into the latent space.

        Parameters
        ----------
        input_data : Union[np.ndarray, torch.Tensor], optional
            The input data. Defaults to None.

        Returns
        -------
        np.ndarray
            The projected data.

        """
        projected_data = self.projection(input_data=input_data)

        return projected_data.cpu().detach().numpy()

    def reconstruct(
        self, input_data: Union[np.ndarray, torch.Tensor] = None
    ) -> np.ndarray:
        """
        Reconstructs the input data.

        Parameters
        ----------
        input_data : Union[np.ndarray, torch.Tensor], optional
            The input data. Defaults to None.

        Returns
        -------
        np.ndarray
            The reconstructed data.

        """
        reconstructed_data = self.reconstruction(input_data=input_data)

        return reconstructed_data.cpu().detach().numpy()


class AutoencoderVariational(NetworkTemplate):
    r"""
    This is an implementation of a Koopman autoencoder as a reduced order model.

    A variational autoencoder architecture consists of five stages:
    --> The convolutional encoder [Optional]
    --> Fully-connected encoder
    --> Gaussian noise
    --> Fully connected decoder
    --> The convolutional decoder [Optional]

    SCHEME:
                                                  Gaussian noise
                                                  ^
                                           |      |      |
                                           |  |   |   |  |
    Z -> [Conv] -> [Conv] -> ... [Conv] -> |  | | - | |  | -> [Conv.T] -> [Conv.T] -> ... [Conv.T] -> Z_til
                                           |  |       |  |
                                           |             |

                   ENCODER               DENSE BOTTLENECK           DECODER
    """

    def __init__(
        self,
        encoder: Union[ConvolutionalNetwork, DenseNetwork] = None,
        bottleneck_encoder: Optional[Union[Linear, DenseNetwork]] = None,
        bottleneck_decoder: Optional[Union[Linear, DenseNetwork]] = None,
        decoder: Union[ConvolutionalNetwork, DenseNetwork] = None,
        encoder_activation: str = "relu",
        input_dim: Optional[Tuple[int, ...]] = None,
        output_dim: Optional[Tuple[int, ...]] = None,
        latent_dim: Optional[int] = None,
        activation: Optional[Union[list, str]] = None,
        channels: Optional[int] = None,
        kernel_size: Optional[int] = None,
        case: Optional[str] = None,
        architecture: Optional[str] = None,
        use_batch_norm: Optional[bool] = False,
        shallow: Optional[bool] = False,
        scale: float = 1e-3,
        devices: Union[str, list] = "cpu",
        name: str = None,
        **kwargs,
    ) -> None:
        """
        Constructor method.

        Parameters
        ----------
        encoder : Union[ConvolutionalNetwork, DenseNetwork], optional
            The encoder network. Defaults to None.
        bottleneck_encoder : Optional[Union[Linear, DenseNetwork]], optional
            The bottleneck encoder network. Defaults to None.
        bottleneck_decoder : Optional[Union[Linear, DenseNetwork]], optional
            The bottleneck decoder network. Defaults to None.
        decoder : Union[ConvolutionalNetwork, DenseNetwork], optional
            The decoder network. Defaults to None.
        encoder_activation : str, optional
            The activation function to use in the encoder. Defaults to "relu".
        input_dim : Optional[Tuple[int, ...]], optional
            The input dimension of the data. Defaults to None.
        output_dim : Optional[Tuple[int, ...]], optional
            The output dimension of the data. Defaults to None.
        latent_dim : Optional[int], optional
            The size of the bottleneck layer. Defaults to None.
        activation : Optional[Union[list, str]], optional
            The activation function to use in the networks. Defaults to None.
        channels : Optional[int], optional
            The number of channels in the input data. Defaults to None.
        kernel_size : Optional[int]
            Convolutional kernel size.
        case : Optional[str], optional
            The name of the autoencoder variant. Defaults to None.
        architecture : Optional[str], optional
            The architecture of the networks. Defaults to None.
        shallow : Optional[bool], optional
            Whether to use a shallow network architecture. Defaults to False.
        scale : float, optional
            The scale of the initialization. Defaults to 1e-3.
        devices : Union[str, list], optional
            The device(s) to use for computation. Defaults to "cpu".
        name : str, optional
            The name of the autoencoder. Defaults to None.

        """
        super(AutoencoderVariational, self).__init__(name=name)

        self.weights = list()

        # Determining the kind of device to be used for allocating the
        # subnetworks
        self.device = self._set_device(devices=devices)

        self.input_dim = None

        # If not network is provided, the automatic generation
        # pipeline is activated.
        if all(
            [
                isn == None
                for isn in [encoder, decoder, bottleneck_encoder, bottleneck_decoder]
            ]
        ):
            self.input_dim = input_dim

            encoder, decoder, bottleneck_encoder, bottleneck_decoder = autoencoder_auto(
                input_dim=input_dim,
                latent_dim=latent_dim,
                output_dim=output_dim,
                activation=activation,
                channels=channels,
                kernel_size=kernel_size,
                architecture=architecture,
                case=case,
                shallow=shallow,
                use_batch_norm=use_batch_norm,
                name=self.name,
                **kwargs
            )

        self.encoder = self.to_wrap(entity=encoder, device=self.device)
        self.decoder = decoder.to(self.device)

        self.add_module("encoder", self.encoder)
        self.add_module("decoder", self.decoder)

        self.weights += self.encoder.weights
        self.weights += self.decoder.weights

        self.there_is_bottleneck = False

        # These subnetworks are optional
        if bottleneck_encoder is not None and bottleneck_decoder is not None:
            self.bottleneck_encoder = self.to_wrap(entity=bottleneck_encoder, device=self.device)
            self.bottleneck_decoder = self.to_wrap(entity=bottleneck_decoder, device=self.device)

            self.add_module("bottleneck_encoder", self.bottleneck_encoder)
            self.add_module("bottleneck_decoder", self.bottleneck_decoder)

            self.weights += self.bottleneck_encoder.weights
            self.weights += self.bottleneck_decoder.weights

            self.projection = self._projection_with_bottleneck
            self.reconstruction = self._reconstruction_with_bottleneck

            self.there_is_bottleneck = True

        else:
            self.projection = self._projection
            self.reconstruction = self._reconstruction

        self.last_encoder_channels = None
        self.before_flatten_dimension = None

        self.latent_dimension = None

        if bottleneck_encoder is not None:
            self.latent_dimension = bottleneck_encoder.output_size
        else:
            self.latent_dimension = self.encoder.output_size

        self.z_mean = self.to_wrap(entity=torch.nn.Linear(self.latent_dimension,
                                                          self.latent_dimension),
            device=self.device
        )

        self.z_log_var = self.to_wrap(entity=torch.nn.Linear(self.latent_dimension,
                                                            self.latent_dimension),
            device=self.device
        )

        self.add_module("z_mean", self.z_mean)
        self.add_module("z_log_var", self.z_log_var)

        self.weights += [self.z_mean.weight]
        self.weights += [self.z_log_var.weight]

        self.mu = None
        self.log_v = None
        self.scale = scale

        self.encoder_activation = self._get_operation(operation=encoder_activation)

        self.shapes_dict = dict()

    def summary(
        self,
        input_data: Union[np.ndarray, torch.Tensor] = None,
        input_shape: list = None,
        verbose: bool = True,
        display: bool = True,
    ) -> torch.Tensor:
        """
        Summarizes the overall architecture of the autoencoder and saves the content of the subnetworks to a dictionary.

        Parameters
        ----------
        input_data : Union[np.ndarray, torch.Tensor], optional
            Input data to pass through the encoder, by default None
        input_shape : list, optional
            The shape of the input data if input_data is None, by default None

        Returns
        -------
        torch.Tensor
            The output of the autoencoder's decoder applied to the input data.

        Raises
        ------
        Exception
            If self.input_dim is not a tuple or an integer.

        AssertionError
            If input_shape is None when input_data is None.

        Notes
        -----
        The summary method calls the `summary` method of each of the subnetworks and saves the content of the subnetworks to the overall architecture dictionary. If there is a bottleneck network, it is also summarized and saved to the architecture dictionary.

        Examples
        --------
        >>> autoencoder = AutoencoderVariational(input_dim=(28, 28, 1))
        >>> input_data = np.random.rand(1, 28, 28, 1)
        >>> output_data = autoencoder.summary(input_data=input_data)
        """

        if verbose == True:
            if self.input_dim != None:
                if type(self.input_dim) == tuple:
                    input_shape = list(self.input_dim)
                elif type(self.input_dim) == int:
                    input_shape = [None, self.input_dim]
                else:
                    raise Exception(
                        f"input_dim is expected to be tuple or int, but received {type(self.input_dim)}"
                    )
            else:
                pass

            self.encoder.summary(
                input_data=input_data, input_shape=input_shape, device=self.device, display=display
            )

            if type(self.encoder.output_size) == tuple:
                self.before_flatten_dimension = tuple(self.encoder.output_size[1:])
                input_shape = self.encoder.input_size
            elif type(self.encoder.output_size) == int:
                input_shape = [None, self.encoder.input_size]
            else:
                pass

            if isinstance(input_data, np.ndarray):
                btnk_input = self.encoder.forward(input_data=input_data)
            else:
                assert (
                    input_shape
                ), "It is necessary to have input_shape when input_data is None."

                input_shape[0] = 1

                input_data = self.to_wrap(entity=torch.ones(input_shape), device=self.device)

                btnk_input = self.encoder.forward(input_data=input_data)

            before_flatten_dimension = tuple(btnk_input.shape[1:])
            btnk_input = btnk_input.reshape((-1, np.prod(btnk_input.shape[1:])))

            # Bottleneck networks is are optional
            if self.there_is_bottleneck:
                latent = self.bottleneck_encoder.forward(input_data=btnk_input)

                self.bottleneck_encoder.summary(display=display)
                self.bottleneck_decoder.summary(display=display)

                bottleneck_output = self.encoder_activation(
                    self.bottleneck_decoder.forward(input_data=latent)
                )

                bottleneck_output = bottleneck_output.reshape(
                    (-1, *before_flatten_dimension)
                )
            else:
                bottleneck_output = btnk_input

            self.decoder.summary(input_data=bottleneck_output, device=self.device, display=display)

            # Saving the content of the subnetworks to the overall architecture dictionary
            self.shapes_dict.update({"encoder": self.encoder.shapes_dict})

            # Bottleneck networks is are optional
            if self.there_is_bottleneck:
                self.shapes_dict.update(
                    {"bottleneck_encoder": self.bottleneck_encoder.shapes_dict}
                )
                self.shapes_dict.update(
                    {"bottleneck_decoder": self.bottleneck_decoder.shapes_dict}
                )

            self.shapes_dict.update({"decoder": self.decoder.shapes_dict})

        else:
            print(self)

    @as_tensor
    def _projection_with_bottleneck(
        self, input_data: Union[np.ndarray, torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Applies the encoder and bottleneck encoder to input data and returns the output.

        Parameters
        ----------
        input_data : Union[np.ndarray, torch.Tensor], optional
            The input data to pass through the encoder, by default None

        Returns
        -------
        torch.Tensor
            The output of the bottleneck encoder applied to the input data.

        Notes
        -----
        This function is used for projection of the input data into the bottleneck space.

        Examples
        --------
        >>> autoencoder = AutoencoderVariational(input_dim=(28, 28, 1))
        >>> input_data = np.random.rand(1, 28, 28, 1)
        >>> output_data = autoencoder._projection_with_bottleneck(input_data=input_data)
        """
        btnk_input = self.encoder.forward(input_data=input_data)

        self.before_flatten_dimension = tuple(self.encoder.output_size[1:])

        btnk_input = btnk_input.reshape((-1, np.prod(self.before_flatten_dimension)))

        latent = self.bottleneck_encoder.forward(input_data=btnk_input)

        return latent

    @as_tensor
    def _projection(
        self, input_data: Union[np.ndarray, torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Applies the encoder to input data and returns the output.

        Parameters
        ----------
        input_data : Union[np.ndarray, torch.Tensor], optional
            The input data to pass through the encoder, by default None

        Returns
        -------
        torch.Tensor
            The output of the encoder applied to the input data.

        Examples
        --------
        >>> autoencoder = AutoencoderVariational(input_dim=(28, 28, 1))
        >>> input_data = np.random.rand(1, 28, 28, 1)
        >>> output_data = autoencoder._projection(input_data=input_data)
        """
        latent = self.encoder.forward(input_data=input_data)

        return latent

    @as_tensor
    def _reconstruction_with_bottleneck(
        self, input_data: Union[torch.Tensor, np.ndarray] = None
    ) -> torch.Tensor:
        """
        Applies the bottleneck decoder and decoder to input data and returns the output.

        Parameters
        ----------
        input_data : Union[np.ndarray, torch.Tensor], optional
            The input data to pass through the bottleneck decoder and decoder, by default None

        Returns
        -------
        torch.Tensor
            The output of the decoder applied to the bottleneck decoder's output.

        Notes
        -----
        This function is used for reconstruction of the input data from the bottleneck space.

        Examples
        --------
        >>> autoencoder = AutoencoderVariational(input_dim=(28, 28, 1))
        >>> input_data = np.random.rand(1, 28, 28, 1)
        >>> bottleneck_output = autoencoder._projection_with_bottleneck(input_data=input_data)
        >>> output_data = autoencoder._reconstruction_with_bottleneck(input_data=bottleneck_output)
        """
        bottleneck_output = self.encoder_activation(
            (self.bottleneck_decoder.forward(input_data=input_data))
        )

        bottleneck_output = bottleneck_output.reshape(
            (-1,) + self.before_flatten_dimension
        )

        reconstructed = self.decoder.forward(input_data=bottleneck_output)

        return reconstructed

    @as_tensor
    def _reconstruction(
        self, input_data: Union[torch.Tensor, np.ndarray] = None
    ) -> torch.Tensor:
        """
        Applies the decoder to input data and returns the output.

        Parameters
        ----------
        input_data : Union[np.ndarray, torch.Tensor], optional
            The input data to pass through the decoder, by default None

        Returns
        -------
        torch.Tensor
            The output of the decoder applied to the input data.

        Examples
        --------
        >>> autoencoder = AutoencoderVariational(input_dim=(28, 28, 1))
        >>> input_data = np.random.rand(1, 28, 28, 1)
        >>> output_data = autoencoder._reconstruction(input_data=input_data)
        """
        reconstructed = self.decoder.forward(input_data=input_data)

        return reconstructed

    def Mu(
        self, input_data: Union[np.ndarray, torch.Tensor] = None, to_numpy: bool = False
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Computes the mean of the encoded input data.

        Parameters
        ----------
        input_data : Union[np.ndarray, torch.Tensor], optional
            The input data to encode and compute the mean, by default None
        to_numpy : bool, optional
            If True, returns the result as a NumPy array, by default False

        Returns
        -------
        Union[np.ndarray, torch.Tensor]
            The mean of the encoded input data.

        Examples
        --------
        >>> autoencoder = AutoencoderVariational(input_dim=(28, 28, 1))
        >>> input_data = np.random.rand(1, 28, 28, 1)
        >>> mu = autoencoder.Mu(input_data=input_data)
        """
        latent = self.projection(input_data=input_data)

        if to_numpy == True:
            return self.z_mean(latent).detach().numpy()
        else:
            return self.z_mean(latent)

    def Sigma(
        self, input_data: Union[np.ndarray, torch.Tensor] = None, to_numpy: bool = False
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Computes the standard deviation of the encoded input data.

        Parameters
        ----------
        input_data : Union[np.ndarray, torch.Tensor], optional
            The input data to encode and compute the standard deviation, by default None
        to_numpy : bool, optional
            If True, returns the result as a NumPy array, by default False

        Returns
        -------
        Union[np.ndarray, torch.Tensor]
            The standard deviation of the encoded input data.

        Examples
        --------
        >>> autoencoder = AutoencoderVariational(input_dim=(28, 28, 1))
        >>> input_data = np.random.rand(1, 28, 28, 1)
        >>> sigma = autoencoder.Sigma(input_data=input_data)
        """
        latent = self.projection(input_data=input_data)

        if to_numpy == True:
            return torch.exp(self.z_log_var(latent) / 2).detach().numpy()
        else:
            return torch.exp(self.z_log_var(latent) / 2)

    def CoVariance(
        self,
        input_data: Union[np.ndarray, torch.Tensor] = None,
        inv: bool = False,
        to_numpy: bool = False,
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Computes the covariance matrix of the encoded input data.

        Parameters
        ----------
        input_data : Union[np.ndarray, torch.Tensor], optional
            The input data to encode and compute the covariance matrix, by default None
        inv : bool, optional
            If True, returns the inverse of the covariance matrix, by default False
        to_numpy : bool, optional
            If True, returns the result as a NumPy array, by default False

        Returns
        -------
        Union[np.ndarray, torch.Tensor]
            The covariance matrix (or its inverse) of the encoded input data.

        Examples
        --------
        >>> autoencoder = AutoencoderVariational(input_dim=(28, 28, 1))
        >>> input_data = np.random.rand(1, 28, 28, 1)
        >>> covariance = autoencoder.CoVariance(input_data=input_data)
        """
        if inv == False:
            Sigma_inv = 1 / self.Sigma(input_data=input_data)
            covariance = torch.diag_embed(Sigma_inv)

        else:
            Sigma = self.Sigma(input_data=input_data)
            covariance = torch.diag_embed(Sigma)

        if to_numpy == True:
            return covariance.detach().numpy()
        else:
            return covariance

    def latent_gaussian_noisy(
        self, input_data: Union[np.ndarray, torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Generates a noisy latent representation of the input data.

        Parameters
        ----------
        input_data : Union[np.ndarray, torch.Tensor], optional
            The input data to encode and generate a noisy latent representation, by default None

        Returns
        -------
        torch.Tensor
            A noisy latent representation of the input data.

        Notes
        -----
        This function adds Gaussian noise to the mean and standard deviation of the encoded input data to generate a noisy latent representation.

        Examples
        --------
        >>> autoencoder = AutoencoderVariational(input_dim=(28, 28, 1))
        >>> input_data = np.random.rand(1, 28, 28, 1)
        >>> noisy_latent = autoencoder.latent_gaussian_noisy(input_data=input_data)
        """
        self.mu = self.z_mean(input_data)
        self.log_v = self.z_log_var(input_data)
        eps = self.scale * torch.autograd.Variable(
            torch.randn(*self.log_v.size())
        ).type_as(self.log_v)

        return self.mu + torch.exp(self.log_v / 2.0) * eps

    def reconstruction_forward(
        self, input_data: Union[np.ndarray, torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Applies the encoder, adds Gaussian noise to the encoded data, and then applies the decoder to generate a reconstructed output.

        Parameters
        ----------
        input_data : Union[np.ndarray, torch.Tensor], optional
            The input data to pass through the autoencoder, by default None

        Returns
        -------
        torch.Tensor
            The reconstructed output of the autoencoder.

        Examples
        --------
        >>> autoencoder = AutoencoderVariational(input_dim=(28, 28, 1))
        >>> input_data = np.random.rand(1, 28, 28, 1)
        >>> reconstructed_data = autoencoder.reconstruction_forward(input_data=input_data)
        """
        latent = self.projection(input_data=input_data)
        latent_noisy = self.latent_gaussian_noisy(input_data=latent)
        reconstructed = self.reconstruction(input_data=latent_noisy)

        return reconstructed

    def reconstruction_eval(
        self, input_data: Union[np.ndarray, torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Applies the encoder, computes the mean of the encoded data, and then applies the decoder to generate a reconstructed output.

        Parameters
        ----------
        input_data : Union[np.ndarray, torch.Tensor], optional
            The input data to pass through the autoencoder, by default None

        Returns
        -------
        torch.Tensor
            The reconstructed output of the autoencoder.

        Examples
        --------
        >>> autoencoder = AutoencoderVariational(input_dim=(28, 28, 1))
        >>> input_data = np.random.rand(1, 28, 28, 1)
        >>> reconstructed_data = autoencoder.reconstruction_eval(input_data=input_data)
        """
        encoder_output = self.projection(input_data=input_data)
        latent = self.z_mean(encoder_output)
        reconstructed = self.reconstruction(input_data=latent)

        return reconstructed

    def project(self, input_data: Union[np.ndarray, torch.Tensor] = None) -> np.ndarray:
        """
        Projects the input data onto the autoencoder's latent space.

        Parameters
        ----------
        input_data : Union[np.ndarray, torch.Tensor], optional
            The input data to project onto the autoencoder's latent space, by default None

        Returns
        -------
        np.ndarray
            The input data projected onto the autoencoder's latent space.

        Examples
        --------
        >>> autoencoder = AutoencoderVariational(input_dim=(28, 28, 1))
        >>> input_data = np.random.rand(1, 28, 28, 1)
        >>> projected_data = autoencoder.project(input_data=input_data)
        """
        if isinstance(input_data, np.ndarray):
            input_data = torch.from_numpy(input_data.astype(ARRAY_DTYPE))

        input_data = input_data.to(self.device)

        projected_data_latent = self.Mu(input_data=input_data)

        return projected_data_latent.cpu().detach().numpy()

    def reconstruct(
        self, input_data: Union[np.ndarray, torch.Tensor] = None
    ) -> np.ndarray:
        """
        Reconstructs the input data using the trained autoencoder.

        Parameters
        ----------
        input_data : Union[np.ndarray, torch.Tensor], optional
            The input data to reconstruct, by default None

        Returns
        -------
        np.ndarray
            The reconstructed data.

        Examples
        --------
        >>> autoencoder = Autoencoder(input_dim=(28, 28, 1))
        >>> input_data = np.random.rand(1, 28, 28, 1)
        >>> reconstructed_data = autoencoder.reconstruct(input_data=input_data)
        """
        if isinstance(input_data, np.ndarray):
            input_data = torch.from_numpy(input_data.astype(ARRAY_DTYPE))

        input_data = input_data.to(self.device)

        reconstructed_data = self.reconstruction(input_data=input_data)

        return reconstructed_data.cpu().detach().numpy()

    def eval(self, input_data: Union[np.ndarray, torch.Tensor] = None) -> np.ndarray:
        """
        Reconstructs the input data using the mean of the encoded data.

        Parameters
        ----------
        input_data : Union[np.ndarray, torch.Tensor], optional
            The input data to reconstruct, by default None

        Returns
        -------
        np.ndarray
            The reconstructed data.

        Examples
        --------
        >>> autoencoder = Autoencoder(input_dim=(28, 28, 1))
        >>> input_data = np.random.rand(1, 28, 28, 1)
        >>> reconstructed_data = autoencoder.eval(input_data=input_data)
        """
        if isinstance(input_data, np.ndarray):
            input_data = torch.from_numpy(input_data.astype(ARRAY_DTYPE))

        input_data = input_data.to(self.device)

        return self.reconstruction_eval(input_data=input_data).cpu().detach().numpy()

### Hybrid Autoencoder architectures

class MultiScaleAutoencoder(NetworkTemplate):

    def __init__(self,
                 input_dim: Tuple[int, ...] = None, 
                 output_dim: Optional[Tuple[int, ...]] = None,
                 latent_dim: int=None,
                 kernel_sizes_list: List[int] = None, 
                 activation: str = None,
                 case: str = "2d",
                 shallow:bool=True,
                 scale: float = 1e-3,
                 devices:Union[str, List]="cpu",
                 kind_of_ae: str = "variational",
                 name:str=None,
                 **kwargs) -> None:

        super(MultiScaleAutoencoder, self).__init__(name=name)

        self.architecture = "cnn"
        self.kernel_size_list = kernel_sizes_list
        self.latent_dimension = latent_dim
        self.scale = scale
        self.kind_of_ae = kind_of_ae
        self.device = self._set_device(devices=devices)

        if self.kind_of_ae == "variational":
            self.ae_class = AutoencoderVariational
        else:
            self.ae_class = AutoencoderCNN

        self.AutoencodersList = torch.nn.ModuleList()

        for kernel_size in kernel_sizes_list:

            autoencoder = self.ae_class(
                        input_dim=input_dim,
                        output_dim=output_dim,
                        latent_dim=latent_dim,
                        kernel_size=kernel_size,
                        activation=activation,
                        architecture=self.architecture,
                        case=case,
                        devices=devices,
                        shallow=shallow,
                        name=name,
                        **kwargs,
                    )

            self.AutoencodersList.append(autoencoder)

        ### These methods are used just when the architecture chosen is
        ### variational.
        self.z_mean = self.to_wrap(entity=torch.nn.Linear(self.latent_dimension,
                                                          self.latent_dimension),
            device=self.device
        )

        self.z_log_var = self.to_wrap(entity=torch.nn.Linear(self.latent_dimension,
                                                             self.latent_dimension),
            device=self.device
        )

        self.add_module("z_mean", self.z_mean)
        self.add_module("z_log_var", self.z_log_var)
        ###

        if self.kind_of_ae == "variational":
            self.latent_op = self.latent_gaussian_noisy
            self.latent_op_eval = self.z_mean
        else:
            self.latent_op = self.latent_bypass
            self.latent_op_eval = self.latent_bypass

        self.weights = sum([ae.weights for ae in self.AutoencodersList], [])

        self.forward = self.reconstruction_forward

    def Mu(
        self, input_data: Union[np.ndarray, torch.Tensor] = None, to_numpy: bool = False
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Computes the mean of the encoded input data.

        Parameters
        ----------
        input_data : Union[np.ndarray, torch.Tensor], optional
            The input data to encode and compute the mean, by default None
        to_numpy : bool, optional
            If True, returns the result as a NumPy array, by default False

        Returns
        -------
        Union[np.ndarray, torch.Tensor]
            The mean of the encoded input data.

        """

        latent_list = list()
        for ae in self.AutoencodersList:
            latent = ae.projection(input_data=input_data)
            latent_list.append(latent)

        latent = sum(latent_list)

        if to_numpy == True:
            return self.latent_op_eval(latent).detach().numpy()
        else:
            return self.latent_op_eval(latent)

    def latent_bypass(
        self, input_data: Union[np.ndarray, torch.Tensor] = None
    ) -> torch.Tensor:

        return input_data

    def latent_gaussian_noisy(
        self, input_data: Union[np.ndarray, torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Generates a noisy latent representation of the input data.

        Parameters
        ----------
        input_data : Union[np.ndarray, torch.Tensor], optional
            The input data to encode and generate a noisy latent representation, by default None

        Returns
        -------
        torch.Tensor
            A noisy latent representation of the input data.

        Notes
        -----
        This function adds Gaussian noise to the mean and standard deviation of the encoded input data to generate a noisy latent representation.

        """

        self.mu = self.z_mean(input_data)
        self.log_v = self.z_log_var(input_data)
        eps = self.scale * torch.autograd.Variable(
            torch.randn(*self.log_v.size())
        ).type_as(self.log_v)

        return self.mu + torch.exp(self.log_v / 2.0) * eps

    def reconstruction_forward(
        self, input_data: Union[np.ndarray, torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Applies the encoder, adds Gaussian noise to the encoded data, and then applies the decoder to generate a reconstructed output.

        Parameters
        ----------
        input_data : Union[np.ndarray, torch.Tensor], optional
            The input data to pass through the autoencoder, by default None

        Returns
        -------
        torch.Tensor
            The reconstructed output of the autoencoder.

       """

        latent_list = list()
        for ae in self.AutoencodersList:
            latent = ae.projection(input_data=input_data)
            latent_list.append(latent)

        latent = sum(latent_list)
        latent_mod = self.latent_op(input_data=latent)

        reconstructed_list = list()
        for ae in self.AutoencodersList:
            reconstructed_ = ae.reconstruction(input_data=latent_mod)
            reconstructed_list.append(reconstructed_)

        reconstructed = sum(reconstructed_list)

        return reconstructed

    def reconstruction_eval(
        self, input_data: Union[np.ndarray, torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Applies the encoder, computes the mean of the encoded data, and then applies the decoder to generate a reconstructed output.

        Parameters
        ----------
        input_data : Union[np.ndarray, torch.Tensor], optional
            The input data to pass through the autoencoder, by default None

        Returns
        -------
        torch.Tensor
            The reconstructed output of the autoencoder.

        """

        latent_list = list()
        for ae in self.AutoencodersList:
            latent = ae.projection(input_data=input_data)
            latent_list.append(latent)

        latent_ = sum(latent_list)
        latent_mod = self.latent_op_eval(latent_)

        reconstructed_list = list()
        for ae in self.AutoencodersList:
            reconstructed_ = ae.reconstruction(input_data=latent_mod)
            reconstructed_list.append(reconstructed_)

        reconstructed = sum(reconstructed_list)

        return reconstructed


    def project(self, input_data: Union[np.ndarray, torch.Tensor] = None) -> np.ndarray:
        """
        Projects the input data onto the autoencoder's latent space.

        Parameters
        ----------
        input_data : Union[np.ndarray, torch.Tensor], optional
            The input data to project onto the autoencoder's latent space, by default None

        Returns
        -------
        np.ndarray
            The input data projected onto the autoencoder's latent space.

        """

        if isinstance(input_data, np.ndarray):
            input_data = torch.from_numpy(input_data.astype(ARRAY_DTYPE))

        input_data = input_data.to(self.device)

        projected_data_latent = self.Mu(input_data=input_data)

        return projected_data_latent.cpu().detach().numpy()

    def reconstruct(
        self, input_data: Union[np.ndarray, torch.Tensor] = None
    ) -> np.ndarray:
        """
        Reconstructs the input data using the trained autoencoder.

        Parameters
        ----------
        input_data : Union[np.ndarray, torch.Tensor], optional
            The input data to reconstruct, by default None

        Returns
        -------
        np.ndarray
            The reconstructed data.

        """

        if isinstance(input_data, np.ndarray):
            input_data = torch.from_numpy(input_data.astype(ARRAY_DTYPE))

        input_data = input_data.to(self.device)

        reconstructed_data = sum([ae.reconstruction(input_data=input_data)
                                  for ae in self.AutoencodersList])

        return reconstructed_data.cpu().detach().numpy()

    def eval(self, input_data: Union[np.ndarray, torch.Tensor] = None) -> np.ndarray:
        """
        Reconstructs the input data using the mean of the encoded data.

        Parameters
        ----------
        input_data : Union[np.ndarray, torch.Tensor], optional
            The input data to reconstruct, by default None

        Returns
        -------
        np.ndarray
            The reconstructed data.

        """

        if isinstance(input_data, np.ndarray):
            input_data = torch.from_numpy(input_data.astype(ARRAY_DTYPE))

        input_data = self.to_wrap(entity=input_data, device=self.device)

        return self.reconstruction_eval(input_data=input_data).cpu().detach().numpy()

    def summary(self) -> None:

        print(self)



