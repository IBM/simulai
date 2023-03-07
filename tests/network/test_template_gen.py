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
import torch
from utils import configure_device

DEVICE = configure_device()


def generate_data_2d(
    n_samples: int = None,
    image_size: tuple = None,
    n_inputs: int = None,
    n_outputs: int = None,
) -> (torch.Tensor, torch.Tensor):
    input_data = np.random.rand(n_samples, n_inputs, *image_size)
    output_data = np.random.rand(n_samples, n_outputs)

    return torch.from_numpy(input_data.astype("float32")), torch.from_numpy(
        output_data.astype("float32")
    )


def generate_data_1d(
    n_samples: int = None,
    vector_size: int = None,
    n_inputs: int = None,
    n_outputs: int = None,
) -> (torch.Tensor, torch.Tensor):
    input_data = np.random.rand(n_samples, n_inputs, vector_size)
    output_data = np.random.rand(n_samples, n_outputs)

    return torch.from_numpy(input_data.astype("float32")), torch.from_numpy(
        output_data.astype("float32")
    )


# Model template
def model_2d(
    reduce_dimensionality: bool = True,
    flatten: bool = True,
    channels: int = 2,
    input_dim: tuple = (None, 1, 16, 16),
    output_dim: tuple = (None, 16, 1, 1),
):
    from simulai.templates import NetworkInstanceGen

    # Configuring model

    auto_gen = NetworkInstanceGen(architecture="cnn", dim="2d")

    convnet = auto_gen(
        input_dim=input_dim,
        output_dim=output_dim,
        channels=channels,
        activation="tanh",
        name="conv_2d",
        flatten=flatten,
        reduce_dimensionality=reduce_dimensionality,
    )

    return convnet


def model_1d(
    reduce_dimensionality: bool = True,
    flatten: bool = True,
    channels: int = 2,
    input_dim: tuple = (None, 1, 16),
    output_dim: tuple = (None, 16, 1),
):
    from simulai.templates import NetworkInstanceGen

    # Configuring model

    auto_gen = NetworkInstanceGen(architecture="cnn", dim="1d")

    convnet = auto_gen(
        input_dim=input_dim,
        output_dim=output_dim,
        channels=channels,
        activation="tanh",
        name="conv_1d",
        flatten=flatten,
        reduce_dimensionality=reduce_dimensionality,
    )

    return convnet


def model_dense(input_dim: int = 16, output_dim: int = 8):
    from simulai.templates import NetworkInstanceGen

    # Configuring model

    auto_gen = NetworkInstanceGen(architecture="dense")

    convnet = auto_gen(
        input_dim=input_dim, output_dim=output_dim, activation="tanh", name="dense_net"
    )

    return convnet


class TestAutoGenNet(TestCase):
    def setUp(self) -> None:
        pass

    def test_autogen_convnet_2d_eval(self):
        input_data, output_data = generate_data_2d(
            n_samples=100, image_size=(16, 16), n_inputs=1, n_outputs=16
        )

        convnet = model_2d()

        estimated_output_data = convnet.eval(input_data=input_data)

        assert estimated_output_data.shape == output_data.shape, (
            "The output of eval is not correct."
            f" Expected {output_data.shape},"
            f" but received {estimated_output_data.shape}."
        )

    def test_autogen_convnet_1d_eval(self):
        input_data, output_data = generate_data_1d(
            n_samples=100, vector_size=16, n_inputs=1, n_outputs=16
        )

        convnet = model_1d()

        estimated_output_data = convnet.eval(input_data=input_data)

        assert estimated_output_data.shape == output_data.shape, (
            "The output of eval is not correct."
            f" Expected {output_data.shape},"
            f" but received {estimated_output_data.shape}."
        )

    def test_autogen_upsample_convnet_2d_eval(self):
        input_data, output_data = generate_data_2d(
            n_samples=100, image_size=(16, 16), n_inputs=1, n_outputs=16
        )

        convnet = model_2d(
            reduce_dimensionality=False,
            flatten=False,
            channels=1,
            input_dim=(None, 16, 1, 1),
            output_dim=(None, 1, 16, 16),
        )

        estimated_output_data = convnet.eval(input_data=output_data[:, :, None, None])

        assert estimated_output_data.shape == input_data.shape, (
            "The output of eval is not correct."
            f" Expected {output_data.shape},"
            f" but received {estimated_output_data.shape}."
        )

    def test_autogen_upsample_convnet_1d_eval(self):
        input_data, output_data = generate_data_1d(
            n_samples=100, vector_size=16, n_inputs=1, n_outputs=16
        )

        convnet = model_1d(
            reduce_dimensionality=False,
            flatten=False,
            channels=1,
            input_dim=(None, 16, 1),
            output_dim=(None, 1, 16),
        )

        estimated_output_data = convnet.eval(input_data=output_data[:, :, None])

        assert estimated_output_data.shape == input_data.shape, (
            "The output of eval is not correct."
            f" Expected {output_data.shape},"
            f" but received {estimated_output_data.shape}."
        )

    def test_densenetwork_reduce(self) -> None:
        input_data = np.random.rand(100, 64)
        output_data = np.random.rand(100, 8)

        net = model_dense(input_dim=64, output_dim=8)
        net.summary()

        output_estimated = net.eval(input_data=input_data)

        assert output_estimated.shape == output_data.shape

    def test_densenetwork_increase(self) -> None:
        input_data = np.random.rand(100, 8)
        output_data = np.random.rand(100, 64)

        net = model_dense(input_dim=8, output_dim=64)
        net.summary()

        output_estimated = net.eval(input_data=input_data)

        assert output_estimated.shape == output_data.shape

    def test_autoencoder_mlp(self) -> None:
        from simulai.models import AutoencoderMLP

        input_data = np.random.rand(100, 64)

        autoencoder = AutoencoderMLP(
            input_dim=64, latent_dim=8, output_dim=64, activation="tanh"
        )

        autoencoder.summary()

        estimated_data = autoencoder.eval(input_data=input_data)

        assert estimated_data.shape == input_data.shape

        # Removing explicit reference to output_dim
        autoencoder = AutoencoderMLP(input_dim=64, latent_dim=8, activation="tanh")

        autoencoder.summary()

        estimated_data = autoencoder.eval(input_data=input_data)

        assert estimated_data.shape == input_data.shape

    def test_autoencoder_cnn(self) -> None:
        from simulai.models import AutoencoderCNN

        input_data = np.random.rand(100, 1, 64, 64)

        autoencoder = AutoencoderCNN(
            input_dim=(None, 1, 64, 64),
            latent_dim=8,
            output_dim=(None, 1, 64, 64),
            activation="tanh",
            case="2d",
        )

        estimated_data = autoencoder.eval(input_data=input_data)

        assert estimated_data.shape == input_data.shape

        # Removing explicit reference to output_dim
        autoencoder = AutoencoderCNN(
            input_dim=(None, 1, 64, 64), latent_dim=8, activation="tanh", case="2d"
        )

        estimated_data = autoencoder.eval(input_data=input_data)

        assert estimated_data.shape == input_data.shape

    def test_autoencoder_koopman(self) -> None:
        from simulai.models import AutoencoderKoopman

        input_data = np.random.rand(100, 1, 64, 64)

        autoencoder = AutoencoderKoopman(
            input_dim=(None, 1, 64, 64),
            latent_dim=8,
            output_dim=(None, 1, 64, 64),
            activation="tanh",
            architecture="cnn",
            case="2d",
        )

        estimated_data = autoencoder.predict(input_data=input_data, n_steps=1)

        assert estimated_data.shape == input_data.shape

        # Removing explicit reference to output_dim
        autoencoder = AutoencoderKoopman(
            input_dim=(None, 1, 64, 64),
            latent_dim=8,
            activation="tanh",
            architecture="cnn",
            case="2d",
        )

        autoencoder.summary()
        estimated_data = autoencoder.predict(input_data=input_data, n_steps=1)

        assert estimated_data.shape == input_data.shape

    def test_autoencoder_rectangle(self) -> None:
        from simulai.models import AutoencoderVariational

        input_data = np.random.rand(100, 1, 64, 128)

        autoencoder = AutoencoderVariational(
            input_dim=(None, 1, 64, 128),
            latent_dim=8,
            activation="tanh",
            architecture="cnn",
            case="2d",
        )

        estimated_data = autoencoder.eval(input_data=input_data)

        autoencoder.summary()

        assert estimated_data.shape == input_data.shape

    def test_autoencoder_rectangle_shallow(self) -> None:
        from simulai.models import AutoencoderVariational

        input_data = np.random.rand(100, 1, 64, 128)

        autoencoder = AutoencoderVariational(
            input_dim=(None, 1, 64, 128),
            latent_dim=8,
            activation="tanh",
            architecture="cnn",
            case="2d",
            shallow=True,
        )

        estimated_data = autoencoder.eval(input_data=input_data)

        autoencoder.summary()

        assert estimated_data.shape == input_data.shape

    def test_autoencoder_variational_mlp(self) -> None:
        from simulai.models import AutoencoderVariational

        input_data = np.random.rand(100, 1_000)

        autoencoder = AutoencoderVariational(
            input_dim=1_000,
            latent_dim=8,
            activation="tanh",
            architecture="dense",
            case="2d",
        )

        estimated_data = autoencoder.eval(input_data=input_data)

        autoencoder.summary()

        assert estimated_data.shape == input_data.shape
