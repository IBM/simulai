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

from simulai.file import SPFile

# Model template
def model_2d(
    reduce_dimensionality: bool = True,
    flatten: bool = True,
    channels: int = 2,
    input_dim: tuple = (None, 1, 16, 16),
    output_dim: tuple = (None, 16, 1, 1),
    unflattened_size: tuple = None,
):
    from simulai.templates import NetworkInstanceGen

    # Configuring model

    auto_gen = NetworkInstanceGen(architecture="cnn",
                                  dim="2d",
                                  unflattened_size=unflattened_size)

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


class TestSPFile(TestCase):
    def setUp(self) -> None:
        pass

    def test_model_without_arguments(self):

        model = model_2d()

        try:
            filemng = SPFile()
            filemng.write(
                save_dir='/tmp/',
                name=f"{id(model)}",
                model=model,
                template=model_2d,
            )

            filemng = SPFile()
            filemng.read(model_path=f"/tmp/{id(model)}")

        except Exception:

            raise Exception(f"It was not possible to save/restore the model {model}.")

    def test_model_with_arguments(self):

        channels = 4
        input_dim = (None, 1, 32, 32)
        output_dim = (None, 32, 1, 1)

        model = model_2d(channels=channels,
                         input_dim=input_dim,
                         output_dim=output_dim)

        args = {
                'channels': channels,
                'input_dim': input_dim,
                'output_dim': output_dim
                }

        try:
            filemng = SPFile()
            filemng.write(
                save_dir='/tmp/',
                name=f"{id(model)}",
                model=model,
                template=model_2d,
                args=args
            )

            filemng = SPFile()
            filemng.read(model_path=f"/tmp/{id(model)}")

        except Exception:

            raise Exception(f"It was not possible to save/restore the model {model}.")

