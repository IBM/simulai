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
from simulai.models import Transformer
from simulai.regression import DenseNetwork

DEVICE = configure_device()


class TestTransformer(TestCase):
    def setUp(self) -> None:
        pass

    def test_instantiate(self):

        num_heads = 4
        embed_dim = 128
        hidden_dim = int(embed_dim//2)
        number_of_encoders = 2
        number_of_decoders = 2
        output_size = embed_dim
        n_samples = 100

        input_data = np.random.rand(n_samples, embed_dim)

        # Configuration for the fully-connected branch network
        config = {
            "layers_units": [hidden_dim, hidden_dim, hidden_dim],  # Hidden layers
            "activations": 'Wavelet',
            "input_size": embed_dim,
            "output_size": embed_dim,
            "name": "mlp_layer",
        }

        # Instantiating and training the surrogate model

        transformer = Transformer(num_heads_encoder=num_heads,
                                  num_heads_decoder=num_heads,
                                  embed_dim_encoder=embed_dim,
                                  embed_dim_decoder=embed_dim,
                                  encoder_activation='relu',
                                  decoder_activation='relu',
                                  encoder_mlp_layer_config=config,
                                  decoder_mlp_layer_config=config,
                                  number_of_encoders=number_of_encoders,
                                  number_of_decoders=number_of_decoders)

        transformer.summary()

        estimated_output_data = transformer(input_data)

        assert estimated_output_data.shape == (n_samples, embed_dim), "The output has not the expected shape."

        print(estimated_output_data.shape)
