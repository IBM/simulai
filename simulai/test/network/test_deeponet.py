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

import numpy as np
from unittest import TestCase

from simulai.file import SPFile
from simulai.metrics import L2Norm
from simulai.optimization import Optimizer

# Model template
def model():

    from simulai.regression import DenseNetwork
    from simulai.models import DeepONet

    n_inputs = 4
    n_outputs = 2

    n_latent = 50

    # Configuration for the fully-connected trunk network
    trunk_config = {
        'layers_units': [50, 50, 50],  # Hidden layers
        'activations': 'elu',
        'input_size': 1,
        'output_size': n_latent * n_outputs,
        'name': 'trunk_net'
    }

    # Configuration for the fully-connected branch network
    branch_config = {
        'layers_units': [50, 50, 50],  # Hidden layers
        'activations': 'elu',
        'input_size': n_inputs,
        'output_size': n_latent * n_outputs,
        'name': 'branch_net'
    }

    # Instantiating and training the surrogate model
    trunk_net = DenseNetwork(**trunk_config)
    branch_net = DenseNetwork(**branch_config)

    # It prints a summary of the network features
    trunk_net.summary()
    branch_net.summary()

    net = DeepONet(trunk_network=trunk_net,
                   branch_network=branch_net,
                   var_dim=n_outputs,
                   model_id='deeponet')

    return net

class TestDenseNetwork(TestCase):

    def setUp(self) -> None:
        self.errors = list()

    def test_deeponet(self):

        net = model()


