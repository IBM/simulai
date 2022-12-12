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
import torch

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

class TestDeeponet(TestCase):

    def setUp(self) -> None:
        self.errors = list()

    def test_deeponet_forward(self):

        net = model()

        data_trunk = torch.rand(1_000, 1)
        data_branch = torch.rand(1_000, 4)

        output = net.forward(input_trunk=data_trunk, input_branch=data_branch)

        assert output.shape[1] == 2, "The network output is not like expected."

    def test_deeponet_train(self):

        from simulai.optimization import Optimizer

        optimizer_config = {'lr': 1e-3}

        data_trunk = torch.rand(1_000, 1)
        data_branch = torch.rand(1_000, 4)
        output_target = torch.rand(1_000, 2)

        n_epochs = 1_000
        maximum_values = (1 / np.linalg.norm(output_target, 2, axis=0)).tolist()
        params = {'lambda_1': 0.0, 'lambda_2': 1e-10, 'weights': maximum_values}

        input_data = {'input_branch': data_branch, 'input_trunk': data_trunk}

        optimizer = Optimizer('adam', params=optimizer_config)
        net = model()

        optimizer.fit(op=net, input_data=input_data, target_data=output_target,
                      n_epochs=n_epochs, loss="wrmse", params=params, device='gpu')

        output = net.forward(input_trunk=data_trunk, input_branch=data_branch)

        assert output.shape[1] == 2, "The network output is not like expected."