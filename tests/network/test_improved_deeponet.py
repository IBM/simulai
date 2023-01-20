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

from utils import configure_device

DEVICE = configure_device()

# Model template
def model(product_type=None, multiply_by_trunk:bool=False, n_outputs:int=4, residual:bool=False):

    import numpy as np

    from simulai.regression import ConvexDenseNetwork, SLFNN
    from simulai.models import ImprovedDeepONet

    n_latent = 100
    n_inputs_b = 4
    n_inputs_t = 1

    if product_type == None:
        output_size = n_latent * n_outputs
    else:
        output_size = n_latent

    # Configuration for the fully-connected trunk network
    trunk_config = {
        'layers_units': 6 * [100],  # Hidden layers
        'activations': 'tanh',
        'input_size': n_inputs_t,
        'output_size': output_size,
        'name': 'trunk_net'
    }

    # Configuration for the fully-connected branch network
    branch_config = {
        'layers_units': 6 * [100],  # Hidden layers
        'activations': 'tanh',
        'input_size': n_inputs_b,
        'output_size': n_latent * n_outputs,
        'name': 'branch_net',
    }

    # Instantiating and training the surrogate model
    trunk_net = ConvexDenseNetwork(**trunk_config)
    branch_net = ConvexDenseNetwork(**branch_config)

    encoder_trunk = SLFNN(input_size=n_inputs_t, output_size=100, activation='tanh')
    encoder_branch = SLFNN(input_size=n_inputs_b, output_size=100, activation='tanh')

    # It prints a summary of the network features
    trunk_net.summary()
    branch_net.summary()

    net = ImprovedDeepONet(trunk_network=trunk_net,
                           branch_network=branch_net,
                           encoder_trunk=encoder_trunk,
                           encoder_branch=encoder_branch,
                           var_dim=n_outputs,
                           rescale_factors=np.random.rand(n_outputs),
                           product_type=product_type,
                           multiply_by_trunk=multiply_by_trunk,
                           residual=residual,
                           devices=DEVICE,
                           model_id='net')

    return net

class TestImprovedDeeponet(TestCase):

    def setUp(self) -> None:
        pass

    def test_deeponet_forward(self):
        
        net = model()
        net.summary()

        data_trunk = torch.rand(1_000, 1)
        data_branch = torch.rand(1_000, 4)

        print(f"Network has {net.n_parameters} parameters.")
        
        output = net.forward(input_trunk=data_trunk, input_branch=data_branch)

        assert output.shape[1] == 4, "The network output is not like expected."

        output = net.eval_subnetwork(name='trunk', trunk_data=data_trunk, branch_data=data_branch)
        assert output.shape[1] == 400, "The network output is not like expected."
        assert isinstance(output, np.ndarray)

        output = net.eval_subnetwork(name='branch', trunk_data=data_trunk, branch_data=data_branch)
        assert output.shape[1] == 400, "The network output is not like expected."
        assert isinstance(output, np.ndarray)

    def test_deeponet_train(self):

        from simulai.optimization import Optimizer

        optimizer_config = {'lr': 1e-3}

        data_trunk = torch.rand(1_000, 1)
        data_branch = torch.rand(1_000, 4)
        output_target = torch.rand(1_000, 2)

        n_epochs = 1_00
        maximum_values = (1 / np.linalg.norm(output_target, 2, axis=0)).tolist()
        params = {'lambda_1': 0.0, 'lambda_2': 1e-10, 'weights': maximum_values}

        input_data = {'input_branch': data_branch, 'input_trunk': data_trunk}

        optimizer = Optimizer('adam', params=optimizer_config)

        for product_type in [None, 'dense']:

            for multiply_by_trunk in [True, False]:

                print(f"Multiply by trunk: {multiply_by_trunk}, Product type: {product_type}")

                net = model(multiply_by_trunk=multiply_by_trunk, product_type=product_type)

                optimizer.fit(op=net, input_data=input_data, target_data=output_target,
                              n_epochs=n_epochs, loss="wrmse", params=params, device=DEVICE)

                output = net.forward(input_trunk=data_trunk, input_branch=data_branch)

                assert output.shape[1] == 4, "The network output is not like expected."

    # Vanilla DeepONets are single output
    def test_vanilla_deeponet_train(self):

        from simulai.optimization import Optimizer

        optimizer_config = {'lr': 1e-3}

        data_trunk = torch.rand(1_000, 1)
        data_branch = torch.rand(1_000, 4)
        output_target = torch.rand(1_000, 1)

        n_epochs = 1_00
        maximum_values = (1 / np.linalg.norm(output_target, 2, axis=0)).tolist()
        params = {'lambda_1': 0.0, 'lambda_2': 1e-10, 'weights': maximum_values}

        input_data = {'input_branch': data_branch, 'input_trunk': data_trunk}

        optimizer = Optimizer('adam', params=optimizer_config)

        net = model(n_outputs=1)

        optimizer.fit(op=net, input_data=input_data, target_data=output_target,
                      n_epochs=n_epochs, loss="wrmse", params=params, device=DEVICE)

        output = net.forward(input_trunk=data_trunk, input_branch=data_branch)

        assert output.shape[1] == 1, "The network output is not like expected."
