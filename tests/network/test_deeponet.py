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
def model(product_type=None, n_outputs:int=2, residual:bool=False, multiply_by_trunk:bool=False):

    import importlib

    from simulai.regression import DenseNetwork

    if residual is True:
        arch_name = 'ResDeepONet'
    else:
        arch_name = 'DeepONet'

    models_engine = importlib.import_module('simulai.models', arch_name)

    DeepONet = getattr(models_engine, arch_name)

    n_inputs = 4
    n_outputs = n_outputs

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

    config = {'trunk_network': trunk_net,
              'branch_network': branch_net,
              'var_dim': n_outputs,
              'product_type': product_type,
              'model_id': 'deeponet'}

    if residual is True:
        config['multiply_by_trunk'] = multiply_by_trunk

    net = DeepONet(**config)

    return net

def model_dense_product(product_type=None, n_outputs:int=2):

    from simulai.regression import DenseNetwork
    from simulai.models import DeepONet

    n_inputs = 4
    n_outputs = n_outputs

    n_latent = 50

    # Configuration for the fully-connected trunk network
    trunk_config = {
        'layers_units': [50, 50, 50],  # Hidden layers
        'activations': 'elu',
        'input_size': 1,
        'output_size': n_latent,
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

    net = DeepONet(trunk_network=trunk_net,
                   branch_network=branch_net,
                   var_dim=n_outputs,
                   product_type=product_type,
                   model_id='deeponet')

    return net

def model_conv(product_type=None):

    from simulai.regression import DenseNetwork, ConvolutionalNetwork
    from simulai.models import DeepONet

    n_inputs = 1
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

    layers = [

        {'in_channels': n_inputs, 'out_channels': 2, 'kernel_size': 3, 'stride': 1, 'padding': 1,
         'after_conv': {'type': 'maxpool2d', 'kernel_size': 2, 'stride': 2}},

        {'in_channels': 2, 'out_channels': 4, 'kernel_size': 3, 'stride': 1, 'padding': 1,
         'after_conv': {'type': 'maxpool2d', 'kernel_size': 2, 'stride': 2}},

        {'in_channels': 4, 'out_channels': 8, 'kernel_size': 3, 'stride': 1, 'padding': 1,
         'after_conv': {'type': 'maxpool2d', 'kernel_size': 2, 'stride': 2}},

        {'in_channels': 8, 'out_channels': n_latent * n_outputs, 'kernel_size': 3, 'stride': 1, 'padding': 1,
         'after_conv': {'type': 'maxpool2d', 'kernel_size': 2, 'stride': 2}}
    ]

    # Instantiating and training the surrogate model
    trunk_net = DenseNetwork(**trunk_config)
    branch_net = ConvolutionalNetwork(layers=layers, activations='sigmoid', case='2d', name='net', flatten=True)

    # It prints a summary of the network features
    trunk_net.summary()
    branch_net.summary(input_shape=[None, 1, 16, 16])

    net = DeepONet(trunk_network=trunk_net,
                   branch_network=branch_net,
                   var_dim=n_outputs,
                   product_type=product_type,
                   model_id='deeponet')

    return net

class TestDeeponet(TestCase):

    def setUp(self) -> None:
        pass

    def test_deeponet_forward(self):
        
        net = model()
        net.summary()

        print(f"Network has {net.n_parameters} parameters.")

        data_trunk = torch.rand(1_000, 1)
        data_branch = torch.rand(1_000, 4)

        output = net.forward(input_trunk=data_trunk, input_branch=data_branch)

        assert output.shape[1] == 2, "The network output is not like expected."

        output = net.eval_subnetwork(name='trunk', input_data=data_trunk)
        assert output.shape[1] == 100, "The network output is not like expected."
        assert isinstance(output, np.ndarray)

        output = net.eval_subnetwork(name='branch', input_data=data_branch)
        assert output.shape[1] == 100, "The network output is not like expected."
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

        model_dict = {None: model, 'dense': model_dense_product}

        for multiply_by_trunk in [True, False]:

            net = model(n_outputs=4, residual=True, multiply_by_trunk=multiply_by_trunk)

            optimizer.fit(op=net, input_data=input_data, target_data=output_target,
                          n_epochs=n_epochs, loss="wrmse", params=params, device=DEVICE)

            output = net.forward(input_trunk=data_trunk, input_branch=data_branch)

            assert output.shape[1] == 4, "The network output is not like expected."

        for product_type in [None, 'dense']:

            net = model_dict.get(product_type)(product_type=product_type)

            optimizer.fit(op=net, input_data=input_data, target_data=output_target,
                          n_epochs=n_epochs, loss="wrmse", params=params, device=DEVICE)

            output = net.forward(input_trunk=data_trunk, input_branch=data_branch)

            assert output.shape[1] == 2, "The network output is not like expected."

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

class TestDeeponet_with_Conv(TestCase):

    def setUp(self) -> None:
        pass

    def test_deeponet_forward(self):

        net = model_conv()

        data_trunk = torch.rand(1_000, 1)
        data_branch = torch.rand(1_000, 1, 16, 16)

        output = net.forward(input_trunk=data_trunk, input_branch=data_branch)

        print(f"Network has {net.n_parameters} parameters.")

        assert output.shape[1] == 2, "The network output is not like expected."

    def test_deeponet_train(self):

        from simulai.optimization import Optimizer

        optimizer_config = {'lr': 1e-3}

        data_trunk = torch.rand(1_000, 1)
        data_branch = torch.rand(1_000, 1, 16, 16)
        output_target = torch.rand(1_000, 2)

        n_epochs = 1_00
        maximum_values = (1 / np.linalg.norm(output_target, 2, axis=0)).tolist()
        params = {'lambda_1': 0.0, 'lambda_2': 1e-10, 'weights': maximum_values}

        input_data = {'input_branch': data_branch, 'input_trunk': data_trunk}

        optimizer = Optimizer('adam', params=optimizer_config)
        net = model_conv()

        optimizer.fit(op=net, input_data=input_data, target_data=output_target,
                      n_epochs=n_epochs, loss="wrmse", params=params, device=DEVICE)

        output = net.forward(input_trunk=data_trunk, input_branch=data_branch)

        assert output.shape[1] == 2, "The network output is not like expected."