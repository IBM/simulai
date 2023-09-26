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

from unittest import TestCase
from typing import Union, Optional
import numpy as np
from tests.config import configure_dtype
torch = configure_dtype()
from utils import configure_device

DEVICE = configure_device()


# Model template
def model(
    product_type=None,
    n_outputs: int = 2,
    residual: bool = False,
    multiply_by_trunk: bool = False,
    use_bias: bool = False,
    device:Optional[str]="cpu"
):
    import importlib

    from simulai.regression import DenseNetwork

    if residual is True:
        arch_name = "ResDeepONet"
    else:
        arch_name = "DeepONet"

    models_engine = importlib.import_module("simulai.models", arch_name)

    DeepONet = getattr(models_engine, arch_name)

    n_inputs = 4
    n_outputs = n_outputs

    n_latent = 50

    if use_bias:
        extra_dim = n_outputs
    else:
        extra_dim = 0

    # Configuration for the fully-connected trunk network
    trunk_config = {
        "layers_units": [50, 50, 50],  # Hidden layers
        "activations": "elu",
        "input_size": 1,
        "output_size": n_latent * n_outputs,
        "name": "trunk_net",
    }

    # Configuration for the fully-connected branch network
    branch_config = {
        "layers_units": [50, 50, 50],  # Hidden layers
        "activations": "elu",
        "input_size": n_inputs,
        "output_size": n_latent * n_outputs + extra_dim,
        "name": "branch_net",
    }

    # Instantiating and training the surrogate model
    trunk_net = DenseNetwork(**trunk_config)
    branch_net = DenseNetwork(**branch_config)

    # It prints a summary of the network features
    trunk_net.summary()
    branch_net.summary()

    config = {
        "trunk_network": trunk_net,
        "branch_network": branch_net,
        "var_dim": n_outputs,
        "product_type": product_type,
        "model_id": "deeponet",
    }

    if residual is True:
        config["multiply_by_trunk"] = multiply_by_trunk

    if use_bias is True:
        config["use_bias"] = use_bias

    net = DeepONet(**config, devices=device)

    return net


def model_dense_product(product_type=None, n_outputs: int = 2, device:Optional[str]="cpu"):
    from simulai.models import DeepONet
    from simulai.regression import DenseNetwork

    n_inputs = 4
    n_outputs = n_outputs

    n_latent = 50

    # Configuration for the fully-connected trunk network
    trunk_config = {
        "layers_units": [50, 50, 50],  # Hidden layers
        "activations": "elu",
        "input_size": 1,
        "output_size": n_latent,
        "name": "trunk_net",
    }

    # Configuration for the fully-connected branch network
    branch_config = {
        "layers_units": [50, 50, 50],  # Hidden layers
        "activations": "elu",
        "input_size": n_inputs,
        "output_size": n_latent * n_outputs,
        "name": "branch_net",
    }

    # Instantiating and training the surrogate model
    trunk_net = DenseNetwork(**trunk_config)
    branch_net = DenseNetwork(**branch_config)

    net = DeepONet(
        trunk_network=trunk_net,
        branch_network=branch_net,
        var_dim=n_outputs,
        product_type=product_type,
        model_id="deeponet",
        devices=device,
    )

    return net


def model_conv(product_type=None, device:Optional[str]="cpu"):
    from simulai.models import DeepONet
    from simulai.regression import ConvolutionalNetwork, DenseNetwork

    n_inputs = 1
    n_outputs = 2

    n_latent = 50

    # Configuration for the fully-connected trunk network
    trunk_config = {
        "layers_units": [50, 50, 50],  # Hidden layers
        "activations": "elu",
        "input_size": 1,
        "output_size": n_latent * n_outputs,
        "name": "trunk_net",
    }

    layers = [
        {
            "in_channels": n_inputs,
            "out_channels": 2,
            "kernel_size": 3,
            "stride": 1,
            "padding": 1,
            "after_conv": {"type": "maxpool2d", "kernel_size": 2, "stride": 2},
        },
        {
            "in_channels": 2,
            "out_channels": 4,
            "kernel_size": 3,
            "stride": 1,
            "padding": 1,
            "after_conv": {"type": "maxpool2d", "kernel_size": 2, "stride": 2},
        },
        {
            "in_channels": 4,
            "out_channels": 8,
            "kernel_size": 3,
            "stride": 1,
            "padding": 1,
            "after_conv": {"type": "maxpool2d", "kernel_size": 2, "stride": 2},
        },
        {
            "in_channels": 8,
            "out_channels": n_latent * n_outputs,
            "kernel_size": 3,
            "stride": 1,
            "padding": 1,
            "after_conv": {"type": "maxpool2d", "kernel_size": 2, "stride": 2},
        },
    ]

    # Instantiating and training the surrogate model
    trunk_net = DenseNetwork(**trunk_config)
    branch_net = ConvolutionalNetwork(
        layers=layers, activations="sigmoid", case="2d", name="net", flatten=True
    )

    # It prints a summary of the network features
    trunk_net.summary()
    branch_net.summary(input_shape=[None, 1, 16, 16])

    net = DeepONet(
        trunk_network=trunk_net,
        branch_network=branch_net,
        var_dim=n_outputs,
        product_type=product_type,
        model_id="deeponet",
        devices=device,
    )

    return net

def model_conv_template(product_type:Optional[str]=None, device:Optional[str]=None):

    from simulai.workflows import ConvDeepONet

    n_inputs = 1
    n_outputs = 2
    dim="2d"
    n_latent = 64
    product_type = None

    # Configuration for the fully-connected trunk network
    trunk_config = {
        "layers_units": [50, 50, 50],  # Hidden layers
        "activations": "elu",
        "input_size": n_inputs,
        "output_size": n_latent * n_outputs,
        "name": "trunk_net",
    }

    net_template = ConvDeepONet(trunk_config=trunk_config,
                     n_outputs=n_outputs,
                     dim=dim,
                     n_latent = n_latent,
                     branch_input_dim=(None,1,16,16),
                     shallow=True,
                     product_type=product_type,
                     use_batch_norm=True,
                     branch_activation="elu",
                     )

    return net_template()


class TestDeeponet(TestCase):
    def setUp(self) -> None:
        pass

    def test_deeponet_forward(self):

        for device in ["cpu", "gpu", None]:

            net = model(device=device)
            net.summary()

            # Checking if the model is coretly placed when no device is
            # informed
            if not device:
                assert net.device == "cpu", ("When no device is provided it is expected the model"+
                                             f"being on cpu, but received {net.device}.")

            print(f"Network has {net.n_parameters} parameters.")

            data_trunk = torch.rand(1_000, 1)
            data_branch = torch.rand(1_000, 4)

            output = net.forward(input_trunk=data_trunk, input_branch=data_branch)

            assert output.shape[1] == 2, "The network output is not like expected."

            output = net.eval_subnetwork(name="trunk", input_data=data_trunk)
            assert output.shape[1] == 100, "The network output is not like expected."
            assert isinstance(output, np.ndarray)

            output = net.eval_subnetwork(name="branch", input_data=data_branch)
            assert output.shape[1] == 100, "The network output is not like expected."
            assert isinstance(output, np.ndarray)

    def test_deeponet_train(self):
        from simulai.optimization import Optimizer

        optimizer_config = {"lr": 1e-3}

        data_trunk = torch.rand(1_000, 1)
        data_branch = torch.rand(1_000, 4)
        output_target = torch.rand(1_000, 2)

        n_epochs = 10
        maximum_values = (1 / np.linalg.norm(output_target, 2, axis=0)).tolist()
        params = {"lambda_1": 0.0, "lambda_2": 1e-10, "weights": maximum_values}

        input_data = {"input_branch": data_branch, "input_trunk": data_trunk}

        optimizer = Optimizer("adam", params=optimizer_config)

        model_dict = {None: model, "dense": model_dense_product}

        for multiply_by_trunk in [True, False]:
            net = model(n_outputs=4, residual=True, multiply_by_trunk=multiply_by_trunk)

            optimizer.fit(
                op=net,
                input_data=input_data,
                target_data=output_target,
                n_epochs=n_epochs,
                loss="wrmse",
                params=params,
                device=DEVICE,
            )

            output = net.forward(input_trunk=data_trunk, input_branch=data_branch)

            assert output.shape[1] == 4, "The network output is not like expected."

        for product_type in [None, "dense"]:
            net = model_dict.get(product_type)(product_type=product_type)

            optimizer.fit(
                op=net,
                input_data=input_data,
                target_data=output_target,
                n_epochs=n_epochs,
                loss="wrmse",
                params=params,
                device=DEVICE,
            )

            output = net.forward(input_trunk=data_trunk, input_branch=data_branch)

            assert output.shape[1] == 2, "The network output is not like expected."

        for use_bias in [False, True]:
            net = model(n_outputs=4, use_bias=use_bias)

            optimizer.fit(
                op=net,
                input_data=input_data,
                target_data=output_target,
                n_epochs=n_epochs,
                loss="wrmse",
                params=params,
                device=DEVICE,
            )

            output = net.forward(input_trunk=data_trunk, input_branch=data_branch)

            assert output.shape[1] == 4, "The network output is not like expected."


    # Vanilla DeepONets are single output
    def test_vanilla_deeponet_train(self):
        from simulai.optimization import Optimizer

        optimizer_config = {"lr": 1e-3}

        data_trunk = torch.rand(1_000, 1)
        data_branch = torch.rand(1_000, 4)
        output_target = torch.rand(1_000, 1)

        n_epochs = 1_00
        maximum_values = (1 / np.linalg.norm(output_target, 2, axis=0)).tolist()
        params = {"lambda_1": 0.0, "lambda_2": 1e-10, "weights": maximum_values}

        input_data = {"input_branch": data_branch, "input_trunk": data_trunk}

        optimizer = Optimizer("adam", params=optimizer_config)

        net = model(n_outputs=1)

        optimizer.fit(
            op=net,
            input_data=input_data,
            target_data=output_target,
            n_epochs=n_epochs,
            loss="wrmse",
            params=params,
            device=DEVICE,
        )

        output = net.forward(input_trunk=data_trunk, input_branch=data_branch)

        assert output.shape[1] == 1, "The network output is not like expected."


class TestDeeponet_with_Conv(TestCase):
    def setUp(self) -> None:
        pass

    def test_deeponet_forward(self):
        for device in ["cpu", "gpu", None]:
            net = model_conv(device=device)

            data_trunk = torch.rand(1_000, 1)
            data_branch = torch.rand(1_000, 1, 16, 16)

            output = net.forward(input_trunk=data_trunk, input_branch=data_branch)

            print(f"Network has {net.n_parameters} parameters.")

            assert output.shape[1] == 2, "The network output is not like expected."

    def test_deeponet_template_forward(self):
        net = model_conv_template()

        data_trunk = torch.rand(1_000, 1)
        data_branch = torch.rand(1_000, 1, 16, 16)

        output = net.forward(input_trunk=data_trunk, input_branch=data_branch)

        print(f"Network has {net.n_parameters} parameters.")

        assert output.shape[1] == 2, "The network output is not like expected."

    def test_deeponet_train(self):
        from simulai.optimization import Optimizer

        optimizer_config = {"lr": 1e-3}

        data_trunk = torch.rand(1_000, 1)
        data_branch = torch.rand(1_000, 1, 16, 16)
        output_target = torch.rand(1_000, 2)

        n_epochs = 1_00
        maximum_values = (1 / np.linalg.norm(output_target, 2, axis=0)).tolist()
        params = {"lambda_1": 0.0, "lambda_2": 1e-10, "weights": maximum_values}

        input_data = {"input_branch": data_branch, "input_trunk": data_trunk}

        optimizer = Optimizer("adam", params=optimizer_config)
        net = model_conv()

        optimizer.fit(
            op=net,
            input_data=input_data,
            target_data=output_target,
            n_epochs=n_epochs,
            loss="wrmse",
            params=params,
            device=DEVICE,
        )

        output = net.forward(input_trunk=data_trunk, input_branch=data_branch)

        assert output.shape[1] == 2, "The network output is not like expected."
