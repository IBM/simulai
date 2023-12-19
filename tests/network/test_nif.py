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
    n_outputs: int = 2,
    device: Optional[str] = "cpu",
):
    from simulai.regression import DenseNetwork
    from simulai.models import NIF

    n_inputs = 4
    n_outputs = n_outputs

    n_latent = 50

    # Configuration for the fully-connected shape network
    shape_config = {
        "layers_units": [50, 50, 50],  # Hidden layers
        "activations": "elu",
        "input_size": 1,
        "output_size": n_outputs,
        "name": "shape_net",
    }

    # Configuration for the fully-connected parameter network
    parameter_config = {
        "layers_units": [50, 50, 50],  # Hidden layers
        "activations": "elu",
        "input_size": n_inputs,
        "output_size": n_latent,
        "name": "parameter_net",
    }

    # Instantiating and training the surrogate model
    shape_net = DenseNetwork(**shape_config)
    parameter_net = DenseNetwork(**parameter_config)

    # It prints a summary of the network features
    shape_net.summary()
    parameter_net.summary()

    config = {
        "shape_network": shape_net,
        "parameter_network": parameter_net,
        "var_dim": n_outputs,
        "model_id": "deeponet",
    }

    net = NIF(**config, devices=device)

    return net

class TestNIF(TestCase):
    def setUp(self) -> None:
        pass

    def test_nif_forward(self):
        for device in ["cpu", "gpu", None]:
            net = model(device=device)
            net.summary()

            # Checking if the model is coretly placed when no device is
            # informed
            if not device:
                assert net.device == "cpu", (
                    "When no device is provided it is expected the model"
                    + f"being on cpu, but received {net.device}."
                )

            print(f"Network has {net.n_parameters} parameters.")

            data_shape = torch.rand(1_000, 1)
            data_parameters = torch.rand(1_000, 4)

            output = net.forward(input_shape=data_shape, input_parameter=data_parameters)
            assert output.shape[1] == 2, "The network output is not like expected."

            output = net.eval_subnetwork(name="parameter", input_data=data_parameters)
            assert output.shape[1] == 50, "The network output is not like expected."
            assert isinstance(output, np.ndarray)

    def test_nif_train(self):
        from simulai.optimization import Optimizer

        optimizer_config = {"lr": 1e-3}

        data_shape = torch.rand(1_000, 1)
        data_parameter = torch.rand(1_000, 4)
        output_target = torch.rand(1_000, 1)

        n_epochs = 1_00
        maximum_values = (1 / np.linalg.norm(output_target, 2, axis=0)).tolist()
        params = {"lambda_1": 0.0, "lambda_2": 1e-10, "weights": maximum_values}

        input_data = {"input_parameter": data_parameter, "input_shape": data_shape}

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

        output = net.forward(input_shape=data_shape, input_parameter=data_parameter)

        assert output.shape[1] == 1, "The network output is not like expected."


