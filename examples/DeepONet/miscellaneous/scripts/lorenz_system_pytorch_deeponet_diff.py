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

import matplotlib.pyplot as plt
import numpy as np

# In order to execute this script, it is necessary to
# set the environment variable engine as "pytorch" before initializing
# simulai
os.environ["engine"] = "pytorch"

from examples.utils.lorenz_solver import lorenz_solver_forcing
from simulai.metrics import L2Norm
from simulai.models import DeepONet
from simulai.optimization import Optimizer
from simulai.regression import DenseNetwork


# This is a very basic script for exploring the PDE solving via
# feedforward fully-connected neural networks
class TestLorenzTorch(TestCase):
    def setUp(self) -> None:
        pass

    def test_lorenz_torch(self):
        dt = 0.0025
        T_max = 50
        rho = 28
        beta = 8 / 3
        beta_str = "8/3"
        sigma = 10
        A = 10
        n_steps = int(T_max / dt)
        forcings_file = "forcings.npy"

        # Forcing terms
        if not os.path.isfile(forcings_file):
            forcings = A * np.random.rand(n_steps, 3)
            np.save("forcings.npy", forcings)
        else:
            forcings = np.load("forcings.npy")

        initial_state = np.array([1, 2, 3])[None, :]
        lorenz_data, derivative_lorenz_data = lorenz_solver_forcing(
            rho=rho,
            dt=dt,
            T=T_max,
            sigma=sigma,
            initial_state=initial_state,
            forcing=forcings,
            beta=beta,
            beta_str=beta_str,
            data_path="on_memory",
            solver="RK45",
        )

        # The fraction of data used for training the model.
        train_fraction = 0.8

        # Dumping the CSV data containing the simulation data
        # to a Pandas dataframe.
        input_data = lorenz_data
        output_data = derivative_lorenz_data

        # Choosing the number of training and testing samples
        n_samples = input_data.shape[0]
        train_samples = int(train_fraction * n_samples)
        test_samples = n_samples - train_samples

        input_labels = ["x", "y", "z"]
        output_labels = ["x_dot", "y_dot", "z_dot"]

        time = np.arange(0, n_samples, 1) * dt

        # Training dataset
        train_field_data = input_data[:train_samples]
        train_forcing_data = forcings[:train_samples]
        train_output_data = output_data[:train_samples]
        time_train = time[:train_samples]

        # Testing dataset
        test_input_data = input_data[train_samples:]
        test_forcing_data = forcings[train_samples:]
        test_output_data = output_data[train_samples:]
        time_test = time[train_samples:]

        n_inputs = len(input_labels)
        n_outputs = len(output_labels)

        lambda_1 = 1e-5  # Penalty for the L¹ regularization (Lasso)
        lambda_2 = 1e-5  # Penalty factor for the L² regularization
        n_epochs = int(3e3)  # Maximum number of iterations for ADAM
        lr = 1e-3  # Initial learning rate for the ADAM algorithm
        n_latent = 50

        # Configuration for the fully-connected trunk network
        trunk_config = {
            "layers_units": [50, 50, 50],  # Hidden layers
            "activations": "relu",
            "input_size": n_inputs,
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

        optimizer_config = {"lr": lr}

        # Maximum derivative magnitudes to be used as loss weights
        maximum_values = (1 / np.linalg.norm(train_output_data, 2, axis=0)).tolist()

        params = {"lambda_1": lambda_1, "lambda_2": lambda_2, "weights": maximum_values}

        # It prints a summary of the network features
        trunk_net.summary()
        branch_net.summary()

        input_data = {
            "input_branch": train_forcing_data,
            "input_trunk": train_field_data,
        }

        lorenz_net = DeepONet(
            trunk_network=trunk_net,
            branch_network=branch_net,
            var_dim=n_outputs,
            product_type="dense",
            model_id="lorenz_net",
        )

        optimizer = Optimizer("adam", params=optimizer_config)

        optimizer.fit(
            op=lorenz_net,
            input_data=input_data,
            target_data=train_output_data,
            n_epochs=n_epochs,
            loss="wrmse",
            params=params,
            device="gpu",
        )

        approximated_data = lorenz_net.eval(
            trunk_data=test_input_data, branch_data=test_forcing_data
        )

        l2_norm = L2Norm()

        error = 100 * l2_norm(
            data=approximated_data, reference_data=test_output_data, relative_norm=True
        )

        for ii in range(n_inputs):
            plt.plot(approximated_data[:, ii], label="Approximated")
            plt.plot(test_output_data[:, ii], label="Exact")
            plt.legend()
            plt.show()

        print(f"Approximation error for the derivatives: {error} %")
