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

import matplotlib.pyplot as plt
import numpy as np
import sympy

# In order to execute this script, it is necessary to
# set the environment variable engine as "pytorch" before initializing
# simulai
os.environ["engine"] = "pytorch"

from examples.utils.lorenz_solver import lorenz_solver
from simulai.metrics import L2Norm
from simulai.optimization import Optimizer
from simulai.regression import DenseNetwork
from simulai.residuals import SymbolicOperator
from simulai.tokens import D


# This is a very basic script for exploring the PDE solving via
# feedforward fully-connected neural networks
class TestLorenzTorch:
    def __init__(self):
        pass

    def test_lorenz_torch(self):
        dt = 0.005
        T_max = 50
        rho = 28
        beta = 8 / 3
        beta_str = "8/3"
        sigma = 10

        initial_state = np.array([1, 2, 3])[None, :]
        lorenz_data, derivative_lorenz_data = lorenz_solver(
            rho=rho,
            dt=dt,
            T=T_max,
            sigma=sigma,
            initial_state=initial_state,
            beta=beta,
            beta_str=beta_str,
            data_path="/tmp",
        )

        x = sympy.Symbol("x")
        y = sympy.Symbol("y")
        z = sympy.Symbol("z")
        t = sympy.Symbol("t")

        f_x = D(x, t) - sigma * (y - x)
        f_y = D(y, t) - x * (rho - z) + y
        f_z = D(z, t) - x * y + beta * z

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

        input_labels = ["t"]
        output_labels = ["x", "y", "z"]

        time = np.arange(0, n_samples, 1) * dt

        # Training dataset
        train_input_data = input_data[:train_samples]
        train_output_data = output_data[:train_samples]
        time_train = time[:train_samples]

        # Testing dataset
        test_input_data = input_data[train_samples:]
        test_output_data = output_data[train_samples:]
        time_test = time[train_samples:]

        n_inputs = len(input_labels)
        n_outputs = len(output_labels)

        n_epochs = 2_000  # Maximum number of iterations for ADAM
        lr = 1e-3  # Initial learning rate for the ADAM algorithm

        # Configuration for the fully-connected network
        config = {
            "layers_units": [50, 50, 50],
            "activations": "tanh",
            "input_size": 1,
            "output_size": n_outputs,
            "name": "lorenz_net",
        }

        optimizer_config = {"lr": lr}

        # Instantiating and training the surrogate model
        lorenz_net = DenseNetwork(**config)
        residual = SymbolicOperator(
            expressions=[f_x, f_y, f_z],
            input_vars=[t],
            output_vars=[x, y, z],
            function=lorenz_net,
        )

        # It prints a summary of the network features
        lorenz_net.summary()

        optimizer = Optimizer("adam", params=optimizer_config)

        params = {
            "residual": residual,
            "initial_input": time_train[:, None],
            "initial_state": train_input_data,
            "weights": [1e4, 1e4, 1e4],
            "initial_penalty": 1,
        }

        optimizer.fit(
            op=lorenz_net,
            input_data=time_train[1:, None],
            n_epochs=n_epochs,
            loss="pirmse",
            params=params,
        )

        approximated_data = lorenz_net.eval(input_data=time_train[:, None])

        l2_norm = L2Norm()

        error = 100 * l2_norm(
            data=approximated_data, reference_data=train_input_data, relative_norm=True
        )

        for ii in range(n_outputs):
            plt.plot(approximated_data[:, ii], label="Approximated")
            plt.plot(train_input_data[:, ii], label="Exact")
            plt.legend()
            plt.show()

        print(f"Approximation error for the derivatives: {error} %")
