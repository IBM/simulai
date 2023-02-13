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

# In order to execute this script, it is necessary to
# set the environment variable engine as "pytorch" before initializing
# simulai
os.environ["engine"] = "pytorch"

from simulai.metrics import L2Norm
from simulai.optimization import Optimizer
from simulai.regression import DenseNetwork
from simulai.residuals import SymbolicOperator


# This is a very basic script for exploring the PDE solving via
# feedforward fully-connected neural networks
class TestLorenzTorch:
    def __init__(self):
        pass

    def test_manufactured_torch(self):
        N = 10_000
        n = 1_000
        T_max = 0.5
        omega = 40
        mu = 0.25

        time_train = (np.random.rand(n) * T_max)[:, None]
        time_eval = np.linspace(0, T_max, N)[:, None]
        time_ext = np.linspace(T_max, T_max + 0.5, N)[:, None]

        def dataset(t: np.ndarray = None) -> np.ndarray:
            return (t - mu) ** 2 * np.cos(omega * np.pi * t)

        # Datasets used for comparison
        u_data = dataset(t=time_eval)
        u_data_ext = dataset(t=time_ext)

        # The expression we aim at minimizing
        f = "D(u, t) - 2*(t - mu)*cos(omega*pi*t) + omega*pi*((t - mu)**2)*sin(omega*pi*t)"

        input_labels = ["t"]
        output_labels = ["u"]

        n_inputs = len(input_labels)
        n_outputs = len(output_labels)

        n_epochs = 10_000  # Maximum number of iterations for ADAM
        lr = 1e-3  # Initial learning rate for the ADAM algorithm

        # Configuration for the fully-connected network
        config = {
            "layers_units": [50, 50, 50],
            "activations": "tanh",
            "input_size": n_inputs,
            "output_size": n_outputs,
            "name": "net",
        }

        optimizer_config = {"lr": lr}

        # Instantiating and training the surrogate model
        net = DenseNetwork(**config)

        residual = SymbolicOperator(
            expressions=[f],
            input_vars=["t"],
            output_vars=["u"],
            function=net,
            constants={"omega": omega, "mu": mu},
            engine="torch",
        )

        # It prints a summary of the network features
        net.summary()

        optimizer = Optimizer("adam", params=optimizer_config)

        params = {
            "residual": residual,
            "initial_input": np.array([0])[:, None],
            "initial_state": u_data[0],
            "weights": [1, 1, 1],
            "initial_penalty": 1,
        }

        optimizer.fit(
            op=net,
            input_data=time_train,
            n_epochs=n_epochs,
            loss="pirmse",
            params=params,
        )

        # Evaluation in training dataset
        approximated_data = net.eval(input_data=time_eval)

        l2_norm = L2Norm()

        error = 100 * l2_norm(
            data=approximated_data, reference_data=u_data, relative_norm=True
        )

        for ii in range(n_outputs):
            plt.plot(time_eval, approximated_data, label="Approximated")
            plt.plot(time_eval, u_data, label="Exact")
            plt.xlabel("t")
            plt.ylabel(f"{output_labels[ii]}")
            plt.legend()
            plt.show()

        print(f"Approximation error for the derivatives: {error} %")

        # Evaluation in testing dataset
        approximated_data = net.eval(input_data=time_ext)

        l2_norm = L2Norm()

        error = 100 * l2_norm(
            data=approximated_data, reference_data=u_data_ext, relative_norm=True
        )

        for ii in range(n_outputs):
            plt.plot(time_ext, approximated_data, label="Approximated")
            plt.plot(time_ext, u_data_ext, label="Exact")
            plt.xlabel("t")
            plt.ylabel(f"{output_labels[ii]}")
            plt.legend()
            plt.show()

        print(f"Approximation error for the derivatives: {error} %")

    def test_manufactured_implicit_torch(self):
        N = 5_000
        n = 1_000
        T_max = 0.5
        sigma = 10
        mu = 0.25

        time_train = (np.random.rand(n) * T_max)[:, None]
        time_eval = np.linspace(0, T_max, N)[:, None]
        time_ext = np.linspace(T_max, T_max + 0.5, N)[:, None]

        # Producing reference data
        def dataset(t: np.ndarray = None) -> np.ndarray:
            return np.exp(-sigma * (t - mu) ** 2)

        # Datasets used for comparison
        u_data = dataset(t=time_eval)
        u_data_ext = dataset(t=time_ext)

        # The expression we aim at minimizing
        f = "D(u, t) + 2*sigma*(t - mu)*u"

        input_labels = ["t"]
        output_labels = ["u"]

        n_inputs = len(input_labels)
        n_outputs = len(output_labels)

        n_epochs = 5_000  # Maximum number of iterations for ADAM
        lr = 1e-3  # Initial learning rate for the ADAM algorithm

        # Configuration for the fully-connected network
        net_config = {
            "layers_units": [50, 50, 50],
            "activations": "tanh",
            "input_size": n_inputs,
            "output_size": n_outputs,
            "name": "net",
        }

        optimizer_config = {"lr": lr}

        # Instantiating and training the surrogate model
        net = DenseNetwork(**net_config)

        residual = SymbolicOperator(
            expressions=[f],
            input_vars=["t"],
            output_vars=["u"],
            function=net,
            constants={"sigma": sigma, "mu": mu},
            device="cpu",
            engine="torch",
        )

        # It prints a summary of the network features
        net.summary()

        optimizer = Optimizer("adam", params=optimizer_config)

        params = {
            "residual": residual,
            "initial_input": np.array([0])[:, None],
            "initial_state": u_data[0],
            "weights": [1, 1, 1],
            "initial_penalty": 1,
        }

        optimizer.fit(
            op=net,
            input_data=time_train,
            n_epochs=n_epochs,
            loss="pirmse",
            params=params,
        )

        # Evaluation in training dataset
        approximated_data = net.eval(input_data=time_eval)

        l2_norm = L2Norm()

        error = 100 * l2_norm(
            data=approximated_data, reference_data=u_data, relative_norm=True
        )

        for ii in range(n_outputs):
            plt.plot(time_eval, approximated_data, label="Approximated")
            plt.plot(time_eval, u_data, label="Exact")
            plt.legend()
            plt.show()

        print(f"Approximation error: {error} %")

        # Evaluation in testing dataset
        approximated_data = net.eval(input_data=time_ext)

        l2_norm = L2Norm()

        error = 100 * l2_norm(
            data=approximated_data, reference_data=u_data_ext, relative_norm=True
        )

        for ii in range(n_outputs):
            plt.plot(time_ext, approximated_data, label="Approximated")
            plt.plot(time_ext, u_data_ext, label="Exact")
            plt.legend()
            plt.show()

        print(f"Approximation error: {error} %")
