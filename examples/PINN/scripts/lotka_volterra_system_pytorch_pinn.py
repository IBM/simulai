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

from examples.utils.lotka_volterra_solver import LotkaVolterra
from simulai.optimization import Optimizer
from simulai.regression import DenseNetwork, RBFNetwork
from simulai.residuals import SymbolicOperator


# This is a very basic script for exploring the PDE solving via
# feedforward fully-connected neural networks
class TestLorenzTorch:
    def __init__(self):
        pass

    def test_manufactured_torch(self):
        N = 10_00
        n = 5_000
        n_sampled = 1_000
        T_max = 30
        t = np.linspace(0, T_max, n)

        alpha = 1.1
        beta = 0.4
        gamma = 0.4
        delta = 0.1

        lotka_volterra_solver = LotkaVolterra(
            alpha=alpha, beta=beta, gamma=gamma, delta=delta
        )

        initial_state = np.array([20, 5])

        lotka_volterra_dataset = lotka_volterra_solver.run(initial_state, t)

        indices = np.random.choice(n, n_sampled)
        t_sampled = t[indices, None]
        lotka_volterra_sampled = lotka_volterra_dataset[indices]

        time_train = (np.random.rand(n) * T_max)[:, None]
        time_eval = np.linspace(0, T_max, N)[:, None]

        # The expression we aim at minimizing
        f_x = "D(x, t) - (  alpha * x   - betha * x * y)"
        f_y = "D(y, t) - (delta * x * y - gammma * y   )"

        input_labels = ["t"]
        output_labels = ["x", "y"]

        n_inputs = len(input_labels)
        n_outputs = len(output_labels)

        n_epochs = 20_000  # Maximum number of iterations for ADAM
        lr = 1e-3  # Initial learning rate for the ADAM algorithm

        # Configuration for the fully-connected network
        config = {
            "layers_units": [100, 100],
            "activations": "elu",
            "input_size": n_inputs,
            "output_size": n_outputs,
            "name": "net",
        }

        optimizer_config = {"lr": lr}

        # Instantiating and training the surrogate model
        net = DenseNetwork(**config)

        residual = SymbolicOperator(
            expressions=[f_x, f_y],
            input_vars=input_labels,
            output_vars=output_labels,
            function=net,
            constants={"alpha": alpha, "betha": beta, "gammma": gamma, "delta": delta},
            engine="torch",
        )

        # It prints a summary of the network features
        net.summary()

        optimizer = Optimizer("adam", params=optimizer_config)

        params = {
            "residual": residual,
            "initial_input": t_sampled,
            "initial_state": lotka_volterra_sampled,
            "weights": [1, 1],
            "initial_penalty": 10,
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

        for ii in range(n_outputs):
            plt.plot(time_eval, approximated_data[:, ii], label="Approximated")
            plt.xlabel("t")
            plt.ylabel(f"{output_labels[ii]}")
            plt.legend()
            plt.show()

    def test_manufactured_torch_rbf(self):
        N = 10_00
        n = 5_000
        n_sampled = 1_000
        T_max = 30
        t = np.linspace(0, T_max, n)

        alpha = 1.1
        beta = 0.4
        gamma = 0.4
        delta = 0.1

        lotka_volterra_solver = LotkaVolterra(
            alpha=alpha, beta=beta, gamma=gamma, delta=delta
        )

        initial_state = np.array([20, 5])

        lotka_volterra_dataset = lotka_volterra_solver.run(initial_state, t)

        indices = np.random.choice(n, n_sampled)
        t_sampled = t[indices, None]
        lotka_volterra_sampled = lotka_volterra_dataset[indices]

        time_train = (np.random.rand(n) * T_max)[:, None]
        time_eval = np.linspace(0, T_max, N)[:, None]

        # The expression we aim at minimizing
        f_x = "D(x, t) - (  alpha * x   - betha * x * y)"
        f_y = "D(y, t) - (delta * x * y - gammma * y   )"

        input_labels = ["t"]
        output_labels = ["x", "y"]

        n_inputs = len(input_labels)
        n_outputs = len(output_labels)

        n_epochs = 20_000  # Maximum number of iterations for ADAM
        lr = 1e-3  # Initial learning rate for the ADAM algorithm

        # Configuration for the fully-connected network
        config = {
            "name": "lorenz_net",
            "xmax": T_max,
            "xmin": 0,
            "Nk": 20,
            "output_size": n_outputs,
        }

        optimizer_config = {"lr": lr}

        # Instantiating and training the surrogate model
        net = RBFNetwork(**config)

        residual = SymbolicOperator(
            expressions=[f_x, f_y],
            input_vars=input_labels,
            output_vars=output_labels,
            function=net,
            gradient=net.gradient,
            constants={"alpha": alpha, "betha": beta, "gammma": gamma, "delta": delta},
            engine="torch",
        )

        # It prints a summary of the network features
        net.summary()

        optimizer = Optimizer("adam", params=optimizer_config)

        params = {
            "residual": residual,
            "initial_input": t_sampled,
            "initial_state": lotka_volterra_sampled,
            "weights": [1, 1],
            "initial_penalty": 10,
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

        for ii in range(n_outputs):
            plt.plot(time_eval, approximated_data[:, ii], label="Approximated")
            plt.xlabel("t")
            plt.ylabel(f"{output_labels[ii]}")
            plt.legend()
            plt.show()
