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
from scipy.integrate import odeint

os.environ["engine"] = "pytorch"

from simulai.models import AutoencoderKoopman
from simulai.optimization import Optimizer
from simulai.regression import DenseNetwork


# Projection to interval
def project_to_interval(interval, data):
    return interval[1] * (data - data.min()) / (data.max() - data.min()) + interval[0]


# Pendulum numerical solver
class Pendulum:
    def __init__(self, rho: float = None):
        self.rho = rho

    def eval(self, state: np.ndarray = None, t: float = None) -> np.ndarray:
        x = state[0]
        y = state[1]

        x_residual = y
        y_residual = -self.rho * np.sin(x)

        return np.array([x_residual, y_residual])

    def run(self, initial_state, t):
        solution = odeint(self.eval, initial_state, t)

        return np.vstack(solution)


class TestPendulumKAE(TestCase):
    def setUp(self) -> None:
        pass

    def test_kae(self):
        # Basic configurations
        dt = 0.01
        T_max = 300
        n_samples = int(T_max / dt)
        train_fraction = 0.8
        n_samples_train = int(train_fraction * n_samples)
        n_samples_test = n_samples - n_samples_train

        # Generating data
        t = np.arange(0, T_max, dt)

        initial_state = np.array([10, 0])

        solver = Pendulum(rho=1)

        data = solver.run(initial_state, t)
        data_train = data[:n_samples_train]
        data_test = data[n_samples_train:]

        init = data_train[-1:]

        lr = 1e-3
        lambda_1 = 0.0
        lambda_2 = 1e-10
        batch_size = None
        n_epochs = 2_000

        # Configuration for the fully-connected network
        encoder_config = {
            "layers_units": [50, 50],  # Hidden layers
            "activations": "tanh",
            "input_size": 2,
            "output_size": 5,
            "name": "encoder",
        }

        decoder_config = {
            "layers_units": [50, 50],  # Hidden layers
            "activations": "tanh",
            "input_size": 5,
            "output_size": 2,
            "name": "decoder",
        }

        encoder = DenseNetwork(**encoder_config)
        decoder = DenseNetwork(**decoder_config)

        koopman_ae = AutoencoderKoopman(encoder=encoder, decoder=decoder)

        optimizer_config = {"lr": lr}
        params = {"lambda_1": lambda_1, "lambda_2": lambda_2, "m": 5, "alpha_1": 1e-3}

        optimizer = Optimizer(
            "adam",
            params=optimizer_config,
            lr_decay_scheduler_params={
                "name": "ExponentialLR",
                "gamma": 0.9,
                "decay_frequency": 5_000,
            },
        )

        optimizer.fit(
            op=koopman_ae,
            input_data=data_train,
            target_data=data_train,
            batch_size=batch_size,
            n_epochs=n_epochs,
            loss="kaermse",
            params=params,
        )

        data_test_evaluated = koopman_ae.predict(
            input_data=init, n_steps=n_samples_test
        )

        for j in range(2):
            plt.plot(data_test_evaluated[:, j])
            plt.plot(data_test[:, j])
            plt.show()
