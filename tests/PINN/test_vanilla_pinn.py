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

import matplotlib.pyplot as plt
import numpy as np

from simulai.optimization import Optimizer
from simulai.residuals import SymbolicOperator


class TestAllencahnPINN(TestCase):
    def setUp(self) -> None:
        pass

    def test_allen_cahn(self):
        # Our PDE
        # Allen-cahn equation

        f = "D(u, t) - mu*D(D(u, x), x) + alpha*(u**3) + beta*u"

        g_u = "u"
        g_ux = "D(u, x)"

        input_labels = ["x", "t"]
        output_labels = ["u"]

        # Some fixed values
        X_DIM = 25
        T_DIM = 10

        L = 1
        x_0 = -1
        T = 1

        # Generating the training grid

        x_interval = [x_0, L]
        t_interval = [0, T]

        intervals = [x_interval, t_interval]

        # Regular grid
        x_0, x_L = x_interval
        t_0, t_L = t_interval
        dx = (x_L - x_0) / X_DIM
        dt = (t_L - t_0) / T_DIM

        grid = np.mgrid[t_0 + dt : t_L + dt : dt, x_0:x_L:dx]

        data = np.hstack([grid[1].flatten()[:, None], grid[0].flatten()[:, None]])

        data_init = np.linspace(*x_interval, X_DIM)
        u_init = (data_init**2) * np.cos(np.pi * data_init)[:, None]

        # Boundary grids
        data_boundary_x0 = np.hstack(
            [
                x_interval[0] * np.ones((T_DIM, 1)),
                np.linspace(*t_interval, T_DIM)[:, None],
            ]
        )

        data_boundary_xL = np.hstack(
            [
                x_interval[-1] * np.ones((T_DIM, 1)),
                np.linspace(*t_interval, T_DIM)[:, None],
            ]
        )

        data_boundary_t0 = np.hstack(
            [
                np.linspace(*x_interval, X_DIM)[:, None],
                t_interval[0] * np.ones((X_DIM, 1)),
            ]
        )

        n_epochs = 5  # Maximum number of iterations for ADAM
        lr = 1e-3  # Initial learning rate for the ADAM algorithm

        def model():
            from simulai.regression import DenseNetwork

            input_labels = ["x", "t"]
            output_labels = ["u"]

            n_inputs = len(input_labels)
            n_outputs = len(output_labels)

            # Configuration for the fully-connected network
            config = {
                "layers_units": [128, 128, 128, 128],
                "activations": "tanh",
                "input_size": n_inputs,
                "output_size": n_outputs,
                "name": "allen_cahn_net",
            }

            # Instantiating and training the surrogate model
            net = DenseNetwork(**config)

            return net

        optimizer_config = {"lr": lr}

        net = model()

        residual = SymbolicOperator(
            expressions=[f],
            input_vars=input_labels,
            auxiliary_expressions={"periodic_u": g_u, "periodic_du": g_ux},
            constants={"mu": 1e-4, "alpha": 5, "beta": -5},
            output_vars=output_labels,
            function=net,
            engine="torch",
            device="gpu",
        )

        # It prints a summary of the network features
        net.summary()

        for optimizer_str in ["adam", "bbi"]:
            optimizer = Optimizer(
                optimizer_str,
                params=optimizer_config,
                lr_decay_scheduler_params={
                    "name": "ExponentialLR",
                    "gamma": 0.9,
                    "decay_frequency": 5_000,
                },
                shuffle=False,
                summary_writer=True,
            )

            params = {
                "residual": residual,
                "initial_input": data_boundary_t0,
                "initial_state": u_init,
                "boundary_input": {
                    "periodic_u": [data_boundary_xL, data_boundary_x0],
                    "periodic_du": [data_boundary_xL, data_boundary_x0],
                },
                "boundary_penalties": [1, 1],
                "initial_penalty": 100,
            }

            optimizer.fit(
                op=net,
                input_data=data,
                n_epochs=n_epochs,
                loss="pirmse",
                params=params,
                device="gpu",
            )
