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

import matplotlib.pyplot as plt
import numpy as np
import torch

from simulai.optimization import Optimizer
from simulai.regression import DenseNetwork
from simulai.residuals import SymbolicOperator


class TestBurgersPINN:
    def __init__(self):
        pass

    def test_burgers(self):
        # Our PDE
        # Inviscid Burgers equation

        f = "D(u, t) + (1 / 2) * D(u ** 2, x)"
        g_u = "u"
        g_l = "u"

        input_labels = ["x", "t"]
        output_labels = ["u"]

        n_inputs = len(input_labels)
        n_outputs = len(output_labels)

        # Some fixed values
        X_DIM = 100
        T_DIM = 200

        L = 1
        x_0 = -1
        T = 1.5
        u0 = 1

        # Generating the training grid

        x_interval = [x_0, L]
        t_interval = [0, T]

        intervals = [x_interval, t_interval]

        intv_array = np.vstack(intervals).T

        # Regular grid
        x_0, x_L = x_interval
        t_0, t_L = t_interval
        dx = (x_L - x_0) / X_DIM
        dt = (t_L - t_0) / T_DIM

        grid = np.mgrid[t_0 + dt : t_L + dt : dt, x_0:x_L:dx]

        data = np.hstack([grid[1].flatten()[:, None], grid[0].flatten()[:, None]])

        data_init = np.linspace(*x_interval, X_DIM)
        u_init = u0 * np.sin(2 * np.pi * data_init / L)[:, None]

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

        # Visualizing the training mesh
        plt.scatter(*np.split(data, 2, axis=1))
        plt.scatter(*np.split(data_boundary_x0, 2, axis=1))
        plt.scatter(*np.split(data_boundary_xL, 2, axis=1))
        plt.scatter(*np.split(data_boundary_t0, 2, axis=1))

        plt.show()
        plt.close()

        n_epochs = 2_000  # Maximum number of iterations for ADAM
        lr = 1e-3  # Initial learning rate for the ADAM algorithm

        # Configuration for the fully-connected network
        config = {
            "layers_units": [50, 50, 50],
            "activations": "tanh",
            "input_size": n_inputs,
            "output_size": n_outputs,
            "name": "burgers_net",
        }

        # optimizer_config = {'lr' : 1e-2, 'threshold0' : 20, 'threshold' : 100, 'v0' : 1e-4, 'n_fixed_bounces' : 4, 'consEn' : True}
        optimizer_config = {"lr": lr}

        # Instantiating and training the surrogate model
        net = DenseNetwork(**config)

        residual = SymbolicOperator(
            expressions=[f],
            input_vars=input_labels,
            auxiliary_expressions={"upper": g_l, "lower": g_u},
            output_vars=output_labels,
            function=net,
            engine="torch",
        )

        # It prints a summary of the network features
        net.summary()

        # optimizer = Optimizer('bbi', params=optimizer_config)
        optimizer = Optimizer("adam", params=optimizer_config)

        params = {
            "residual": residual,
            "initial_input": data_boundary_t0,
            "initial_state": u_init,
            "boundary_input": {"upper": data_boundary_xL, "lower": data_boundary_x0},
            "boundary_penalties": [1, 1],
            "initial_penalty": 10,
            "causality_preserving": True,
            "grid_shape": (T_DIM, X_DIM),
            "causality_parameter": 0.10,
        }

        optimizer.fit(
            op=net, input_data=data, n_epochs=n_epochs, loss="pirmse", params=params
        )

        # Evaluation and post-processing
        X_DIM_F = 5 * X_DIM
        T_DIM_F = 5 * T_DIM

        x_f = np.linspace(*x_interval, X_DIM_F)
        t_f = np.linspace(*t_interval, T_DIM_F)

        T_f, X_f = np.meshgrid(t_f, x_f, indexing="ij")

        data_f = np.hstack([X_f.flatten()[:, None], T_f.flatten()[:, None]])

        # Evaluation in training dataset
        approximated_data = net.eval(input_data=data_f)

        U_f = approximated_data.reshape(T_DIM_F, X_DIM_F)

        fig, ax = plt.subplots()
        ax.set_aspect("auto")
        gf = ax.pcolormesh(X_f, T_f, U_f, cmap="jet")
        fig.colorbar(gf)

        plt.show()
