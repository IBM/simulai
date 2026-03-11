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

from simulai.optimization import Optimizer
from simulai.regression import DenseNetwork
from simulai.residuals import SymbolicOperator


class TestWaveEquationPINN:
    def __init__(self):
        pass

    def test_wave_equation(self):
        """Solve the 1D wave equation using a PINN.

        PDE: d^2u/dt^2 = c^2 * d^2u/dx^2

        Domain: x in [0, 1], t in [0, 1]
        Initial condition: u(x, 0) = sin(pi * x)
        Initial velocity:  du/dt(x, 0) = 0
        Boundary conditions: u(0, t) = 0,  u(1, t) = 0

        Analytical solution: u(x, t) = sin(pi * x) * cos(pi * c * t)
        """

        # Wave equation: d^2u/dt^2 - c^2 * d^2u/dx^2 = 0
        # Using c^2 = 1 (wave speed c = 1)
        f = "D(D(u, t), t) - c_sq * D(D(u, x), x)"

        g_u = "u"
        g_l = "u"

        input_labels = ["x", "t"]
        output_labels = ["u"]

        n_inputs = len(input_labels)
        n_outputs = len(output_labels)

        # Some fixed values
        X_DIM = 100
        T_DIM = 100

        L = 1
        x_0 = 0
        T = 1
        c = 1.0  # wave speed

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

        # Initial condition: u(x, 0) = sin(pi * x)
        data_init = np.linspace(*x_interval, X_DIM)
        u_init = np.sin(np.pi * data_init)[:, None]

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
        plt.scatter(*np.split(data, 2, axis=1), s=1, label="Collocation")
        plt.scatter(*np.split(data_boundary_x0, 2, axis=1), s=5, label="BC x=0")
        plt.scatter(*np.split(data_boundary_xL, 2, axis=1), s=5, label="BC x=L")
        plt.scatter(*np.split(data_boundary_t0, 2, axis=1), s=5, label="IC t=0")
        plt.xlabel("x")
        plt.ylabel("t")
        plt.legend()
        plt.title("Training mesh")
        plt.savefig("wave_equation_mesh.png")
        plt.close()

        n_epochs = 20_000  # Maximum number of iterations for ADAM
        lr = 1e-3  # Initial learning rate for the ADAM algorithm

        # Configuration for the fully-connected network
        config = {
            "layers_units": [50, 50, 50, 50],
            "activations": "tanh",
            "input_size": n_inputs,
            "output_size": n_outputs,
            "name": "wave_net",
        }

        optimizer_config = {"lr": lr}

        # Instantiating and training the surrogate model
        net = DenseNetwork(**config)

        residual = SymbolicOperator(
            expressions=[f],
            input_vars=input_labels,
            auxiliary_expressions={"upper": g_l, "lower": g_u},
            constants={"c_sq": c**2},
            output_vars=output_labels,
            function=net,
            engine="torch",
        )

        # It prints a summary of the network features
        net.summary()

        optimizer = Optimizer("adam", params=optimizer_config)

        params = {
            "residual": residual,
            "initial_input": data_boundary_t0,
            "initial_state": u_init,
            "boundary_input": {"upper": data_boundary_xL, "lower": data_boundary_x0},
            "boundary_penalties": [1, 1],
            "initial_penalty": 10,
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

        # Analytical solution for comparison
        U_exact = np.sin(np.pi * X_f) * np.cos(np.pi * c * T_f)

        # Relative L2 error
        l2_error = np.linalg.norm(U_f - U_exact) / np.linalg.norm(U_exact)
        print(f"\nRelative L2 error: {l2_error:.6e}")

        # Plot approximation
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        ax = axes[0]
        ax.set_aspect("auto")
        gf = ax.pcolormesh(X_f, T_f, U_f, cmap="jet", vmin=-1, vmax=1)
        fig.colorbar(gf, ax=ax)
        ax.set_xlabel("x")
        ax.set_ylabel("t")
        ax.set_title("PINN Approximation")

        ax = axes[1]
        ax.set_aspect("auto")
        gf = ax.pcolormesh(X_f, T_f, U_exact, cmap="jet", vmin=-1, vmax=1)
        fig.colorbar(gf, ax=ax)
        ax.set_xlabel("x")
        ax.set_ylabel("t")
        ax.set_title("Analytical Solution")

        ax = axes[2]
        ax.set_aspect("auto")
        gf = ax.pcolormesh(X_f, T_f, np.abs(U_f - U_exact), cmap="hot")
        fig.colorbar(gf, ax=ax)
        ax.set_xlabel("x")
        ax.set_ylabel("t")
        ax.set_title(f"Absolute Error (L2 rel = {l2_error:.4e})")

        plt.tight_layout()
        plt.savefig("wave_equation_result.png", dpi=150)
        plt.show()


if __name__ == "__main__":
    TestWaveEquationPINN().test_wave_equation()
