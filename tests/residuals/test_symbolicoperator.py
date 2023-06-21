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

import numpy as np
from tests.config import configure_dtype
torch = configure_dtype()

from simulai.residuals import SymbolicOperator
from simulai.tokens import Dot, Gp

def model(n_inputs: int = 1, n_outputs: int = 1):
    from simulai.regression import DenseNetwork

    # Configuration for the fully-connected network
    config = {
        "layers_units": [128, 128, 128, 128],
        "activations": "sin",
        "input_size": n_inputs,
        "output_size": n_outputs,
        "name": "allen_cahn_net",
    }

    # Instantiating and training the surrogate model
    net = DenseNetwork(**config)

    return net

def model_operator():

    import numpy as np

    from simulai.models import DeepONet, ImprovedDenseNetwork
    from simulai.regression import SLFNN, ConvexDenseNetwork

    n_latent = 100
    n_inputs_b = 5
    n_inputs_t = 1
    n_outputs = 1

    # Configuration for the fully-connected trunk network
    trunk_config = {
        "layers_units": 7 * [100],  # Hidden layers
        "activations": "tanh",
        "input_size": n_inputs_t,
        "output_size": n_latent * n_outputs,
        "name": "trunk_net",
    }

    # Configuration for the fully-connected branch network
    branch_config = {
        "layers_units": 7 * [100],  # Hidden layers
        "activations": "tanh",
        "input_size": n_inputs_b,
        "output_size": n_latent * n_outputs,
        "name": "branch_net",
    }

    encoder_u_trunk = SLFNN(input_size=n_inputs_t, output_size=100, activation="tanh")
    encoder_v_trunk = SLFNN(input_size=n_inputs_t, output_size=100, activation="tanh")
    encoder_u_branch = SLFNN(input_size=n_inputs_b, output_size=100, activation="tanh")
    encoder_v_branch = SLFNN(input_size=n_inputs_b, output_size=100, activation="tanh")

    # Instantiating and training the surrogate model
    trunk_net_dense = ConvexDenseNetwork(**trunk_config)
    branch_net_dense = ConvexDenseNetwork(**branch_config)

    trunk_net = ImprovedDenseNetwork(
        network=trunk_net_dense, encoder_u=encoder_u_trunk, encoder_v=encoder_v_trunk
    )

    branch_net = ImprovedDenseNetwork(
        network=branch_net_dense, encoder_u=encoder_u_branch, encoder_v=encoder_v_branch
    )

    # It prints a summary of the network features
    trunk_net.summary()
    branch_net.summary()

    deep_o_net = DeepONet(
        trunk_network=trunk_net,
        branch_network=branch_net,
        var_dim=n_outputs,
        rescale_factors=np.array([1]),
        devices="gpu",
        model_id="flame_net",
    )

    return deep_o_net


class TestSymbolicOperator(TestCase):
    def setUp(self) -> None:
        pass

    @staticmethod
    def k1(t: torch.Tensor) -> torch.Tensor:
        return torch.cos(torch.sqrt(t) + 1) / 2

    @staticmethod
    def k2(t: torch.Tensor) -> torch.Tensor:
        return 3 * torch.exp(torch.sin(t))

    def test_symbolic_external_functions(self):
        f = f"D(u, t) - alpha*k2(t)*k1(t)"

        input_labels = ["t"]
        output_labels = ["u"]

        T = 1
        t_interval = [0, T]

        net = model(n_inputs=len(input_labels), n_outputs=len(output_labels))

        residual = SymbolicOperator(
            expressions=[f],
            input_vars=input_labels,
            constants={"alpha": 5},
            external_functions={"k1": self.k1, "k2": self.k2},
            output_vars=output_labels,
            function=net,
            engine="torch",
        )

        t = np.linspace(*t_interval)[:, None]

        assert all([isinstance(item, torch.Tensor) for item in residual(t)])

    def test_symbolic_buitin_functions(self):

        f = f"D(u, t) - Dot(a, Gp(t, t, 4))"

        input_labels = ["t", "a"]
        output_labels = ["u"]

        T = 1
        t_interval = [0, T]
        N = 100
        a = torch.from_numpy(np.random.rand(100, 5).astype("float32"))

        net = model_operator()

        residual = SymbolicOperator(
            expressions=[f],
            input_vars=input_labels,
            external_functions={"Dot": Dot, "Gp": Gp},
            output_vars=output_labels,
            inputs_key="input_trunk|input_branch[0,4]",
            function=net,
            engine="torch",
        )

        t = torch.from_numpy(np.linspace(*t_interval, N)[:, None].astype("float32"))
        input_data = {"input_trunk": t, "input_branch": a}

        assert all([isinstance(item, torch.Tensor) for item in residual(input_data)])

    def test_symbolic_operator_ode(self):
        for token in ["sin", "cos", "sqrt"]:
            f = f"D(u, t) - alpha*{token}(u)"

            input_labels = ["t"]
            output_labels = ["u"]

            T = 1
            t_interval = [0, T]

            net = model(n_inputs=len(input_labels), n_outputs=len(output_labels))

            residual = SymbolicOperator(
                expressions=[f],
                input_vars=input_labels,
                constants={"alpha": 5},
                output_vars=output_labels,
                function=net,
                engine="torch",
            )

            t = np.linspace(*t_interval)[:, None]

            assert all([isinstance(item, torch.Tensor) for item in residual(t)])

    def test_symbolic_operator_diff_operators(self):
        for operator in ["L", "Div"]:
            f = f"D(u, x) - alpha*{operator}(u, (x, y))"

            input_labels = ["x", "y"]
            output_labels = ["u"]

            L_x = 1
            L_y = 1
            N_x = 100
            N_y = 100
            dx = L_x / N_x
            dy = L_y / N_y

            grid = np.mgrid[0:L_x:dx, 0:L_y:dy]

            data = np.hstack([grid[1].flatten()[:, None], grid[0].flatten()[:, None]])

            net = model(n_inputs=len(input_labels), n_outputs=len(output_labels))

            residual = SymbolicOperator(
                expressions=[f],
                input_vars=input_labels,
                constants={"alpha": 5},
                output_vars=output_labels,
                function=net,
                engine="torch",
            )

            assert all([isinstance(item, torch.Tensor) for item in residual(data)])

    def test_symbolic_operator_1d_pde(self):
        # Allen-Cahn equation
        f_0 = "D(u, t) - mu*D(D(u, x), x) + alpha*(u**3) + beta*u"
        # Invented 1
        f_1 = "D(D(u, t),t) - mu*D(D(u, x), x) + alpha*(u**3) + beta*u"
        # Invented 2
        f_2 = "D(D(u, t), x) - D(mu*D(D(u, x), x), t) + alpha*(u**3) + beta*u"

        g_u = "u"
        g_ux = "D(u, x)"

        input_labels = ["x", "t"]
        output_labels = ["u"]

        net = model(n_inputs=len(input_labels), n_outputs=len(output_labels))

        # Generating the training grid
        L = 1
        x_0 = -1
        T = 1
        X_DIM = 100
        T_DIM = 200
        x_interval = [x_0, L]
        t_interval = [0, T]

        # Regular grid
        x_0, x_L = x_interval
        t_0, t_L = t_interval
        dx = (x_L - x_0) / X_DIM
        dt = (t_L - t_0) / T_DIM

        grid = np.mgrid[t_0 + dt : t_L + dt : dt, x_0:x_L:dx]

        data = np.hstack([grid[1].flatten()[:, None], grid[0].flatten()[:, None]])

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

        for f in [f_0, f_1, f_2]:
            residual = SymbolicOperator(
                expressions=[f],
                input_vars=input_labels,
                auxiliary_expressions={"periodic_u": g_u, "periodic_du": g_ux},
                constants={"mu": 1e-4, "alpha": 5, "beta": -5},
                output_vars=output_labels,
                function=net,
                engine="torch",
            )

            # Verifying if all the PDE expressions are callable
            assert all([callable(i) for i in residual.f_expressions])

            # Verifying if all the boundary expressions are callable
            try:
                assert all([callable(i) for i in residual.g_expressions.values()])
            except:
                assert all([callable(i) for i in residual.g_expressions])

            # Testing to evaluate a list of PDE expressions
            assert all([isinstance(item, torch.Tensor) for item in residual(data)])

            # Testing to evaluate auxiliary (boundary) expressions
            assert isinstance(
                residual.eval_expression(
                    "periodic_u", [data_boundary_xL, data_boundary_x0]
                ),
                torch.Tensor,
            )
            assert isinstance(
                residual.eval_expression(
                    "periodic_u", [data_boundary_xL, data_boundary_x0]
                ),
                torch.Tensor,
            )
