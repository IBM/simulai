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

from simulai.optimization import Optimizer
from simulai.residuals import SymbolicOperator


class TestDeepOperatorROBERPINN(TestCase):
    def setUp(self) -> None:
        pass

    def test_rober_deeponet_pinn(self):
        Q = 1_00
        N = int(5)

        k1 = 0.04
        k2 = 3e7
        k3 = 1e4

        t_intv = [0, 1]
        s_intv = np.stack([[0, 0, 0], [1, 1e-4, 1]], axis=0)

        # The expression we aim at minimizing
        f_s1 = "D(s1, t) + k1*s1 - k3*s2*s3"
        f_s2 = "D(s2, t) - k1*s1 + k2*(s2**2) + k3*s2*s3"
        f_s3 = "D(s3, t) - k2*(s2**2)"

        U_t = np.random.uniform(low=t_intv[0], high=t_intv[1], size=Q)
        U_s = np.random.uniform(low=s_intv[0], high=s_intv[1], size=(N, 3))

        branch_input_train = np.tile(U_s[:, None, :], (1, Q, 1)).reshape(N * Q, -1)
        trunk_input_train = np.tile(U_t[:, None], (N, 1))

        initial_states = U_s

        input_labels = ["t"]
        output_labels = ["s1", "s2", "s3"]

        lambda_1 = 0.0  # Penalty for the L¹ regularization (Lasso)
        lambda_2 = 0.0  # Penalty factor for the L² regularization
        n_epochs = 5  # Maximum number of iterations for ADAM
        lr = 1e-3  # Initial learning rate for the ADAM algorithm

        def model():
            import numpy as np

            from simulai.models import ImprovedDeepONet
            from simulai.regression import SLFNN, ConvexDenseNetwork

            n_latent = 100
            n_inputs_b = 3
            n_inputs_t = 1
            n_outputs = 3

            # Configuration for the fully-connected trunk network
            trunk_config = {
                "layers_units": 3 * [20],  # Hidden layers
                "activations": "tanh",
                "input_size": n_inputs_t,
                "output_size": n_latent * n_outputs,
                "name": "trunk_net",
            }

            # Configuration for the fully-connected branch network
            branch_config = {
                "layers_units": 3 * [20],  # Hidden layers
                "activations": "tanh",
                "input_size": n_inputs_b,
                "output_size": n_latent * n_outputs,
                "name": "branch_net",
            }

            # Instantiating and training the surrogate model
            trunk_net = ConvexDenseNetwork(**trunk_config)
            branch_net = ConvexDenseNetwork(**branch_config)

            encoder_trunk = SLFNN(
                input_size=n_inputs_t, output_size=20, activation="tanh"
            )
            encoder_branch = SLFNN(
                input_size=n_inputs_b, output_size=20, activation="tanh"
            )

            # It prints a summary of the network features
            trunk_net.summary()
            branch_net.summary()

            rober_net = ImprovedDeepONet(
                trunk_network=trunk_net,
                branch_network=branch_net,
                encoder_trunk=encoder_trunk,
                encoder_branch=encoder_branch,
                var_dim=n_outputs,
                rescale_factors=np.array([1, 1e-4, 1]),
                devices="gpu",
                model_id="rober_net",
            )

            return rober_net

        rober_net = model()

        residual = SymbolicOperator(
            expressions=[f_s1, f_s2, f_s3],
            input_vars=input_labels,
            output_vars=output_labels,
            function=rober_net,
            inputs_key="input_trunk",
            constants={"k1": k1, "k2": k2, "k3": k3},
            device="gpu",
            engine="torch",
        )

        # Maximum derivative magnitudes to be used as loss weights
        penalties = [1, 1e6, 1]

        optimizer_config = {"lr": lr}

        input_data = {
            "input_branch": branch_input_train,
            "input_trunk": trunk_input_train,
        }

        optimizer = Optimizer(
            "adam",
            params=optimizer_config,
            lr_decay_scheduler_params={
                "name": "ExponentialLR",
                "gamma": 0.9,
                "decay_frequency": 5_000,
            },
            summary_writer=True,
        )

        params = {
            "lambda_1": lambda_1,
            "lambda_2": lambda_2,
            "residual": residual,
            "initial_input": {
                "input_trunk": np.zeros((N, 1)),
                "input_branch": initial_states,
            },
            "initial_state": initial_states,
            "weights_residual": [1, 1, 1],
            "weights": penalties,
        }

        optimizer.fit(
            op=rober_net,
            input_data=input_data,
            n_epochs=n_epochs,
            loss="opirmse",
            params=params,
            device="gpu",
        )
