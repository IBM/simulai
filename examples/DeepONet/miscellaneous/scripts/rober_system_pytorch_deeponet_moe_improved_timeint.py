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

import copy
import os
from typing import List, Union
from unittest import TestCase

import matplotlib.pyplot as plt
import numpy as np

# In order to execute this script, it is necessary to
# set the environment variable engine as "pytorch" before initializing
# simulai
os.environ["engine"] = "pytorch"

from simulai.models import ImprovedDeepONet as DeepONet
from simulai.models import MoEPool
from simulai.optimization import Optimizer
from simulai.regression import SLFNN, ConvexDenseNetwork
from simulai.residuals import SymbolicOperator
from simulai.templates import NetworkTemplate


def project_to_interval(interval, data):
    return interval[1] * (data - data.min()) / (data.max() - data.min()) + interval[0]


class ConvexMoEPool(MoEPool):
    def __init__(
        self,
        experts_list: List[NetworkTemplate],
        input_size: int = None,
        devices: Union[list, str] = None,
        hidden_size: int = None,
    ) -> None:
        super(ConvexMoEPool, self).__init__(
            experts_list=experts_list, input_size=input_size, devices=devices
        )

        self.hidden_size = hidden_size


# This is a very basic script for exploring the PDE solving via
# feedforward fully-connected neural networks
class TestROBER(TestCase):
    def setUp(self) -> None:
        pass

    def test_rober_torch(self):
        Q = 1_000
        N = int(5e4)

        k1 = 0.04
        k2 = 3e4
        k3 = 1e7

        initial_state_test = np.array([1, 0, 0])

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

        branch_input_test = np.tile(initial_state_test[None, :], Q)
        trunk_input_test = U_t

        initial_states = U_s

        input_labels = ["t"]
        output_labels = ["s1", "s2", "s3"]

        n_inputs = len(input_labels)
        n_outputs = len(output_labels)

        lambda_1 = 0.0  # Penalty for the L¹ regularization (Lasso)
        lambda_2 = 0.0  # Penalty factor for the L² regularization
        n_epochs = 5_000  # Maximum number of iterations for ADAM
        lr = 1e-3  # Initial learning rate for the ADAM algorithm
        n_latent = 100

        # Configuration for the fully-connected trunk network
        trunk_config = {
            "layers_units": 7 * [100],  # Hidden layers
            "activations": "tanh",
            "input_size": 1,
            "output_size": n_latent * n_outputs,
            "name": "trunk_net",
        }

        # Configuration for the fully-connected branch network
        branch_config = {
            "layers_units": 7 * [100],  # Hidden layers
            "activations": "tanh",
            "input_size": n_outputs,
            "output_size": n_latent * n_outputs,
            "name": "branch_net",
        }

        # Instantiating and training the surrogate model
        trunk_net = ConvexDenseNetwork(**trunk_config)

        branch_net_0 = ConvexDenseNetwork(**branch_config)
        branch_net_1 = ConvexDenseNetwork(**branch_config)
        branch_net_2 = ConvexDenseNetwork(**branch_config)
        branch_net_3 = ConvexDenseNetwork(**branch_config)

        encoder_trunk = SLFNN(input_size=1, output_size=100, activation="tanh")
        encoder_branch = SLFNN(input_size=3, output_size=100, activation="tanh")

        branch_net = ConvexMoEPool(
            experts_list=[branch_net_0, branch_net_1, branch_net_2, branch_net_3],
            input_size=3,
            hidden_size=100,
        )

        optimizer_config = {"lr": lr}

        # Maximum derivative magnitudes to be used as loss weights
        penalties = [1, 1e6, 1]
        batch_size = 1_000

        # It prints a summary of the network features
        trunk_net.summary()

        input_data = {
            "input_branch": branch_input_train,
            "input_trunk": trunk_input_train,
        }

        rober_net = DeepONet(
            trunk_network=trunk_net,
            branch_network=branch_net,
            encoder_trunk=encoder_trunk,
            encoder_branch=encoder_branch,
            var_dim=n_outputs,
            rescale_factors=np.array([1e-1, 1e-5, 1e-1]),
            model_id="rober_net",
        )

        residual = SymbolicOperator(
            expressions=[f_s1, f_s2, f_s3],
            input_vars=input_labels,
            output_vars=output_labels,
            function=rober_net,
            inputs_key="input_trunk",
            constants={"k1": k1, "k2": k2, "k3": k3},
            engine="torch",
        )

        optimizer = Optimizer(
            "adam",
            params=optimizer_config,
            lr_decay_scheduler_params={
                "name": "ExponentialLR",
                "gamma": 0.9,
                "decay_frequency": 5_0,
            },
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
            batch_size=batch_size,
        )

        approximated_data = rober_net.eval(
            trunk_data=trunk_input_test, branch_data=branch_input_test
        )

        for ii in range(n_inputs):
            plt.plot(approximated_data[:, ii], label="Approximated")
            plt.legend()
            plt.savefig(f"rober_deeponet_time_int_{ii}.png")
            plt.show()
