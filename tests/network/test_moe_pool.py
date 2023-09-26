from unittest import TestCase

import numpy as np

from simulai.models import MoEPool
from simulai.optimization import Optimizer
from simulai.regression import DenseNetwork


class TestMoEPool(TestCase):
    def test_moe_pool(self):
        n_inputs_b = 20
        n_latent = 50
        n_outputs = 20

        config = {
            "layers_units": 7 * [100],  # Hidden layers
            "activations": "tanh",
            "input_size": n_inputs_b,
            "output_size": n_latent * n_outputs,
            "name": "branch_net",
        }

        experts_list = list()
        n_experts = 4

        for ex in range(n_experts):
            experts_list.append(DenseNetwork(**config))

        for device in ["cpu", "gpu", None]:

            net = MoEPool(experts_list=experts_list, input_size=n_inputs_b, devices=device)

            # Checking if the model is coretly placed when no device is
            # informed
            if not device:
                assert net.device == "cpu", ("When no device is provided it is expected the model"+
                                             f"being on cpu, but received {net.device}.")


            input_data = np.random.rand(1_000, n_inputs_b)

            estimated_output = net.eval(input_data=input_data)
            weights = net.gate(input_data=input_data)

            assert estimated_output.shape[1] == n_latent * n_outputs
            assert weights.shape[1] == n_experts

    def test_moe_pool_binary(self):
        n_inputs_b = 20
        n_latent = 50
        n_outputs = 20

        config = {
            "layers_units": 7 * [100],  # Hidden layers
            "activations": "tanh",
            "input_size": n_inputs_b,
            "output_size": n_latent * n_outputs,
            "name": "branch_net",
        }

        experts_list = list()
        n_experts = 4

        for ex in range(n_experts):
            experts_list.append(DenseNetwork(**config))

        input_data = np.random.rand(1_000, n_inputs_b)

        for device in ["cpu", "gpu", None]:

            net = MoEPool(
                experts_list=experts_list,
                input_size=n_inputs_b,
                binary_selection=True,
                devices=device,
            )

            # Checking if the model is coretly placed when no device is
            # informed
            if not device:
                assert net.device == "cpu", ("When no device is provided it is expected the model"+
                                             f"being on cpu, but received {net.device}.")


            weights = net.gate(input_data=input_data)
            weights = weights.numpy()

            assert np.array_equal(
                np.where(weights == 1, 0, weights), np.zeros_like(weights)
            )

    def test_moe_pool_optimization(self):
        lr = 1e-3
        optimizer_config = {"lr": lr}

        optimizer = Optimizer(
            "adam",
            params=optimizer_config,
            lr_decay_scheduler_params={
                "name": "ExponentialLR",
                "gamma": 0.9,
                "decay_frequency": 5_000,
            },
        )

        n_inputs_b = 20
        n_outputs = 10

        config = {
            "layers_units": 7 * [100],  # Hidden layers
            "activations": "tanh",
            "input_size": n_inputs_b,
            "output_size": n_outputs,
            "name": "branch_net",
        }

        experts_list = list()
        n_experts = 4
        n_epochs = 10

        for ex in range(n_experts):
            experts_list.append(DenseNetwork(**config))

        net = MoEPool(experts_list=experts_list, input_size=n_inputs_b, devices="gpu")

        input_data = np.random.rand(1_000, n_inputs_b)
        target_data = np.random.rand(1_000, n_outputs)

        params = {"lambda_1": 0.0, "lambda_2": 1e-6}

        optimizer.fit(
            op=net,
            input_data=input_data,
            target_data=target_data,
            params=params,
            n_epochs=n_epochs,
            loss="rmse",
            device="gpu",
        )

    def test_moe_pool_optimization_bce(self):
        lr = 1e-3
        optimizer_config = {"lr": lr}

        optimizer = Optimizer(
            "adam",
            params=optimizer_config,
            lr_decay_scheduler_params={
                "name": "ExponentialLR",
                "gamma": 0.9,
                "decay_frequency": 5_000,
            },
        )

        n_inputs_b = 20
        n_outputs = 10

        config = {
            "layers_units": 7 * [100],  # Hidden layers
            "activations": "tanh",
            "input_size": n_inputs_b,
            "output_size": n_outputs,
            "name": "branch_net",
        }

        experts_list = list()
        n_experts = 4
        n_epochs = 10

        for ex in range(n_experts):
            experts_list.append(DenseNetwork(**config))

        net = MoEPool(experts_list=experts_list, input_size=n_inputs_b, devices="gpu")

        input_data = np.random.rand(1_000, n_inputs_b)
        target_data = np.random.rand(1_000, n_outputs)

        params = {"lambda_1": 0.0, "lambda_2": 1e-6}

        optimizer.fit(
            op=net,
            input_data=input_data,
            target_data=target_data,
            params=params,
            n_epochs=n_epochs,
            loss="rmse",
            device="gpu",
        )
