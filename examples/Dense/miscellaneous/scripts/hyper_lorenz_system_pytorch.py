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
import torch

from examples.utils.lorenz_solver import lorenz_solver
from simulai.metrics import L2Norm
from simulai.optimization import Optimizer
from simulai.regression import DenseNetwork
from simulai.templates import HyperTrainTemplate
from simulai.workflows import ParamHyperOpt

# In order to execute this script, it is necessary to
# set the environment variable engine as "pytorch" before initializing
# simulai


# Customized class to be used in the hyper-search
class HyperTrainDense(HyperTrainTemplate):
    def __init__(
        self, trial_config: dict = None, set_type="hard", path_to_model: str = None
    ):
        super().__init__(trial_config=trial_config, set_type=set_type)

        self.path_to_model = path_to_model

    def _set_model(self):
        config = {
            "layers_units": self.trial_config.get("layers_units"),
            "activations": self.trial_config.get("activations"),
            "input_size": self.trial_config.get("input_size"),
            "output_size": self.trial_config.get("output_size"),
            "name": self.trial_config.get("name"),
        }

        self.model = DenseNetwork(**config)

    def _set_optimizer(self):
        optimizer = self.trial_config.get("optimizer")

        optimizer_config = {"lr": self.trial_config.get("lr")}

        # The trial values for the regularization terms are exponents
        self.params = {
            "lambda_1": 10 ** self.trial_config.get("lambda_1_exp"),
            "lambda_2": 10 ** self.trial_config.get("lambda_2_exp"),
            "weights": self.trial_config.get("weights"),
        }

        self.loss = self.trial_config.get("loss")
        self.n_epochs = self.trial_config.get("n_epochs")

        self.optimizer = Optimizer(optimizer=optimizer, params=optimizer_config)

    def fit(self, input_train_data=None, target_train_data=None):
        if self.path_to_model:
            self.model.load_state_dict(torch.load(self.path_to_model))
        else:
            pass

        self.optimizer.fit(
            self.model,
            input_data=input_train_data,
            target_data=target_train_data,
            n_epochs=self.n_epochs,
            loss=self.loss,
            params=self.params,
        )


def objective(model, input_validation_data=None, target_validation_data=None):
    approximated_data = model.eval(input_data=input_validation_data)

    l2_norm = L2Norm()

    error = 100 * l2_norm(
        data=approximated_data,
        reference_data=target_validation_data,
        relative_norm=True,
    )

    return error


# This is a very basic script for exploring the PDE solving via
# feedforward fully-connected neural networks
class TestLorenzTorch(TestCase):
    def test_lorenz_torch(self):
        dt = 0.005
        T_max = 50
        rho = 28
        beta = 8 / 3
        beta_str = "8/3"
        sigma = 10

        initial_state = np.array([1, 2, 3])[None, :]
        lorenz_data, derivative_lorenz_data = lorenz_solver(
            rho=rho,
            dt=dt,
            T=T_max,
            sigma=sigma,
            initial_state=initial_state,
            beta=beta,
            beta_str=beta_str,
            data_path="on_memory",
        )

        # The fraction of data used for training the model.
        train_fraction = 0.8
        validation_fraction = 0.1

        # Number of attempts for Optuna
        n_trials = 3

        # Dumping the CSV data containing the simulation data
        # to a Pandas dataframe.
        input_data = lorenz_data
        output_data = derivative_lorenz_data

        # Choosing the number of training and testing samples
        n_samples = input_data.shape[0]
        train_samples = int(train_fraction * n_samples)
        validation_samples = int(validation_fraction * n_samples)
        test_samples = n_samples - train_samples - validation_samples

        input_labels = ["x", "y", "z"]
        output_labels = ["x_dot", "y_dot", "z_dot"]

        time = np.arange(0, n_samples, 1) * dt

        # Training dataset
        train_input_data = input_data[:train_samples]
        train_output_data = output_data[:train_samples]
        time_train = time[:train_samples]

        # Validation dataset
        validation_input_data = input_data[
            train_samples : train_samples + validation_samples
        ]
        validation_output_data = output_data[
            train_samples : train_samples + validation_samples
        ]
        time_validation = time[train_samples : train_samples + validation_samples]

        # Testing dataset
        test_input_data = input_data[train_samples + validation_samples :]
        test_output_data = output_data[train_samples + validation_samples :]
        time_test = time[train_samples + validation_samples :]

        n_inputs = len(input_labels)
        n_outputs = len(output_labels)

        params_intervals = {"lambda_1_exp": [-5, -3], "lambda_2_exp": [-5, -3]}
        params_suggestions = {"lambda_1_exp": "float", "lambda_2_exp": "float"}

        maximum_values = (1 / np.linalg.norm(train_output_data, 2, axis=0)).tolist()

        # Parameters left fixed
        others_params = {
            "layers_units": [50, 50, 50],  # Hidden layers
            "activations": "elu",
            "input_size": n_inputs,
            "output_size": n_outputs,
            "name": "lorenz_net",
            "lr": 1e-3,
            "n_epochs": int(2e3),
            "loss": "wrmse",
            "optimizer": "adam",
            "weights": maximum_values,
            "batch_size": 100,
        }

        hyper_search = ParamHyperOpt(
            params_intervals=params_intervals,
            params_suggestions=params_suggestions,
            name="lorenz_hyper_search",
            direction="minimize",
            trainer_template=HyperTrainDense,
            objective_function=objective,
            others_params=others_params,
        )

        hyper_search.set_data(
            input_train_data=train_input_data,
            target_train_data=train_output_data,
            input_validation_data=validation_input_data,
            target_validation_data=validation_output_data,
            input_test_data=test_input_data,
            target_test_data=test_output_data,
        )

        hyper_search.optimize(n_trials=n_trials)

    def test_lorenz_torch_predefined(self):
        dt = 0.005
        T_max = 50
        rho = 28
        beta = 8 / 3
        beta_str = "8/3"
        sigma = 10

        initial_state = np.array([1, 2, 3])[None, :]
        lorenz_data, derivative_lorenz_data = lorenz_solver(
            rho=rho,
            dt=dt,
            T=T_max,
            sigma=sigma,
            initial_state=initial_state,
            beta=beta,
            beta_str=beta_str,
            data_path="on_memory",
        )

        # The fraction of data used for training the model.
        train_fraction = 0.8
        validation_fraction = 0.1

        # Number of attempts for Optuna
        n_trials = 3

        # Dumping the CSV data containing the simulation data
        # to a Pandas dataframe.
        input_data = lorenz_data
        output_data = derivative_lorenz_data

        # Choosing the number of training and testing samples
        n_samples = input_data.shape[0]
        train_samples = int(train_fraction * n_samples)
        validation_samples = int(validation_fraction * n_samples)
        test_samples = n_samples - train_samples - validation_samples

        input_labels = ["x", "y", "z"]
        output_labels = ["x_dot", "y_dot", "z_dot"]

        time = np.arange(0, n_samples, 1) * dt

        # Training dataset
        train_input_data = input_data[:train_samples]
        train_output_data = output_data[:train_samples]
        time_train = time[:train_samples]

        # Validation dataset
        validation_input_data = input_data[
            train_samples : train_samples + validation_samples
        ]
        validation_output_data = output_data[
            train_samples : train_samples + validation_samples
        ]
        time_validation = time[train_samples : train_samples + validation_samples]

        # Testing dataset
        test_input_data = input_data[train_samples + validation_samples :]
        test_output_data = output_data[train_samples + validation_samples :]
        time_test = time[train_samples + validation_samples :]

        n_inputs = len(input_labels)
        n_outputs = len(output_labels)

        params_intervals = {"lambda_1_exp": [-5, -3], "lambda_2_exp": [-5, -3]}
        params_suggestions = {"lambda_1_exp": "float", "lambda_2_exp": "float"}

        maximum_values = (1 / np.linalg.norm(train_output_data, 2, axis=0)).tolist()

        # Parameters left fixed
        others_params = {
            "layers_units": [50, 50, 50],  # Hidden layers
            "activations": "elu",
            "input_size": n_inputs,
            "output_size": n_outputs,
            "name": "lorenz_net",
            "lr": 1e-3,
            "n_epochs": int(2e3),
            "loss": "wrmse",
            "optimizer": "adam",
            "weights": maximum_values,
        }

        zero_trial = {"lambda_1_exp": -3, "lambda_2_exp": -5}
        zero_trial.update(others_params)

        hyperdense_train = HyperTrainDense(trial_config=zero_trial, set_type="soft")
        hyperdense_train.save_model(path="lorenz_weights.pth")

        hyper_search = ParamHyperOpt(
            params_intervals=params_intervals,
            params_suggestions=params_suggestions,
            name="lorenz_hyper_search",
            direction="minimize",
            trainer_template=hyperdense_train,
            objective_function=objective,
            weights_initialization="lorenz_weights.pth",
            others_params=others_params,
        )

        hyper_search.set_data(
            input_train_data=train_input_data,
            target_train_data=train_output_data,
            input_validation_data=validation_input_data,
            target_validation_data=validation_output_data,
            input_test_data=test_input_data,
            target_test_data=test_output_data,
        )

        hyper_search.optimize(n_trials=n_trials)

        lorenz_net = hyper_search.retrain_best_trial()

        approximated_data = lorenz_net.eval(input_data=test_input_data)

        l2_norm = L2Norm()

        error = 100 * l2_norm(
            data=approximated_data, reference_data=test_output_data, relative_norm=True
        )

        for ii in range(n_inputs):
            plt.plot(approximated_data[:, ii], label="Approximated")
            plt.plot(test_output_data[:, ii], label="Exact")
            plt.legend()
            plt.show()

        print(f"Approximation error for the derivatives: {error} %")
