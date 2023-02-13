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
from argparse import ArgumentParser

import numpy as np

from examples.utils.oscillator_solver import oscillator_solver_forcing
from simulai.metrics import L2Norm
from simulai.regression import EchoStateNetwork
from simulai.templates import HyperTrainTemplate
from simulai.workflows import ParamHyperOpt, StepwiseExtrapolation


# Hyperparameter search for  ESN+ModelPool using Optuna
class HyperModelPoolESN(HyperTrainTemplate):
    def __init__(
        self, trial_config: dict = None, set_type="hard", other_params: dict = None
    ):
        self.model = None

        self.others_params = other_params

        required_keys = [
            "sub_model_number_of_inputs",
            "global_matrix_constructor_str",
            "n_workers",
            "solver",
            "initial_state",
            "path_to_save",
            "tag",
            "id",
        ]

        self.sub_model_number_of_inputs = None
        self.global_matrix_constructor_str = None
        self.n_workers = None
        self.solver = None
        self.initial_state = None
        self.n_steps = None
        self.path_to_save = None
        self.tag = "model_"
        self.id = None

        for key in required_keys:
            assert (
                key in trial_config.keys()
            ), f"The required parameter {key} is not in others_params."

        for key, value in trial_config.items():
            setattr(self, key, value)

        super().__init__(trial_config=trial_config, set_type=set_type)

        self.model_id = self.tag + str(self.id)

    def _set_model(self):
        rc_config = {
            "reservoir_dim": self.trial_config["reservoir_dim"],
            "sparsity_level": self.trial_config["sparsity_level"]
            * self.trial_config["reservoir_dim"],
            "radius": self.trial_config["radius"],
            "sigma": self.trial_config["sigma"],
            "beta": 10 ** self.trial_config["beta_exp"],
        }

        extra_params = {
            "number_of_inputs": self.sub_model_number_of_inputs,
            "global_matrix_constructor_str": self.global_matrix_constructor_str,
            "solver": self.solver,
            "n_workers": self.n_workers,
        }

        rc_config.update(extra_params)

        self.model = EchoStateNetwork(**rc_config)

    def fit(self, input_train_data=None, target_train_data=None):
        self.model.fit(input_data=input_train_data, target_data=target_train_data)

        self.model.save_model(save_path=self.path_to_save, model_name=self.model_id)


class ObjectiveWrapper:
    def __init__(
        self, test_data=None, forcings_input=None, initial_state=None, n_steps=None
    ):
        self.test_data = test_data
        self.forcings_input = forcings_input
        self.initial_state = initial_state
        self.n_steps = n_steps

    def __call__(self, trainer_instance=None, objective_function=None):
        return objective_function(
            model=trainer_instance,
            initial_state=self.initial_state,
            test_data=self.test_data,
            forcings_input=self.forcings_input,
            n_steps=self.n_steps,
        )


def objective(
    model=None, initial_state=None, test_data=None, forcings_input=None, n_steps=None
):
    extrapolator = StepwiseExtrapolation(model=model.model, keys=["ESN_0"])

    l2_norm = L2Norm()

    estimated_data = extrapolator.predict(
        initial_state=initial_state, auxiliary_data=forcings_input, horizon=n_steps
    )

    error = 100 * l2_norm(
        data=estimated_data, reference_data=test_data, relative_norm=True
    )

    return error


def test_esn_nonlinear_forcing_numba(path_to_save: str = None):
    n_steps = 1000
    A = 1
    T = 50
    dt = T / n_steps

    if not os.path.isfile("forcings.npy"):
        forcings = A * np.random.rand(n_steps, 2)
        np.save("forcings.npy", forcings)
    else:
        forcings = np.load("forcings.npy")

    initial_state = np.array([2, 0])[None, :]

    n_field = 2  # number of field values to predict
    n_forcing = 2  # number of forcing terms
    sub_model_number_of_inputs = n_field + n_forcing  # size of the data

    train_fraction = 0.6
    validation_fraction = 0.3
    test_fraction = 0.3

    n_workers = 8
    n_trials = 5

    n_train = int(train_fraction * n_steps)  # size of time steps
    n_validation = int(validation_fraction * n_steps)
    n_test = int(test_fraction * n_steps)

    oscillator_data, _ = oscillator_solver_forcing(
        T, dt, initial_state, forcing=forcings
    )

    field_data = oscillator_data  # manufactured nonlinear oscillator data

    train_data = field_data[:n_train, :]
    validation_data = field_data[n_train : n_train + n_validation, :]
    test_data = field_data[n_train + n_validation :, :]

    input_data = train_data[:-1, :]
    target_data = train_data[1:, :]

    forcings_train_data = forcings[:n_train, :][:-1]

    forcings_validation_data = forcings[n_train : n_train + n_validation, :]

    forcings_test_data = forcings[n_train + n_validation :, :]

    initial_state_validation = train_data[-1:, :]
    initial_state_test = validation_data[-1:, :]

    params_intervals = {
        "reservoir_dim": (1000, 2000),
        "sparsity_level": (0.05, 0.1),
        "radius": (0.3, 0.7),
        "sigma": (0.3, 0.7),
        "beta_exp": (-5, -2),
    }

    params_suggestions = {
        "reservoir_dim": "int",
        "sparsity_level": "float",
        "radius": "float",
        "sigma": "float",
        "beta_exp": "int",
    }

    others_params = {
        "number_of_inputs": sub_model_number_of_inputs,
        "sub_model_number_of_inputs": sub_model_number_of_inputs,
        "global_matrix_constructor_str": "multiprocessing",
        "solver": "linear_system",
        "n_workers": n_workers,
        "initial_state": initial_state_validation,
        "n_steps": n_validation,
        "tag": "model_",
        "path_to_save": path_to_save,
    }

    objective_wrapper = ObjectiveWrapper(
        test_data=validation_data,
        forcings_input=forcings_validation_data,
        initial_state=initial_state_validation,
        n_steps=n_validation,
    )

    hyper_search = ParamHyperOpt(
        params_intervals=params_intervals,
        params_suggestions=params_suggestions,
        name="oscillator_search",
        direction="minimize",
        trainer_template=HyperModelPoolESN,
        objective_wrapper=objective_wrapper,
        objective_function=objective,
        others_params=others_params,
        refresh=True,
    )

    hyper_search.set_data(
        input_train_data=input_data,
        target_train_data=target_data,
        auxiliary_train_data=forcings_train_data,
        input_validation_data=validation_data,
        target_validation_data=validation_data,
        auxiliary_validation_data=forcings_validation_data,
        input_test_data=test_data,
        target_test_data=test_data,
        auxiliary_test_data=forcings_test_data,
    )

    hyper_search.optimize(n_trials=n_trials)


if __name__ == "__main__":
    # Reading command-line arguments
    parser = ArgumentParser(description="Argument parsers")

    parser.add_argument("--path_to_save", type=str)

    args = parser.parse_args()

    path_to_save = args.path_to_save

    test_esn_nonlinear_forcing_numba(path_to_save=path_to_save)
