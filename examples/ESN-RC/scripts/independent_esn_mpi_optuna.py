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
import warnings

import numpy as np

MPI_GLOBAL_AVAILABILITY = True

try:
    from mpi4py import MPI
except:
    MPI_GLOBAL_AVAILABILITY = False
    warnings.warn(f"Trying to import MPI in {__file__}.")
    warnings.warn(
        "mpi4py is not installed. If you want to execute MPI jobs, we recommend you install it."
    )

from argparse import ArgumentParser

from examples.utils.oscillator_solver import oscillator_solver_forcing
from simulai.metrics import L2Norm
from simulai.models import ModelPool
from simulai.templates import HyperTrainTemplate
from simulai.workflows import ParamHyperOpt


# Hyperparameter search for  ESN+ModelPool using Optuna
class HyperModelPoolESN(HyperTrainTemplate):
    def __init__(
        self,
        trial_config: dict = None,
        set_type="hard",
        path_to_model: str = None,
        other_params: dict = None,
    ):
        self.model = None

        self.path_to_model = path_to_model
        self.others_params = other_params

        required_keys = [
            "n_field",
            "n_forcing",
            "model_type",
            "sub_model_number_of_inputs",
            "global_matrix_constructor_str",
            "n_numba_workers",
            "memory_percent",
            "parallel",
            "id",
            "tag",
        ]

        self.n_field = None
        self.n_forcing = None
        self.model_type = None
        self.sub_model_number_of_inputs = None
        self.global_matrix_constructor_str = None
        self.n_numba_workers = None
        self.memory_percent = None
        self.parallel = None
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
            "n_workers": self.n_numba_workers,
            "memory_percent": self.memory_percent,
        }

        rc_config.update(extra_params)

        self.model = ModelPool(
            config={
                "template": "independent_series",
                "n_inputs": self.n_field + self.n_forcing,
                "n_auxiliary": self.n_forcing,
                "n_outputs": self.n_field,
            },
            model_type=self.model_type,
            model_config=rc_config,
            parallel=self.parallel,
        )

    def fit(self, input_train_data=None, target_train_data=None, forcings_input=None):
        msg = self.model.fit(
            input_data=input_train_data,
            target_data=target_train_data,
            auxiliary_data=forcings_input,
        )

        return msg


def objective(
    model=None,
    initial_state=None,
    test_data=None,
    forcings_input_test=None,
    nt_test=None,
):
    extrapolation_data = model.predict(
        initial_state=initial_state, auxiliary_data=forcings_input_test, horizon=nt_test
    )

    l2_norm = L2Norm()

    error = l2_norm(
        data=extrapolation_data, reference_data=test_data, relative_norm=True
    )

    return error


def test_modelpool_nonlinear_forcing_numba_MPI_Optuna(
    n_numba_workers: int = None, n_trials: int = None, parallel: str = "mpi"
):
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

    train_fraction = 0.6
    validation_fraction = 0.3
    test_fraction = 0.3

    n_field = 2  # number of field values to predict
    n_forcing = 2  # number of forcing terms
    sub_model_number_of_inputs = n_field + n_forcing  # size of the data

    n_train = int(train_fraction * n_steps)  # size of time steps
    n_validation = int(validation_fraction * n_steps)
    n_test = int(test_fraction * n_steps)

    oscillator_data, _ = oscillator_solver_forcing(
        T, dt, initial_state, forcing=forcings
    )

    model_type = "EchoStateNetwork"

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
        "reservoir_dim": (2000, 5000),
        "sparsity_level": (0.1, 0.5),
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
        "global_matrix_constructor_str": "numba",
        "solver": "linear_system",
        "initial_state": initial_state_validation,
        "n_steps": n_validation,
        "n_field": n_field,
        "n_forcing": n_forcing,
        "n_numba_workers": n_numba_workers,
        "model_type": model_type,
        "memory_percent": 0.8,
        "tag": "model_",
        "parallel": parallel,
    }

    hyper_search = ParamHyperOpt(
        params_intervals=params_intervals,
        params_suggestions=params_suggestions,
        name="lorenz_hyper_search",
        direction="minimize",
        trainer_template=HyperModelPoolESN,
        objective_function=objective,
        others_params=others_params,
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

    parser.add_argument("--n_numba_workers", type=int, default=4)
    parser.add_argument("--n_trials", type=int, default=1)
    parser.add_argument("--parallel", type=str, default="mpi")

    args = parser.parse_args()

    n_numba_workers = args.n_numba_workers
    n_trials = args.n_trials
    parallel = args.parallel

    if parallel == "None":
        parallel = None

    test_modelpool_nonlinear_forcing_numba_MPI_Optuna(
        n_numba_workers=n_numba_workers, n_trials=n_trials, parallel=parallel
    )
