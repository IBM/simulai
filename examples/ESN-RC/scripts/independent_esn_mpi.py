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
import random
import warnings

import matplotlib.pyplot as plt
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
from simulai.utilities import make_temp_directory


def test_modelpool_nonlinear_forcing_numba_MPI(n_numba_workers: int = None):
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

    nt = int(0.9 * n_steps)  # size of time steps
    nt_test = n_steps - nt

    oscillator_data, _ = oscillator_solver_forcing(
        T, dt, initial_state, forcing=forcings
    )

    field_data = oscillator_data  # manufactured nonlinear oscillator data

    model_type = "EchoStateNetwork"

    train_data = field_data[:nt, :]
    test_data = field_data[nt:, :]

    input_data = train_data[:-1, :]
    target_data = train_data[1:, :]

    forcings_train_data = forcings[:nt, :]

    forcings_input = forcings_train_data[:-1]
    forcings_input_test = forcings[nt:, :]

    initial_state = train_data[-1:, :]

    with make_temp_directory() as default_model_dir:
        reservoir_dim = random.randint(1000, 2000)

        rc_config = {
            "reservoir_dim": reservoir_dim,
            "sparsity_level": 0.05 * reservoir_dim,
            "radius": 0.5,
            "sigma": 0.5,
            "beta": 1e-4,
            "number_of_inputs": sub_model_number_of_inputs,
            "global_matrix_constructor_str": "numba",
            "n_workers": n_numba_workers,
            "solver": "linear_system",
            "memory_percent": 0.8,
        }

        solution_pool = ModelPool(
            config={
                "template": "independent_series",
                "n_inputs": n_field + n_forcing,
                "n_auxiliary": n_forcing,
                "n_outputs": n_field,
            },
            model_type=model_type,
            model_config=rc_config,
            parallel="mpi",
        )

        msg = solution_pool.fit(
            input_data=input_data,
            target_data=target_data,
            auxiliary_data=forcings_input,
        )

        if msg is True:
            extrapolation_data = solution_pool.predict(
                initial_state=initial_state,
                auxiliary_data=forcings_input_test,
                horizon=nt_test,
            )

            for j in range(n_field):
                plt.plot(test_data[:, j], label=f"Exact variable {j}")
                plt.plot(extrapolation_data[:, j], label=f"Approximated variable {j}")
                plt.legend()
                plt.show()

            l2_norm = L2Norm()

            error = l2_norm(
                data=extrapolation_data, reference_data=test_data, relative_norm=True
            )

            print(f"Error {100*error} %")


if __name__ == "__main__":
    # Reading command-line arguments
    parser = ArgumentParser(description="Argument parsers")

    parser.add_argument("--n_numba_workers", type=str, default=4)
    args = parser.parse_args()

    n_numba_workers = args.n_numba_workers

    test_modelpool_nonlinear_forcing_numba_MPI(n_numba_workers=n_numba_workers)
