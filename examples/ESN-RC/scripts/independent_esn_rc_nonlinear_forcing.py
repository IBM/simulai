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

import glob
import os
import random

import matplotlib.pyplot as plt
import numpy as np

from examples.utils.oscillator_solver import (
    oscillator_solver,
    oscillator_solver_forcing,
)
from simulai.models import ModelPool
from simulai.regression import EchoStateNetwork
from simulai.utilities import make_temp_directory


class TestModelPoolESN:
    def __init__(self):
        pass

    def test_modelpool_nonlinear(self, save_path=None):
        n_steps = 1000
        T = 50
        dt = T / n_steps

        initial_state = np.array([2, 0])[None, :]

        n_field = 2  # number of field values to predict
        sub_model_number_of_inputs = n_field  # size of the data

        nt = int(0.5 * n_steps)  # size of time steps
        nt_test = n_steps - nt

        oscillator_data, _ = oscillator_solver(T, dt, initial_state)

        field_data = oscillator_data  # manufactured nonlinear oscillator data

        n_esn = 50  # 20 random models search space for the solution

        model_type = "EchoStateNetwork"

        train_data = field_data[:nt, :]
        test_data = field_data[nt:, :]

        input_data = train_data[:-1, :]
        target_data = train_data[1:, :]

        initial_state = train_data[-1:, :]

        reservoir_dim = random.randint(5000, 10000)

        with make_temp_directory() as default_model_dir:
            rc_config = {
                "reservoir_dim": reservoir_dim,
                "sparsity_level": 0.25 * reservoir_dim,
                "radius": 0.5,
                "sigma": 0.5,
                "beta": 1e-3,
                "number_of_inputs": sub_model_number_of_inputs,
            }

            # Pool of sub-models
            solution_pool = ModelPool(
                config={
                    "template": "independent_series",
                    "n_inputs": n_field,
                    "n_outputs": n_field,
                },
                model_type=model_type,
                model_config=rc_config,
            )

            # Fitting each sub-model independently
            for idx in range(n_field):
                solution_pool.fit(
                    input_data=input_data, target_data=target_data, index=idx
                )

            extrapolation_data = solution_pool.predict(
                initial_state=initial_state, horizon=nt_test
            )

            for j in range(n_field):
                plt.plot(test_data[:, j], label=f"Exact variable {j}")
                plt.plot(extrapolation_data[:, j], label=f"Approximated variable {j}")
                plt.legend()
                plt.show()

    def test_modelpool_nonlinear_forcing_multiprocessing(self):
        n_steps = 1000
        A = 1
        T = 50
        dt = T / n_steps

        forcings = A * np.random.rand(n_steps, 2)
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

        n_esn = 50  # 20 random models search space for the solution

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
            reservoir_dim = random.randint(3000, 4000)

            rc_config = {
                "reservoir_dim": reservoir_dim,
                "sparsity_level": 0.25 * reservoir_dim,
                "radius": 0.5,
                "sigma": 0.5,
                "beta": 1e-4,
                "number_of_inputs": sub_model_number_of_inputs,
                "global_matrix_constructor_str": "multiprocessing",
                "n_workers": 4,
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
            )

            for idx in range(n_field):
                solution_pool.fit(
                    input_data=input_data,
                    target_data=target_data,
                    auxiliary_data=forcings_input,
                    index=idx,
                )

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

    def test_modelpool_nonlinear_forcing_numba(self):
        n_steps = 1000
        A = 1
        T = 50
        dt = T / n_steps

        forcings = A * np.random.rand(n_steps, 2)
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

        n_esn = 50  # 20 random models search space for the solution

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
            reservoir_dim = random.randint(3000, 4000)

            rc_config = {
                "reservoir_dim": reservoir_dim,
                "sparsity_level": 0.25 * reservoir_dim,
                "radius": 0.5,
                "sigma": 0.5,
                "beta": 1e-4,
                "number_of_inputs": sub_model_number_of_inputs,
                "global_matrix_constructor_str": "numba",
                "n_workers": 4,
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
            )

            for idx in range(n_field):
                solution_pool.fit(
                    input_data=input_data,
                    target_data=target_data,
                    auxiliary_data=forcings_input,
                    index=idx,
                )

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

    def test_modelpool_nonlinear_forcing_no_communication(self):
        n_steps = 1000
        A = 1
        T = 50
        dt = T / n_steps

        forcings = A * np.random.rand(n_steps, 2)
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

        n_esn = 50  # 20 random models search space for the solution

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
            reservoir_dim = random.randint(3000, 4000)

            rc_config = {
                "reservoir_dim": reservoir_dim,
                "sparsity_level": 0.25 * reservoir_dim,
                "radius": 0.5,
                "sigma": 0.5,
                "beta": 1e-4,
                "number_of_inputs": sub_model_number_of_inputs,
                #'global_matrix_constructor_str': 'numba',
                "n_workers": 4,
                "memory_percent": 0.8,
            }

            solution_pool = ModelPool(
                config={
                    "template": "no_communication_series",
                    "n_inputs": n_field + n_forcing,
                    "n_auxiliary": n_forcing,
                    "n_outputs": n_field,
                },
                model_type=model_type,
                model_config=rc_config,
            )

            for idx in range(n_field):
                solution_pool.fit(
                    input_data=input_data,
                    target_data=target_data,
                    auxiliary_data=forcings_input,
                    index=idx,
                )

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

    def test_modelpool_nonlinear_forcing_deepesn(self):
        n_steps = 1000
        A = 1
        T = 50
        dt = T / n_steps

        forcings = A * np.random.rand(n_steps, 2)
        initial_state = np.array([2, 0])[None, :]

        n_field = 2  # number of field values to predict
        n_forcing = 2  # number of forcing terms
        sub_model_number_of_inputs = n_field + n_forcing  # size of the data

        nt = int(0.9 * n_steps)  # size of time steps
        nt_test = n_steps - nt

        oscillator_data, _ = oscillator_solver_forcing(
            T, dt, initial_state, forcing=forcings
        )

        print("\n")

        field_data = oscillator_data  # manufactured nonlinear oscillator data

        model_type = "DeepEchoStateNetwork"

        train_data = field_data[:nt, :]
        test_data = field_data[nt:, :]

        input_data = train_data[:-1, :]
        target_data = train_data[1:, :]

        forcings_train_data = forcings[:nt, :]

        forcings_input = forcings_train_data[:-1]
        forcings_input_test = forcings[nt:, :]

        initial_state = train_data[-1:, :]

        with make_temp_directory() as default_model_dir:
            reservoir_dim = random.randint(100, 100)

            rc_config = {
                "reservoir_dim": reservoir_dim,
                "sparsity_level": 0.25 * reservoir_dim,
                "radius": 0.5,
                "sigma": 0.5,
                "beta": 1e-4,
                "n_layers": 2,
                "number_of_inputs": sub_model_number_of_inputs,
                "global_matrix_constructor_str": "numba",
                "n_workers": 8,
                "memory_percent": 0.8,
                "all_for_input": 1,
                "transformation": "T0",
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
            )

            for idx in range(n_field):
                solution_pool.fit(
                    input_data=input_data,
                    target_data=target_data,
                    auxiliary_data=forcings_input,
                    index=idx,
                )

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

    def test_modelpool_nonlinear_forcing_wideesn(self):
        n_steps = 1000
        A = 1
        T = 50
        dt = T / n_steps

        forcings = A * np.random.rand(n_steps, 2)
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

        model_type = "WideEchoStateNetwork"

        train_data = field_data[:nt, :]
        test_data = field_data[nt:, :]

        input_data = train_data[:-1, :]
        target_data = train_data[1:, :]

        forcings_train_data = forcings[:nt, :]

        forcings_input = forcings_train_data[:-1]
        forcings_input_test = forcings[nt:, :]

        initial_state = train_data[-1:, :]

        with make_temp_directory() as default_model_dir:
            reservoir_dim = random.randint(400, 401)

            rc_config = {
                "reservoir_dim": reservoir_dim,
                "sparsity_level": 0.25 * reservoir_dim,
                "radius": 0.5,
                "sigma": 0.5,
                "beta": 1e-4,
                "n_layers": 20,
                "number_of_inputs": sub_model_number_of_inputs,
                "global_matrix_constructor_str": "numba",
                "n_workers": 8,
                "memory_percent": 0.8,
                "transformation": "T0",
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
            )

            for idx in range(n_field):
                solution_pool.fit(
                    input_data=input_data,
                    target_data=target_data,
                    auxiliary_data=forcings_input,
                    index=idx,
                )

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

    def test_esn_nonlinear_forcing_numba(self):
        from simulai.workflows import StepwiseExtrapolation

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

        train_data = field_data[:nt, :]
        test_data = field_data[nt:, :]

        input_data = train_data[:-1, :]
        target_data = train_data[1:, :]

        forcings_train_data = forcings[:nt, :]

        forcings_input = forcings_train_data[:-1]
        forcings_input_test = forcings[nt:, :]

        initial_state = train_data[-1:, :]

        data = np.hstack([input_data, forcings_input])

        reservoir_dim = random.randint(5000, 8000)

        rc_config = {
            "reservoir_dim": reservoir_dim,
            "sparsity_level": 0.25 * reservoir_dim,
            "radius": 0.5,
            "sigma": 0.5,
            "beta": 1e-4,
            "number_of_inputs": sub_model_number_of_inputs,
            "global_matrix_constructor_str": "numba",
            "n_workers": 4,
        }

        esn = EchoStateNetwork(**rc_config)

        esn.fit(input_data=data, target_data=target_data)

        extrapolator = StepwiseExtrapolation(model=esn, keys=["ESN_0"])

        estimated_data = extrapolator.predict(
            initial_state=initial_state,
            auxiliary_data=forcings_input_test,
            horizon=nt_test,
        )

        for j in range(n_field):
            plt.plot(test_data[:, j], label=f"Exact variable {j}")
            plt.plot(estimated_data[:, j], label=f"Approximated variable {j}")
            plt.legend()
            plt.show()


class HyperSearchObjective:
    def __init__(
        self,
        ix,
        default_model_dir,
        target_states,
        field_data,
        forcings_data,
        model_type,
    ):
        self.ix = ix
        self.default_model_dir = default_model_dir
        self.models = [
            os.path.basename(os.path.splitext(m)[0])
            for m in glob.glob(os.path.join(default_model_dir, "*.pkl"))
        ]
        self.target_states = target_states
        self.field_data = field_data
        self.forcings_data = forcings_data
        self.model_type = model_type

    #    def __call__(self, trial: optuna.Trial):
    #        model = trial.suggest_categorical('model', self.models)
    #        return self.objective(model)

    def objective(self, model):
        p_ix = ModelPool(
            config={
                "template": "independent_series",
                "n_inputs": self.field_data.shape[1] + self.forcings_data.shape[1],
                "n_outputs": self.field_data.shape[1],
            },
            model_type=self.model_type,
        )
        p_ix.load_model(self.default_model_dir, model, self.ix)

        p_ix.fit(
            input_data=self.field_data,
            target_data=self.target_states,
            auxiliary_data=self.forcings_data,
            index=self.ix,
        )

        esn = p_ix.model_instances_list[p_ix._make_id(self.ix)]
        esn.set_reference(esn.default_state)
        esn.reset()

        prediction = []
        for step in range(0, self.forcings_data.shape[0]):
            current_state = esn.predict(
                initial_data=np.hstack(
                    (
                        self.field_data[step, :],
                        self.forcings_data[step, :],
                    )
                ),
                horizon=1,
            )
            prediction.append(current_state)
        target = np.vstack(prediction)

        error = np.linalg.norm(
            np.reshape(self.target_states[:, self.ix : self.ix + 1] - target, (-1,))
        )

        return error, esn
