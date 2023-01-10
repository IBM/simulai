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


import numpy as np


from simulai.utilities.oscillator_solver import oscillator_solver_forcing
from simulai.models import ModelPool
from simulai.utilities import make_temp_directory

class TestModelPoolAffinemapping:

    def __init__(self):
        pass
    def test_modelpool_nonlinear_forcing_affinemapping(self):

        n_steps = 100
        A = 1
        T = 50
        dt = T/ n_steps

        forcings = A * np.random.rand(n_steps, 2)
        initial_state = np.array([2, 0])[None, :]

        n_field = 2  # number of field values to predict
        n_forcing = 2  # number of forcing terms
        sub_model_number_of_inputs = n_field + n_forcing  # size of the data

        nt = int(0.9 * n_steps)  # size of time steps
        nt_test = n_steps - nt

        oscillator_data, _ = oscillator_solver_forcing(T, dt, initial_state, forcing=forcings)

        field_data = oscillator_data  # manufactured nonlinear oscillator data

        n_esn = 50  # 20 random models search space for the solution

        model_type = 'AffineMapping'

        train_data = field_data[:nt, :]
        test_data = field_data[nt:, :]

        input_data = train_data[:-1, :]
        target_data = train_data[1:, :]

        forcings_train_data = forcings[:nt, :]

        forcings_input = forcings_train_data[:-1]
        forcings_input_test = forcings[nt:, :]

        initial_state = train_data[-1:, :]

        with make_temp_directory() as default_model_dir:

            rc_config = {
                'number_of_inputs': sub_model_number_of_inputs,
                'number_of_outputs': n_field,
                'estimate_linear_transition': True,
                'estimate_bias_transition': True,
            }

            solution_pool = ModelPool(config={'template': 'no_communication_series',
                                              'n_inputs': n_field + n_forcing,
                                              'n_auxiliary': n_forcing,
                                              'n_outputs': n_field},
                                      model_type=model_type,
                                      model_config=rc_config)

            for idx in range(n_field):
                solution_pool.fit(input_data=input_data,
                                  target_data=target_data,
                                  auxiliary_data=forcings_input,
                                  index=idx)

            extrapolation_data = solution_pool.predict(initial_state=initial_state,
                                                       auxiliary_data=forcings_input_test,
                                                       horizon=nt_test)

            assert extrapolation_data.shape == (nt_test, n_field), f"It is expected shape {(nt_test, n_field)}," \
                                                                   f" but received {extrapolation_data.shape}"
