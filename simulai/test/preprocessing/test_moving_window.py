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
from unittest import TestCase

from simulai.math.progression import gp
from simulai.io import MovingWindow

''' Testing to apply the Moving Window operation, a
    pre-processing used for constructing datasets for
    recurrent neural network algorithms as LSTM and GRU
'''

class TestMovingWindow(TestCase):

    def setUp(self) -> None:
        pass

    def test_input_equal_output(self) -> None:

        # Constructing data
        t_max = 1
        t_min = 0
        Nt = 10000

        t = np.linspace(t_min, t_max, Nt)
        omega = np.array(gp(init=np.pi, factor=2, n=10))

        T, Omega = np.meshgrid(t, omega, indexing='ij')

        # Generic function U = cos(omega*t)
        U = np.cos(Omega*T)

        history_size = 10
        horizon_size = 1
        skip_size = 1

        n_samples = U.shape[0]
        n_features = U.shape[-1]

        moving_window = MovingWindow(history_size=history_size,
                                     horizon_size=horizon_size,
                                     skip_size=skip_size)

        input_data, target_data = moving_window(input_data=U, output_data=U)

        assert input_data.shape[1:] == (history_size, n_features),\
                                        f" The input shape should be ({n_samples}, {history_size}, {n_features})," \
                                        f" something is wrong in MovingWindow"

        assert target_data.shape[1:] == (horizon_size, n_features), \
                                         f" The target shape should be ({n_samples}, {history_size}, {n_features})," \
                                         f" something is wrong in MovingWindow"


        print("Moving Window operation performed.")


    def test_input_different_output(self) -> None:

        # Constructing data
        t_max = 1
        t_min = 0
        Nt = 10000

        t = np.linspace(t_min, t_max, Nt)
        omega = np.array(gp(init=np.pi, factor=2, n=10))

        T, Omega = np.meshgrid(t, omega, indexing='ij')

        # Generic function U = cos(omega*t)
        U = np.cos(Omega*T)

        history_size = 10
        horizon_size = 1
        skip_size = 1

        n_samples = U.shape[0]
        n_features_input = U.shape[-1]
        n_features_output = int(n_features_input/2)
        V = U[:, n_features_output:]

        moving_window = MovingWindow(history_size=history_size,
                                     horizon_size=horizon_size,
                                     skip_size=skip_size)

        input_data, target_data = moving_window(input_data=U, output_data=V)

        assert input_data.shape[1:] == (history_size, n_features_input),\
                                        f" The input shape should be ({n_samples}, {history_size}, {n_features_input})," \
                                        f" something is wrong in MovingWindow"

        assert target_data.shape[1:] == (horizon_size, n_features_output), \
                                         f" The target shape should be ({n_samples}, {history_size}, {n_features_output})," \
                                         f" something is wrong in MovingWindow"

        print("Moving Window operation performed.")
