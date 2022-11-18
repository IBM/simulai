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

from simulai.rom import DMD

class TestDMDDecomposition(TestCase):

    def setUp(self) -> None:
        pass

    def test_run_dmd_decomposition(self) -> None:

        # One-dimensional case
        # Constructing dataset
        train_factor = 0.96

        N = 600
        Nt = 2000

        N_train = int(train_factor*Nt)
        N_test = Nt - N_train

        # Constructing dataset
        x = np.linspace(0, 1, N)
        t = np.linspace(0, 1, Nt)
        i = np.linspace(1, 10, N)
        j = np.linspace(1, 10, Nt)

        T, X = np.meshgrid(t, x, indexing='ij')
        J, I = np.meshgrid(j, i, indexing='ij')

        Z = np.sin(J*np.pi*T)*np.cos(I*np.pi*X)

        fit_data = Z[:N_train, :]

        dmd_config = {}

        dmd = DMD(config=dmd_config)

        dmd.fit(data=fit_data)

        states_list = list()
        for step in range(1, N_test+1):
            print(step)
            state = dmd.predict(step=step)
            states_list.append(state)

        states = np.vstack(states_list).T

        states = states[N:2*N].real

        assert isinstance(states, np.ndarray) and states.shape[1] == N_test, \
              "The predicion is not correct. It is necessary to check simulai.rom.DMD"
        print("DMD projection for one-dimensional case performed.")
