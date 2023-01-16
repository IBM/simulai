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

from simulai.regression import ELM
from simulai.file import load_pkl

class TestELM(TestCase):

    def setUp(self) -> None:
        self.errors = list()

    # Data preparing
    def u(self, t, x, L: float = None, t_max: float = None) -> np.ndarray:
        return np.sin(4 * np.pi * t * np.cos(5 * np.pi * (t / t_max)) * (x / L - 1 / 2) ** 2) * np.cos(
            5 * np.pi * (t / t_max - 1 / 2) ** 2)

    def test_elm_instantiation(self):

        t_max = 10
        L = 5
        K = 512
        N = 10_000

        n_train = 20_000

        x_interval = [0, L]
        time_interval = [0, t_max]

        x = np.linspace(*x_interval, K)
        t = np.linspace(*time_interval, N)

        T, X = np.meshgrid(t, x, indexing='ij')
        output_data = self.u(T, X, L=L, t_max=t_max)

        positions = np.stack([X[::100].flatten(), T[::100].flatten()], axis=1)
        positions = 2 * positions / np.array([L, t_max]) - 1

        n_t, n_x = output_data.shape

        x_i = np.random.randint(0, n_x, size=(n_train, 1))
        t_i = np.random.randint(0, n_t, size=(n_train, 1))

        input_train = 2 * np.hstack([x[x_i], t[t_i]]) / np.array([L, t_max]) - 1
        output_train = output_data[t_i, x_i]

        config = {'n_i': 2,
                  'n_o': 1,
                  'h': 800}

        approximator = ELM(**config)

        approximator.fit(input_data=input_train, target_data=output_train, lambd=1e-5)

        evaluated = approximator.eval(input_data=positions[::100])
        assert isinstance(evaluated, np.ndarray)

        approximator.save(name='elm_model', path='/tmp')

        # Testing to save and reload ELM
        approximator_reloaded = load_pkl(path='/tmp/elm_model.pkl')

        assert isinstance(approximator_reloaded, ELM)

    def test_failed_load_pkl(self):

        config = {'n_i': 2,
                  'n_o': 1,
                  'h': 800}

        approximator = ELM(**config)

        model_reloaded = None

        try:
           model_reloaded = load_pkl(path=f'/tmp/elm_model.{id(approximator)}')

        except:

           pass

        assert model_reloaded == None

        try:
           model_reloaded = load_pkl(path=f'/tmp/elm_model_{id(approximator)}.pkl')

        except :
           pass

        assert model_reloaded == None