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

from simulai.rom import GPOD
from simulai.math.progression import gp
from simulai.special import Scattering, bidimensional_map_nonlin_3, time_function

class TestPCADecomposition(TestCase):

    def setUp(self) -> None:
        pass

    ''' Dataset constructed with an expression in which
        the variables are separable: U = exp(y)*cos(x)
    '''
    def test_2D_separable_dataset(self) -> None:

        train_factor = 0.6

        N = 1000
        N_train = int(train_factor*N)

        # Constructing dataset
        x = np.linspace(0, 1, N)
        y = np.linspace(0, 1, N)

        X, Y = np.meshgrid(x, y)
        Z = np.exp(Y)*np.cos(X)

        fit_data = Z[:N_train, :]
        test_data = Z[N_train:, :]

        N_components = gp(init=5, factor=2, n=4)

        self._exec_GPCA_tests(fit_data, test_data, N_components)

    ''' Dataset constructed with an expression in which
        the variables are not clearly separable, see simulai.special for
        a description.
    '''
    def test_2D_non_separable_dataset(self) -> None:

        train_factor = 0.6

        # Constructing dataset
        N_x = 128
        N_y = 128
        N_t = 1000

        x = np.linspace(0, 1, N_x)
        y = np.linspace(0, 1, N_y)
        t = np.linspace(0, 100, N_t)

        N_train = int(train_factor*N_t)

        T, X, Y = np.meshgrid(t, x, y, indexing='ij')

        generator = Scattering(root=time_function, scatter_op=bidimensional_map_nonlin_3)
        Z_ = generator.exec(data=T, scatter_data=(X, Y, 0.5, 0.5))
        Z_ *= generator.exec(data=T, scatter_data=(X, Y, 0.25, 0.25))

        Z = Z_.reshape(-1, Z_.shape[1]*Z_.shape[2])

        fit_data = Z[:N_train, :]
        test_data = Z[N_train:, :]

        N_components = gp(init=5, factor=2, n=4)

        self._exec_GPCA_tests(fit_data, test_data, N_components)

    def _exec_GPCA_tests(self, fit_data, test_data, N_components) -> None:

        for n_components in N_components:

            pca_config = {
                          'n_components': n_components
                         }

            sensors_distribution = n_components*[12]

            gpca_config = {'sensors_distribution': sensors_distribution}

            gpca = GPOD(pca_type='pod', pca_config=pca_config, config=gpca_config)

            print("Testing to fit a PCA ROM")
            gpca.fit(data=fit_data)

            print('Testing to project.')
            projected = gpca.project(data=test_data)

            print('Testing to reconstruct.')
            reconstructed = gpca.reconstruct(projected_data=projected)

            error = 100*np.linalg.norm(test_data - reconstructed, 2)/np.linalg.norm(test_data,2)

            print(f"PCA projection error for one-dimensional case performed with {n_components} components: {error}.")
