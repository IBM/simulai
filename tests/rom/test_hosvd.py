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

from simulai.special import Scattering, bidimensional_map_nonlin_3, time_function
from simulai.rom import HOSVD
from simulai.metrics import L2Norm
from simulai.file import load_pkl

class TestHOSVDDecomposition(TestCase):

    def setUp(self) -> None:
        pass

    def test_hosvd_sklearn(self):

        # Constructing dataset
        N_x = 32
        N_y = 32
        N_t = 200
        N_v = 3

        center_x = 0.5
        center_y = 0.5

        centers_x = center_x + 0.1 * np.random.rand(N_v)
        centers_y = center_y + 0.1 * np.random.rand(N_v)

        train_factor = 0.8

        x = np.linspace(0, 1, N_x)
        y = np.linspace(0, 1, N_y)
        t = np.linspace(0, 100, N_t)

        N_train = int(train_factor * N_t)

        T, X, Y = np.meshgrid(t, x, y, indexing='ij')

        generator = Scattering(root=time_function, scatter_op=bidimensional_map_nonlin_3)

        Z_list = list()

        for c_x, c_y in zip(centers_x, centers_y):

            Z_ = generator.exec(data=T, scatter_data=(X, Y, c_x, c_y))
            Z_ *= generator.exec(data=T, scatter_data=(X, Y, 0.25, 0.25))

            Z_list.append(Z_)

        Z = np.stack(Z_list, axis=-1)
        Z_fit = Z[:N_train, :]
        Z_test = Z[N_train:, :]

        hosvd = HOSVD(n_components=[20, 32, 32, 3], components_names=['t', 'x', 'y', 'v'], engine='sklearn')

        hosvd.fit(data=Z_fit)

        projected_Z_test = hosvd.project(data=Z_fit)

        Z_reconstructed = hosvd.reconstruct(data=projected_Z_test)


        l2_norm = L2Norm()

        error = 100*l2_norm(data=Z_reconstructed, reference_data=Z_fit, relative_norm=True)

        print(f"Projection error: {error} %.")

        # Partial reconstruction

        T_new = hosvd.T_decomp[-100:]

        Z_reconstructed = hosvd.reconstruct(data=projected_Z_test, replace_components={'t': T_new})

        l2_norm = L2Norm()

        error = 100 * l2_norm(data=Z_reconstructed, reference_data=Z_fit[-100:], relative_norm=True)

        print(f"Projection error: {error} %.")

        print("Saving to disk.")
        hosvd.save(save_path='/tmp', model_name='test_hosvd')

        print("Reloading from disk.")
        hosvd_reloaded = load_pkl(path='/tmp/test_hosvd.pkl')

        print('Process completed.')

    def test_hosvd_dask(self):

        import dask.array as da

        # Constructing dataset
        N_x = 32
        N_y = 32
        N_t = 200
        N_v = 3

        center_x = 0.5
        center_y = 0.5

        centers_x = center_x + 0.1 * np.random.rand(N_v)
        centers_y = center_y + 0.1 * np.random.rand(N_v)

        train_factor = 0.8

        x = np.linspace(0, 1, N_x)
        y = np.linspace(0, 1, N_y)
        t = np.linspace(0, 100, N_t)

        N_train = int(train_factor * N_t)

        T, X, Y = np.meshgrid(t, x, y, indexing='ij')

        generator = Scattering(root=time_function, scatter_op=bidimensional_map_nonlin_3)

        Z_list = list()

        for c_x, c_y in zip(centers_x, centers_y):
            Z_ = generator.exec(data=T, scatter_data=(X, Y, c_x, c_y))
            Z_ *= generator.exec(data=T, scatter_data=(X, Y, 0.25, 0.25))

            Z_list.append(Z_)

        Z = np.stack(Z_list, axis=-1)

        Z = da.from_array(Z, chunks=(100, N_x, N_y, N_v))

        Z_fit = Z[:N_train, :]
        Z_test = Z[N_train:, :]

        hosvd = HOSVD(n_components=[20, 32, 32, 3], components_names=['t', 'x', 'y', 'v'], engine='dask')

        hosvd.fit(data=Z_fit)

        projected_Z_test = hosvd.project(data=Z_fit)

        Z_reconstructed = hosvd.reconstruct(data=projected_Z_test)

        l2_norm = L2Norm()

        error = 100 * l2_norm(data=Z_reconstructed, reference_data=Z_fit, relative_norm=True)

        print(f"Projection error: {error} %.")

        # Partial reconstruction

        T_new = hosvd.T_decomp[-100:]

        Z_reconstructed = hosvd.reconstruct(data=projected_Z_test, replace_components={'t': T_new})

        l2_norm = L2Norm()

        error = 100 * l2_norm(data=Z_reconstructed, reference_data=Z_fit[-100:], relative_norm=True)

        print(f"Projection error: {error} %.")

        print("Saving to disk.")
        hosvd.save(save_path='/tmp', model_name='test_hosvd')

        print("Reloading from disk.")
        hosvd_reloaded = load_pkl(path='/tmp/test_hosvd.pkl')

        print('Process completed.')
