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

# (C) Copyright IBM Corporation 2017, 2018, 2019
# U.S. Government Users Restricted Rights:  Use, duplication or disclosure restricted
# by GSA ADP Schedule Contract with IBM Corp.
#
# Author: Leonardo P. Tizzei <ltizzei@br.ibm.com>
from unittest import TestCase
from simulai.metrics import MeanEvaluation
from simulai.metrics import MemorySizeEval
import numpy as np
import pytest
import h5py
import os
from simulai.io import MapValid, Reshaper

from simulai.utilities import make_temp_directory


class TestMeanEvaluator(TestCase):

    def setUp(self) -> None:
        pass

    def test_mean(self):

        batch_sizer = MemorySizeEval(memory_tol_percent=0.1)

        batch_size_list = [3, batch_sizer]
        data_interval_list = [[2, 4], [0, 7]]
        data_preparers = [MapValid(config={}), Reshaper(), None]


        Nx = int(3)
        Ny = int(5)
        Nt = int(7)

        x = np.ones((Nx,))
        y = 2*np.ones((Ny,))
        t = 3*np.ones((Nt,))
        vars_1 = [x, y, t]

        x = np.linspace(0, 1, Nx)
        y = np.linspace(2, 3, Ny)
        t = np.linspace(4, 5, Nt)
        vars_2 = [x, y, t]

        for x, y, t in [vars_1, vars_2]:

            T, X, Y = np.meshgrid(t, x, y, indexing='ij')

            Z_array = np.core.records.fromarrays([T, X, Y],
                                                 names=['T', 'X', 'Y'],
                                                 formats=['f8', 'f8', 'f8'])

            for data_preparer in data_preparers:
                for data_interval in data_interval_list:
                    for batch_size in batch_size_list:
                        with make_temp_directory() as tmp_dir:

                            test_data = os.path.join(tmp_dir, f'data.h5')
                            with h5py.File(test_data, 'w') as fp:

                                dataset = fp.create_dataset('data', shape=Z_array.shape, dtype=Z_array.dtype)

                                dataset[...] = Z_array

                            with h5py.File(test_data, 'r') as fp:
                                dataset = fp.get('data')
                                d = dataset[...]
                                for name in dataset.dtype.names:
                                    self.assertTrue(np.array_equal(d[name], Z_array[name]))

                                mean = MeanEvaluation()(dataset=dataset,
                                                        data_interval=data_interval,
                                                        batch_size=batch_size,
                                                        data_preparer=data_preparer)

                                if data_preparer is not None:
                                    z_cat = data_preparer.prepare_input_structured_data(Z_array)
                                else:
                                    z_cat = Reshaper().prepare_input_structured_data(Z_array)

                                mean_numpy = z_cat[slice(*data_interval)].mean(0).reshape(-1)

                                self.assertTrue(np.array_equal(mean, mean_numpy), f'mean values are not equal')

        self.assertTrue(True, 'end')
