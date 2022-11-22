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
from simulai.io import Sampling
from simulai.metrics import MemorySizeEval
import numpy as np
import pytest
import h5py
import os

from simulai.utilities import make_temp_directory


class TestReshaper(TestCase):

    def setUp(self) -> None:
        pass

    def test_sample(self):

        batch_sizer = MemorySizeEval(memory_tol_percent=0.1)

        Nx = int(30)
        Ny = int(50)
        Nt = int(70)

        x = np.linspace(0, 1, Nx)
        y = np.linspace(2, 3, Ny)
        t = np.linspace(4, 5, Nt)

        T, X, Y = np.meshgrid(t, x, y, indexing='ij')

        Z_array = np.core.records.fromarrays([T[..., None], X[..., None], Y[..., None]],
                                             names=['T', 'X', 'Y'],
                                             formats=['f8', 'f8', 'f8'])

        batch_size_list = [3, batch_sizer]
        data_intervals = [None, [0, 10], [30, 40], [0, 70]]
        for data_interval in data_intervals:
            for sampling_choices_fraction in [0.6, 1]:
                for batch_size in batch_size_list:
                    with make_temp_directory() as tmp_dir:

                        test_data = os.path.join(tmp_dir, f'data.h5')
                        with h5py.File(test_data, 'w') as fp:

                            dataset = fp.create_dataset('data', shape=(Nt, Nx, Ny, 1), dtype=[('T', np.float),
                                                                                              ('X', np.float),
                                                                                              ('Y', np.float)])

                            dataset[...] = Z_array

                        with h5py.File(test_data, 'r') as fp:
                            dataset = fp.get('data')
                            d = dataset[...]
                            for name in dataset.dtype.names:
                                self.assertTrue(np.array_equal(d[name], Z_array[name]))

                            dump_sampler = os.path.join(tmp_dir, f"exec_data_sampled.h5")
                            sampler = Sampling(choices_fraction=sampling_choices_fraction, shuffling=True)
                            sampler.prepare_input_structured_data(dataset, batch_size=batch_size,
                                                                  data_interval=data_interval,
                                                                  dump_path=dump_sampler)

                            sample = sampler.prepare_input_data(d, data_interval=data_interval)

                            with h5py.File(dump_sampler, 'r') as fp_sampled:
                                if data_interval is None:
                                    data_interval = [0, Z_array.shape[0]]
                                dataset_sample = fp_sampled.get('data')
                                dc = dataset_sample[...]

                                dc_cat1 = np.concatenate([dc[n] for n in dc.dtype.names], axis=3)
                                dc_cat2 = np.concatenate([sample[n] for n in sample.dtype.names], axis=3)
                                z_cat = np.concatenate([Z_array[slice(*data_interval)][n] for n in dc.dtype.names], axis=3)
                                for dc_cat in [dc_cat1, dc_cat2]:
                                    self.assertTrue(np.unique(np.concatenate((dc_cat, z_cat), axis=0), axis=0).shape[0] - np.unique(z_cat, axis=0).shape[0] == 0,)

        self.assertTrue(True, 'end')
