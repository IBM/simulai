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

from simulai.metrics import L2Norm
from simulai.metrics import SampleWiseErrorNorm, FeatureWiseErrorNorm
from simulai.metrics import MemorySizeEval


import pytest
import h5py
import os
from simulai.utilities import make_temp_directory


class TestMetrics(TestCase):

    def setUp(self) -> None:
        pass

    def test_l2norm_clean_nan_and_large(self):

        # When there is invalid data in the datasets (masked as NaN or others)
        # it will be removed during the error evaluation
        a = 1.01
        b = 1.0
        data = np.array([a, np.nan, 1e16, -1])
        reference_data = np.array([b, np.nan, -1e16, -1])

        n = L2Norm(mask=-1)(data=data, reference_data=reference_data, )

        self.assertTrue(not np.isnan(n), 'finish')
        self.assertTrue(abs(a-b) == n, 'finish')

    def test_memory_size_eval(self):

        shape = (140000, 1000)
        max_batches = shape[0]
        batch_sizer = MemorySizeEval(memory_tol_percent=0.5)
        n_batches = batch_sizer(max_batches=max_batches, shape=shape)

        if n_batches <= max_batches:
            nbytes = np.prod(shape)*batch_sizer.size_default
            assert nbytes/n_batches <= batch_sizer.available_memory
            print(nbytes, batch_sizer.available_memory)
        else:
            pass

    def test_samplewisenorm(self):

        batch_sizer = MemorySizeEval(memory_tol_percent=0.1)

        Nx = int(3)
        Ny = int(5)
        Nt = int(7)

        x = np.linspace(0, 1, Nx)
        y = np.linspace(3, 4, Ny)
        t = np.linspace(5, 6, Nt)

        T, X, Y = np.meshgrid(t, x, y, indexing='ij')

        Z_array = np.core.records.fromarrays([np.random.rand(*T[..., None].shape),
                                                  np.random.rand(*X[..., None].shape),
                                                  np.random.rand(*Y[..., None].shape),],
                                             names=['T', 'X', 'Y'],
                                             formats=['f8', 'f8', 'f8'])

        Z_array_ref = np.core.records.fromarrays([T[..., None], X[..., None], Y[..., None]],
                                             names=['T', 'X', 'Y'],
                                             formats=['f8', 'f8', 'f8'])

        batch_size_list = [3, batch_sizer]
        data_interval_list = [[2, 4], [0, 7]]
        with make_temp_directory() as tmp_dir:

            test_data = os.path.join(tmp_dir, f'data.h5')
            with h5py.File(test_data, 'w') as fp:
                dataset = fp.create_dataset('data', shape=(Nt, Nx, Ny, 1), dtype=[('T', np.float),
                                                                                  ('X', np.float),
                                                                                  ('Y', np.float)])
                dataset[...] = Z_array

            ref_data = os.path.join(tmp_dir, f'ref_data.h5')
            with h5py.File(ref_data, 'w') as fp:
                dataset = fp.create_dataset('data', shape=(Nt, Nx, Ny, 1), dtype=[('T', np.float),
                                                                                  ('X', np.float),
                                                                                  ('Y', np.float)])
                dataset[...] = Z_array_ref

            for data_interval in data_interval_list:
                for relative_norm in [True, False]:
                    for ord in [1, 2, np.inf]:
                        for batch_size in batch_size_list:
                            with h5py.File(test_data, 'r') as fp:
                                dataset = fp.get('data')
                                with h5py.File(ref_data, 'r') as fp2:
                                    reference = fp2.get('data')

                                    nk = SampleWiseErrorNorm()(data=dataset,
                                                               reference_data=reference,
                                                               relative_norm=relative_norm,
                                                               key=list(dataset.dtype.names),
                                                               data_interval=data_interval,
                                                               batch_size=batch_size,
                                                               ord=ord)

                                    nfk = FeatureWiseErrorNorm()(data=dataset,
                                                                 reference_data=reference,
                                                                 relative_norm=relative_norm,
                                                                 key=list(dataset.dtype.names),
                                                                 data_interval=data_interval,
                                                                 reference_data_interval=data_interval,
                                                                 batch_size=batch_size,
                                                                 ord=ord)

                                    for key in dataset.dtype.names:

                                        n1 = SampleWiseErrorNorm()(data=dataset,
                                                                   reference_data=reference,
                                                                   relative_norm=relative_norm,
                                                                   key=key,
                                                                   data_interval=data_interval,
                                                                   batch_size=batch_size,
                                                                   ord=ord)

                                        nf1 = FeatureWiseErrorNorm()(data=dataset,
                                                                     reference_data=reference,
                                                                     relative_norm=relative_norm,
                                                                     key=key,
                                                                     data_interval=data_interval,
                                                                     reference_data_interval=data_interval,
                                                                     batch_size=batch_size,
                                                                     ord=ord)

                                        self.assertTrue(np.array_equal(nk[key], n1))
                                        self.assertTrue(np.array_equal(nfk[key], nf1))

                                        d = dataset[key][...]
                                        r = reference[key][...]
                                        n2 = SampleWiseErrorNorm()(data=d,
                                                                   reference_data=r,
                                                                   relative_norm=relative_norm,
                                                                   key=None,
                                                                   data_interval=data_interval,
                                                                   batch_size=batch_size,
                                                                   ord=ord)

                                        nf2 = FeatureWiseErrorNorm()(data=d,
                                                                     reference_data=r,
                                                                     relative_norm=relative_norm,
                                                                     key=None,
                                                                     data_interval=data_interval,
                                                                     reference_data_interval=data_interval,
                                                                     batch_size=batch_size,
                                                                     ord=ord)

                                        self.assertTrue(np.allclose(n1, n2))
                                        self.assertTrue(np.allclose(nf1, nf2))

                                        s = data_interval[1] - data_interval[0]
                                        dn = np.linalg.norm(np.reshape(d[slice(*data_interval)]-r[slice(*data_interval)], [s, -1]), ord=ord, axis=0)
                                        rn = np.linalg.norm(np.reshape(r[slice(*data_interval)], [s, -1]), ord=ord, axis=0)

                                        dfn = np.linalg.norm(np.reshape(d[slice(*data_interval)]-r[slice(*data_interval)], [s, -1]), ord=ord, axis=1)
                                        rfn = np.linalg.norm(np.reshape(r[slice(*data_interval)], [s, -1]), ord=ord, axis=1)

                                        if relative_norm:
                                            rel = dn/rn
                                            relf = dfn/rfn
                                            not_nan = np.logical_not(np.isnan(rel))
                                            self.assertTrue(np.allclose(n2[not_nan], rel[not_nan]))

                                            not_nan_f = np.logical_not(np.isnan(relf))
                                            self.assertTrue(np.allclose(nf2[not_nan_f], relf[not_nan_f]))
                                        else:
                                            self.assertTrue(np.allclose(n2, dn))
                                            self.assertTrue(np.allclose(nf2, dfn))


