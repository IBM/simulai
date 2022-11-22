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
import random
import h5py
import os


from simulai.io import MapValid
from simulai.simulation import Pipeline
from simulai.rom import IByPass
from simulai.utilities import make_temp_directory

class TestMapValid(TestCase):

    def setUp(self) -> None:
        pass

    def data_gen(self, size=None, valid_percent=0.10, mask=None):

        assert isinstance(size, tuple), "size must be a tuple of integers"

        data = np.zeros(size)
        data.fill(mask)

        n_samples = size[0]

        n_tot = np.product(size[1:])

        n_valid = int(valid_percent * n_tot)

        indices = list(np.arange(0, n_tot))

        samples = np.array(random.sample(indices, n_valid)).tolist()

        for t in range(n_samples):

            data_t = data[t].reshape((n_tot, 1))

            data_t[samples] = np.random.rand(n_valid)[:, None]

            data[t] = data_t.reshape(size[1:])

        return data

    def data_gen_structured(self, size=None, valid_percent=0.10, mask=None):

        data_ = self.data_gen(size=size, valid_percent=valid_percent, mask=mask)

        data = np.core.records.fromarrays([data_[..., i:i+1] for i in range(data_.shape[-1])],
                                          names=[f'var_{i}' for i in range(data_.shape[-1])],
                                          formats=data_.shape[-1]*['f8'])

        return data

    def test_mapvalid(self):

        # Generating test_data
        Nx = 500
        Ny = 250
        Nz = 20

        masks = [np.inf, np.NaN, np.nan, 1e16, 0, -9999999]

        config = {'replace_mask_with_large_number': False}

        for mask in masks:

            print(f"Testing using the mask: {mask}")

            data = self.data_gen(size=(Nx, Ny, Nz), valid_percent=0.50, mask=mask)

            map_valid = MapValid(config=config, mask=mask)
            reshaped_data = map_valid.prepare_input_data(data=data)

            recovered_data = map_valid.prepare_output_data(data=reshaped_data)

            not_nan = np.logical_not(np.isnan(recovered_data))
            assert np.all(data[not_nan] == recovered_data[not_nan]),\
                "The original and the recovered are not equal"

    def test_mapvalid_structutred(self):

        # Generating test_data
        Nx = 500
        Ny = 250
        Nz = 20

        masks = [np.inf, np.NaN, np.nan, 1e16, 0, -9999999]

        for mask in masks:

            print(f"Testing using the mask: {mask}")

            data = self.data_gen_structured(size=(Nx, Ny, Nz), valid_percent=0.50, mask=mask)
            batch_size = 10

            with make_temp_directory() as tmp_dir:
                with h5py.File(os.path.join(tmp_dir, 'test_data.h5'), 'w') as fp:

                    dset = fp.create_dataset('data', shape=data.shape, dtype=data.dtype)

                    dset[:] = data

                    mapvalid_config = {'return_the_same_mask': True}
                    pipeline = Pipeline(stages=[('data_preparer', MapValid(config=mapvalid_config,
                                                                           mask=mask)),
                                                ('rom', IByPass())],
                                        channels_last=True)

                    pipeline.exec(input_data=dset,
                                  data_interval=[0, dset.shape[0]],
                                  batch_size=batch_size)

                    projected = pipeline.project_data(data=dset,
                                                      data_interval=[0, dset.shape[0]],
                                                      variables_list=dset.dtype.names,
                                                      batch_size=batch_size)

                    recovered_data = pipeline.reconstruct_data(data=projected,
                                                              data_interval=[0, dset.shape[0]],
                                                              variables_list=dset.dtype.names,
                                                              batch_size=batch_size,
                                                              dump_path=os.path.join(tmp_dir, f"reconstruction.h5")
                                                              )

                    recdata = recovered_data[:].view(float)
                    data = dset[:].view(float)

                    not_nan = np.logical_not(np.isnan(recdata.view(float)))
                    assert np.all(data[not_nan] == recdata[not_nan]),\
                        "The original and the recovered are not equal"
