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

import h5py
import numpy as np
import torch
from math import floor, ceil
from typing import List

# Sampling batches from disk
class BatchwiseSampler:

    def __init__(self, dataset:h5py.Group=None,
                       input_variables:List[str]=None, target_variables:List[str]=None,
                       input_normalizer:callable=None, target_normalizer:callable=None,
                       channels_first:bool=None) -> None:

        # This import avoids circular importing
        from simulai.metrics import MinMaxEvaluation

        self.dataset = dataset
        self.input_variables = input_variables
        self.target_variables = target_variables

        self.input_normalizer = input_normalizer
        self.target_normalizer = target_normalizer

        self.channels_first = channels_first

        if self.channels_first is True:
            self.adjust_dimension = self._transpose_first_channel
        else:
            self.adjust_dimension = self._simple_stack

        self.minmax_eval = MinMaxEvaluation()

        # Defining if normalization will be used or not
        if self.input_normalizer is not None:
            self.exec_input_normalization = self._input_normalization
        else:
            self.exec_input_normalization = self._normalization_bypass

        if self.target_normalizer is not None:
            self.exec_target_normalization = self._target_normalization
        else:
            self.exec_target_normalization = self._normalization_bypass

    # Evaluating the global minimum and maximum  for all the
    # datasets in self.dataset
    def minmax(self, batch_size:int=None, data_interval:list=None):

        min_list = list()
        max_list = list()

        for k in self.target_variables:
            min, max = self.minmax_eval(dataset=self.dataset[k], batch_size=batch_size,
                                        data_interval=data_interval)
            min_list.append(min)
            max_list.append(max)

        return  np.min(min_list), np.max(max_list)

    @property
    def input_shape(self):

        if self.channels_first:
            shape_ = self.dataset[self.input_variables[0]].shape
            shape = (shape_[0],) + (len(self.input_variables),) + shape_[1:]
        else:
            shape = self.dataset[self.input_variables[0]].shape + (len(self.input_variables),)

        return list(shape)

    def _normalization_bypass(self, data:np.ndarray=None) -> np.ndarray:

        return data

    def _target_normalization(self, data:np.ndarray=None) -> np.ndarray:

        return self.target_normalizer(data=data)

    def _input_normalization(self, data: np.ndarray = None) -> np.ndarray:

        return self.input_normalizer(data=data)

    def _transpose_first_channel(self, variables_list:list=None) -> torch.Tensor:

        batch = np.stack(variables_list, axis=-1)

        dims = list(range(len(batch.shape)))
        dims_t = [0] + [dims[-1]] + dims[1:-1]

        batch = batch.transpose(*dims_t)

        return torch.from_numpy(batch.astype('float32'))

    def _simple_stack(self, variables_list:list=None) -> torch.Tensor:

        batch = np.stack(variables_list, dim=-1)

        return torch.from_numpy(batch.astype('float32'))

    def input_data(self, indices:np.ndarray=None) -> torch.Tensor:

        indices = np.sort(indices)

        variables_list = [self.dataset[k][indices] for k in self.input_variables]

        return self.exec_input_normalization(self.adjust_dimension(variables_list=variables_list))

    def target_data(self, indices:np.ndarray=None) -> torch.Tensor:

        indices = np.sort(indices)

        variables_list = [torch.from_numpy(self.dataset[k][indices].astype('float32')) for k in self.target_variables]

        return self.exec_target_normalization(self.adjust_dimension(variables_list=variables_list))

def batchdomain_constructor(data_interval:list=None, batch_size:int=None, batch_indices:list=None) -> list:

    if data_interval is not None:
        interval_size = data_interval[1] - data_interval[0]
        interval = data_interval
    elif batch_indices is not None:
        interval_size = len(batch_indices)
        interval = [batch_indices[0], batch_indices[-1]]
    else:
        raise Exception("There is a contradiction. Or data_interval or batch_indices must be provided.")

    if data_interval is not None:

        if interval_size < batch_size:
            batches_ = [interval[0], interval[1]]
            batches_ = np.array(batches_)
        else:
            # divides data_interval in the maximum amount of pieces such that the individual batches >= batch_size
            # and the batch_sizes differ at maximum by 1 in size

            n_batches = floor(interval_size / batch_size)
            residual = interval_size % batch_size
            batch_size_plus = floor(residual / n_batches)
            batch_size_plus_residual = residual % n_batches

            batch_size_up = batch_size+batch_size_plus

            batches_ = [interval[0]] + [batch_size_up+1] * batch_size_plus_residual + [batch_size_up] * (n_batches - batch_size_plus_residual)
            batches_ = np.cumsum(batches_)

        batches = [batches_[i:i + 2]
                   for i in range(batches_.shape[0] - 1)]
    else:
        if interval_size < batch_size:
            batches_ = batch_indices
            batches_ = np.array(batches_)
        else:
            # divides data_interval in the maximum amount of pieces such that the individual batches >= batch_size
            # and the batch_sizes differ at maximum by 1 in size

            n_batches = floor(interval_size / batch_size)
            batches_ = np.array_split(batch_indices, n_batches, axis=0)

        batches = [item.tolist() for item in batches_]

    return batches

def indices_batchdomain_constructor(indices:list=None, batch_size:int=None) -> list:

    interval_size = indices.shape[0]

    n_batches = ceil(interval_size / batch_size)
    batches_ = np.array_split(indices, n_batches)

    batches = [batch.tolist() for batch in batches_]

    return batches
