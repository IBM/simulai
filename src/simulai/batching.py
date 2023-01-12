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
from typing import List, Tuple

# Sampling batches from disk
class BatchwiseSampler:

    def __init__(self, dataset:h5py.Group=None,
                       input_variables:List[str]=None, target_variables:List[str]=None,
                       input_normalizer:callable=None, target_normalizer:callable=None,
                       channels_first:bool=None) -> None:
        """
        Batchwise sampler for loading samples from disk and apply normalization if needed.
        
        Parameters:
        -----------
            dataset : h5py.Group
                Dataset object containing the samples
            input_variables : List[str]
                List of input variables to be loaded
            target_variables : List[str]
                List of target variables to be loaded
            input_normalizer : Callable
                Function to be applied on the input variables
            target_normalizer : Callable
                Function to be applied on the target variables
            channels_first : bool
                Whether the data should be in channels_first format or not. If not provided,
                will be set to None.
        """

        # This import avoids circular importing
        from simulai.metrics import MinMaxEvaluation

        self.dataset = dataset
        self.input_variables = input_variables
        self.target_variables = target_variables

        self.input_normalizer = input_normalizer
        self.target_normalizer = target_normalizer

        self.channels_first = channels_first

        if self.channels_first:
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
    def minmax(self, batch_size: int = None, data_interval: list = None) -> Tuple[float, float]:
         """
         Evaluate the minimum and maximum values of all the target variables in the dataset.

         Parameters:
         -----------
             batch_size : int
                 Number of samples to use in the evaluation
             data_interval : list
                 List of 2 integers representing the starting and ending indexes of the interval in which
                 the values will be evaluated.
                 
         Returns:
         --------
             A tuple of minimum and maximum value of the target variables.
        """
         min_list = []
         max_list = []

         for k in self.target_variables:
                min, max = self.minmax_eval(dataset=self.dataset[k], batch_size=batch_size,
                                            data_interval=data_interval)
                min_list.append(min)
                max_list.append(max)

         return  np.min(min_list), np.max(max_list)


    def input_shape(self) -> list:
        """
         Get the input shape of the dataset. The shape will be adjusted to put the channels dimension first
         if 'channels_first' is True.
         
         Returns:
         --------
             A list of integers representing the shape of the input variables.
        """
        if self.channels_first:
            shape_ = self.dataset[self.input_variables[0]].shape
            shape = (shape_[0],) + (len(self.input_variables),) + shape_[1:]
        else:
            shape = self.dataset[self.input_variables[0]].shape + (len(self.input_variables),)

        return list(shape)

    def _normalization_bypass(self, data:np.ndarray=None) -> np.ndarray:
        """
         Bypass the normalization.
         
         Parameters:
         -----------
             data : np.ndarray
                 The data to be bypassed.
                 
         Returns:
         --------
             Same data
         """
        return data

    def _target_normalization(self, data:np.ndarray=None) -> np.ndarray:
        """
         Normalize the target data using the provided normalizer.
         
         Parameters:
         -----------
             data : np.ndarray
                 The target data to be normalized.
                 
         Returns:
         --------
             Normalized target data.
        """
        return self.target_normalizer(data=data)

    def _input_normalization(self, data: np.ndarray = None) -> np.ndarray:
        """
         Normalize the input data using the provided normalizer.
         
         Parameters:
         -----------
             data : np.ndarray
                 The input data to be normalized.
                 
         Returns:
         --------
             Normalized input data.
        """
        return self.input_normalizer(data=data)

    def _transpose_first_channel(self, variables_list: list = None) -> torch.Tensor:
        """
         Transpose the first channel of the variables list.
         
         Parameters:
         -----------
             variables_list : list
                 The list of variables to be transposed.
                 
         Returns:
         --------
             A torch tensor of transposed variables.
        """
        batch = np.stack(variables_list, axis=-1)

        dims = list(range(len(batch.shape)))
        dims_t = [0] + [dims[-1]] + dims[1:-1]

        batch = batch.transpose(*dims_t)

        return torch.from_numpy(batch.astype('float32'))

    def _simple_stack(self, variables_list: list = None) -> torch.Tensor:
        """
         Stack the variables list along the last axis.
         
         Parameters:
         -----------
             variables_list : list
                 The list of variables to be stacked.
                 
         Returns:
         --------
             A torch tensor of stacked variables.
        """
        batch = np.stack(variables_list, dim=-1)

        return torch.from_numpy(batch.astype('float32'))

    def input_data(self, indices:np.ndarray=None) -> torch.Tensor:
        """
         Retrieve the input data for the given indices, apply normalization and adjust the dimension
         
         Parameters:
         -----------
             indices : np.ndarray
                 The indices of samples for which the input data should be retrieved
                 
         Returns:
         --------
             A torch tensor of input data
        """
        indices = np.sort(indices)

        variables_arr = [self.dataset[i][indices] for i in self.input_variables]

        return self.exec_input_normalization(self.adjust_dimension(variables_list = variables_arr))

    def target_data(self, indices:np.ndarray=None) -> torch.Tensor:
        """
         Retrieve the target data for the given indices, apply normalization and adjust the dimension
         
         Parameters:
         -----------
             indices : np.ndarray
                 The indices of samples for which the target data should be retrieved
                 
         Returns:
         --------
             A torch tensor of target data
        """

        indices = np.sort(indices)

        variables_arr = [torch.from_numpy(self.dataset[i][indices].astype('float32')) for i in self.target_variables]

        return self.exec_target_normalization(self.adjust_dimension(variables_list = variables_arr))

def batchdomain_constructor(data_interval:list=None, batch_size:int=None, batch_indices:list=None) -> list:
    """
     Create a list of indices of the input data in the form of batches, using either an interval or a list of indices.

     Parameters:
     -----------
         data_interval : list
             A list of two integers representing the start and end of the data interval.
         batch_size : int
             The desired size of the batches
         batch_indices : list
             A list of indices to be divided into batches.

     Returns:
     --------
         A list of lists containing the indices of the input data in the form of batches.
     """

    if data_interval is not None:
        interval_size = data_interval[1] - data_interval[0]
        interval = data_interval
    elif batch_indices is not None:
        interval_size = len(batch_indices)
        interval = [batch_indices[0], batch_indices[-1]]
    else:
        raise Exception("Either data_interval or batch_indices must be provided.")

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

        batches = [batches_[i:i + 2]for i in range(batches_.shape[0] - 1)]
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
    """
     Create a list of batches of indices.
     
     Parameters:
     -----------
         indices : list
             A list of indices to be divided into batches.
         batch_size : int
             The desired size of the batches.

     Returns:
     --------
         A list of lists containing the indices of the input data in the form of batches.
    """
    interval_size = indices.shape[0]

    n_batches = ceil(interval_size / batch_size)
    batches_ = np.array_split(indices, n_batches)

    batches = [batch.tolist() for batch in batches_]

    return batches