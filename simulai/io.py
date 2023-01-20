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

import sys
import numpy as np
import torch
import h5py
import random
from typing import Union, Dict, List
from torch import Tensor
from numpy.lib import recfunctions

from simulai.batching import batchdomain_constructor
from simulai.metrics import MemorySizeEval
from simulai.abstract import DataPreparer
from simulai.batching import indices_batchdomain_constructor
from simulai.abstract import Dataset


"""
    BEGIN DataPreparer children
"""

# This class does nothing
class ByPassPreparer(DataPreparer):

    """
    ByPass class, it fills the DataPreparer blank, but does nothing.
    """
    name = "no_preparer"

    def __init__(self, channels_last:bool=False) -> None:

        super().__init__()

        self.channels_last = channels_last
        self.collapsible_shapes = None
        self.dtype = None

    def prepare_input_data(self, data:np.ndarray) -> np.ndarray:

        self.collapsible_shapes = data.shape[1:]

        return data

    def prepare_output_data(self, data:np.ndarray) -> np.ndarray:

        return data

    def prepare_input_structured_data(self, data:np.recarray) -> np.ndarray:

        return data

    def prepare_output_structured_data(self, data:np.ndarray) -> np.recarray:

        return data

# It is used for high-dimensional datasets, in order to convert them to
# 2D ones
class Reshaper(DataPreparer):

    name = "reshaper"

    def __init__(self, channels_last:bool=False) -> None:

        """

        Reshaper converts n-dimensional arrays to two-dimensional ones, performing a
        simple reshaping operation F: (n0, n1, ..., nm) -> (n0, prod(n1, ..., nm))

        """

        super().__init__()

        # Nomenclature: shape = (immutable_shape, *collapsible_shapes)
        self.channels_last = channels_last
        self.collapsible_shapes = None
        self.collapsed_shape = None
        self.dtype = None
        self.n_features = None

    def _set_shapes_from_data(self, data:np.ndarray=None) -> None:

        self.collapsible_shapes = data.shape[1:]
        self.collapsed_shape = np.prod(self.collapsible_shapes).astype(int)
        self._is_recarray = data.dtype.names is not None
        if self._is_recarray:
            self.n_features = len(data.dtype.names)*self.collapsed_shape
        else:
            self.n_features = self.collapsed_shape

    def _prepare_input_data(self, data:np.ndarray=None) -> np.ndarray:

        assert len(data.shape) > 1, 'Error! data must have at least two dimensions'
        return data.reshape((data.shape[0], self.n_features))

    def prepare_input_data(self, data:Union[np.ndarray, np.recarray]) -> np.ndarray:

        self._set_shapes_from_data(data)
        if self._is_recarray:
            return self._prepare_input_structured_data(data)
        else:
            return self._prepare_input_data(data)

    def _reshape_to_output(self, data:np.ndarray) -> np.ndarray:

        return data.reshape((data.shape[0],) + self.collapsible_shapes)

    def _prepare_output_data(self, data:np.ndarray=None, single:bool=False) -> np.ndarray:

        if self._is_recarray:
            return self._prepare_output_structured_data(data)
        else:
            return self._reshape_to_output(data)

    def prepare_output_data(self, data:np.ndarray, single:bool=False) -> np.ndarray:

        return self._prepare_output_data(data)

    def _prepare_input_structured_data(self, data:np.recarray=None) -> np.ndarray:

        self.dtype = data.dtype
        self._set_shapes_from_data(data)
        data_ = recfunctions.structured_to_unstructured(data)
        reshaped_data_ = self._prepare_input_data(data_)
        return reshaped_data_

    def prepare_input_structured_data(self, data:np.recarray=None) -> np.ndarray:

        return self._prepare_input_structured_data(data)

    def prepare_output_structured_data(self, data:np.ndarray=None) -> np.recarray:

        return self._prepare_output_structured_data(data)

    def _prepare_output_structured_data(self, data:np.ndarray=None) -> np.recarray:

        data = data.reshape((data.shape[0], ) + self.collapsible_shapes + (len(self.dtype), ))
        output_data = recfunctions.unstructured_to_structured(data, self.dtype)
        output_data = self._reshape_to_output(output_data)

        return output_data


# It is used for high-dimensional datasets, in order to scale and convert them to
# 2D ones
class ScalerReshaper(Reshaper):

    name = "scalerreshaper"

    def __init__(self, bias:float=0., scale:float=1., channels_last:bool=False) -> None:

        """

        Reshaper converts n-dimensional arrays to two-dimensional ones, performing a
        simple reshaping operation F: (n0, n1, ..., nm) -> (n0, prod(n1, ..., nm))

        """

        super().__init__(channels_last=channels_last)

        # Nomenclature: shape = (immutable_shape, *collapsible_shapes)
        self.bias = bias
        self.scale = scale

    def prepare_input_data(self, data:Union[np.ndarray, np.recarray]=None, *args, **kwargs) -> np.ndarray:

        if data.dtype.names is None:
            return super(ScalerReshaper, self).prepare_input_data((data-self.bias)/self.scale, *args, **kwargs)
        else:
            return self.prepare_input_structured_data(data, *args, **kwargs)

    def prepare_output_data(self, data:Union[np.ndarray, np.recarray]=None, *args, **kwargs) -> np.ndarray:
        if not self._is_recarray:
            return super(ScalerReshaper, self).prepare_output_data(data*self.scale+self.bias, *args, **kwargs)
        else:
            return self.prepare_output_structured_data(data)

    def _get_structured_bias_scale(self, dtype:np.dtype=None) -> (float, float):

        bias = self.bias
        if isinstance(self.bias, float):
            bias = {n: self.bias for n in dtype.names}
        scale = self.scale
        if isinstance(self.scale, float):
            scale = {n: self.scale for n in dtype.names}

        return bias, scale

    def prepare_input_structured_data(self, data:np.recarray=None, *args, **kwargs) -> np.ndarray:

        bias, scale = self._get_structured_bias_scale(data.dtype)

        data = data.copy()
        names = data.dtype.names
        for name in names:
            data[name] = (data[name]-bias[name])/scale[name]

        return super(ScalerReshaper, self).prepare_input_structured_data(data, *args, **kwargs)

    def prepare_output_structured_data(self, data:np.ndarray=None, *args, **kwargs) -> np.recarray:

        bias, scale = self._get_structured_bias_scale(self.dtype)
        data = super(ScalerReshaper, self).prepare_output_structured_data(data, *args, **kwargs)
        data = data.copy()
        for name in self.dtype.names:
            data[name] = data[name]*scale[name]+bias[name]
        return data


# It is used for datasets in which there are invalid data
class MapValid(Reshaper):

    """
    MapValid converts n-dimensional arrays to two-dimensional ones performing a valid values
    mapping operation F: F: data.shape = (n0, n1, ..., nm) -> data'.shape = (n0, n_valids)
    n_valids = dim([k in data[0, ...] if k != mask])
    WARNING: the invalid positions are expected to be static in relation to n0.
    """

    name = "map_valid"

    def __init__(self, config:dict=None, mask=None, channels_last:bool=True) -> None:

        """

        :param config: configurations dictionary
        :type config: dict
        :param mask:  mask to select the invalid values
        :type: int, np.NaN or np.inf

        """

        super().__init__()

        # Some default parameters which can be
        # overwritten by the config content

        self.default_dtype = 'float64'

        if mask == 0 or isinstance(mask, int):
            self.replace_mask_with_large_number = False
        else:
            self.replace_mask_with_large_number = True

        self.return_the_same_mask = True

        for key, value in config.items():
            setattr(self, key, value)

        # Default value for very large numbers
        self.large_number = 1e15

        if not mask or self.replace_mask_with_large_number:
            self.mask = self.large_number
        else:
            self.mask = mask

        self.mask_ = mask

        for key, value in config.items():
            setattr(self, key, value)

        self.valid_indices = None
        self.original_dimensions = None

        self.channels_last = channels_last

    def prepare_input_data(self, data:np.ndarray=None) -> np.ndarray:

        """

        Internal input data preparer, executed for each label of the structured array

        :param data:
        :type data: np.ndarray
        :return:
        :rtype: np.ndarray

        """

        data = super(MapValid, self).prepare_input_data(data)

        if self.mask == self.large_number:
            self.valid_indices_ = np.where(data[0, ...] < self.mask)

        elif not str(self.mask).isnumeric() or isinstance(self.mask, int):
            self.valid_indices_ = np.where(data[0, ...] != self.mask)

        else:
            raise Exception("The chosen mask {} does not fit in any supported case".format(self.mask))

        samples_dim = data.shape[0]

        valid_indices = (slice(0, samples_dim), ) + self.valid_indices_

        return data[valid_indices]

    def prepare_output_data(self, data:np.ndarray=None) -> np.ndarray:

        """

        Internal output.py data preparer, executed for each label of
        the structured array

        :param data:
        :type data: np.ndarray
        :return:
        :rtype: np.ndarray

        """

        immutable_shape = data.shape[0]

        final_shape = (immutable_shape, self.n_features, )

        if self.return_the_same_mask:
            mask = self.mask_
        else:
            mask = np.NaN  # For practical purposes
        reshaped_data = np.full(final_shape, mask)

        if not reshaped_data.dtype.type == self.default_dtype:
            reshaped_data = reshaped_data.astype(self.default_dtype)

        samples_dim = data.shape[0]
        valid_indices = (slice(0, samples_dim),) + self.valid_indices_

        reshaped_data[valid_indices] = data

        reshaped_data = super(MapValid, self).prepare_output_data(reshaped_data)

        return reshaped_data

    def prepare_input_structured_data(self, data:np.recarray=None) -> np.ndarray:

        """

        :param data: structured array to be prepared, the default shape is
        (n_samples, 1, *other_dimensions)
        The features dimensions is 1 in NumPy structured arrays
        :type data: np.ndarray
        :return: np.ndarray

        """

        return self.prepare_input_data(data)

    def prepare_output_structured_data(self, data:np.ndarray=None) -> np.ndarray:

        """

        :param data: np.ndarray
        :return: np.ndarray

        """
        return self.prepare_output_data(data)


# Sampling (and, optionally, shuffling) datasets
class Sampling(DataPreparer):

    name = "sampling"

    def __init__(self, choices_fraction:float=0.1, shuffling:bool=False) -> None:

        super().__init__()

        self.choices_fraction = choices_fraction
        self.shuffling = shuffling

        self.global_indices = None
        self.sampled_indices = None

    @property
    def indices(self) -> list:

        assert self.sampled_indices is not None,\
        "The indices still were not generate." \
        "Run prepare_input_data or prepare_input_structured_data for getting them."

        return sorted(self.sampled_indices.tolist())

    def prepare_input_data(self, data:np.ndarray=None, data_interval:list=None) -> np.ndarray:

        if data_interval is None:
            data_interval = [0, data.shape[0]]
        n_samples = data_interval[1] - data_interval[0]

        self.global_indices = np.arange(start=data_interval[0], stop=data_interval[1])

        n_choices = int(self.choices_fraction*n_samples)

        self.sampled_indices = self.global_indices.copy()
        if self.shuffling:
            np.random.shuffle(self.sampled_indices)
        else:
            self.sampled_indices = self.sampled_indices

        self.sampled_indices = np.random.choice(self.sampled_indices, n_choices)

        return data[self.sampled_indices]

    def prepare_input_structured_data(self, data:h5py.Dataset=None, data_interval:list=None,
                                            batch_size:int=None, dump_path:str=None) -> np.recarray:

        """
        :param data: structured array to be prepared, the default shape is
        (n_samples, 1, *other_dimensions)
        The features dimensions is 1 in NumPy structured arrays
        :type data: np.ndarray
        :return: np.ndarray
        """

        if data_interval is None:
            data_interval = [0, data.shape[0]]

        n_samples = data_interval[1] - data_interval[0]
        self.global_indices = np.arange(start=data_interval[0], stop=data_interval[1])

        n_sampled_preserved = int(self.choices_fraction*n_samples)
        self.sampled_indices = np.random.choice(self.global_indices, n_sampled_preserved, replace=False)

        if isinstance(data, h5py.Dataset):

            if isinstance(batch_size, MemorySizeEval):
                batch_size = batch_size(max_batches=n_sampled_preserved, shape=data.shape[1:])
            else:
                pass

            assert dump_path, "Using a h5py.Dataset as input data a dump_path must be provided."

            fp = h5py.File(dump_path, 'w')
            sampled_data = fp.create_dataset('data', shape=(n_sampled_preserved,) + data.shape[1:],
                                             dtype=data.dtype)

            # Constructing the normalization  using the reference data
            batches = indices_batchdomain_constructor(indices=self.sampled_indices, batch_size=batch_size)

            start_ix = 0
            for batch_id, batch in enumerate(batches):
                print(f"Sampling batch {batch_id+1}/{len(batches)} batch_size={len(batch)}")
                finish_ix = start_ix + len(batch)
                sampled_data[start_ix:finish_ix] = data[sorted(batch)]
                start_ix = finish_ix

            if self.shuffling:
                random.shuffle(sampled_data)

        else:
            raise Exception("Others cases are still not implemented.")

        return sampled_data

"""
    END DataPreparer and its children
"""

"""
    BEGIN Time-series data preparation (MovingWindow, SlidingWindow and BatchExtrapolation)
"""

class MovingWindow:

    """

    MovingWindow is applied over a time-series array (2D array), and it is used for
    creating the necessary augmented data used for LSTM networks, replicating the training
    windows for each sample in the dataset.

    See a graphical example:

    batch n
    ---------|---
    history  | horizon

        batch n+1
        ---------|---
        history  | horizon
    ----
    skip

    """

    def __init__(self, history_size:int=None, skip_size:int=1, horizon_size:int=None, full_output:bool=True) -> None:

        """

        :param history_size: number of history samples
        :type history_size: int
        :param skip_size: number of samples to skip between two windows
        :type skip_size: int
        :param horizon_size: number of samples to use as prediction horizon
        :type horizon_size: int

        """

        self.history_size = history_size
        self.skip_size = skip_size
        self.horizon_size = horizon_size
        self.full_output = full_output

        if self.full_output == True:
            self.process_batch = self.bypass
        else:
            self.process_batch = self.get_last_item

        # Verifying if history and horizon sizes was provided
        assert history_size, f"A value for history_size must be provided, not {history_size}"
        assert horizon_size, f"A value for horizon_size must be provided, not {horizon_size}"

    def get_last_item(self, batch:np.ndarray) -> np.ndarray:

        return batch[-1:]

    def bypass(self, batch:np.ndarray) -> np.ndarray:

        return batch

    def __call__(self, input_data:np.ndarray=None, output_data:np.ndarray=None) -> (np.ndarray, np.ndarray):

        """

        :param input_data: 2D array (time-series) to be used for constructing the history size
        :type input_data: np.ndarray
        :param output_data:
        :type output_data: np.ndarray
        :return: (np.ndarray, np.ndarray) with shapes
        (n_samples, n_history, n_features) and (n_samples, n_horizon, n_features)

        """

        # It is expected series_data to be a set of time-series with shape
        # (n_timesteps, n_variables)

        input_batches_list = list()
        output_batches_list = list()
        data_size = input_data.shape[0]

        assert input_data.shape[0] == output_data.shape[0]

        center = self.history_size

        # Loop for covering the entire time-series dataset constructing the
        # training windows
        while center + self.horizon_size <= data_size:

            input_batch = input_data[center - self.history_size:center, :]
            output_batch = output_data[center:center + self.horizon_size, :]

            input_batches_list.append(input_batch)
            output_batches_list.append(self.process_batch(batch=output_batch))

            center += self.skip_size

        input_data = np.stack(input_batches_list, 0)
        output_data = np.stack(output_batches_list, 0)

        return input_data, output_data

class SlidingWindow:

    """

    SlidingWindow is applied over a time-series array (2D array), and it is used for
    creating the necessary augmented data used for LSTM networks, replicating the training
    windows for each sample in the dataset. The difference between SlidingWindow and MovingWindow
    is that here there is no intersection between two sequential batches

    See an graphical example:

    batch n
    ---------|---
    history  | horizon

                      batch n+1
                      ---------|---
                      history  | horizon

    """

    def __init__(self, history_size:int=None, skip_size:int=None) -> None:

        """

        :param history_size: number of history samples
        :type history_size: int
        :param skip_size: number of samples to skip between two windows
        :type skip_size: int

        """

        self.history_size = history_size
        self.skip_size = skip_size

        # Verifying if history and horizon sizes was provided
        assert history_size, f"A value for history_size must be provided, not {history_size}"
        assert skip_size, f"A value for horizon_size must be provided, not {skip_size}"

    def __call__(self, input_data:np.ndarray=None, output_data:np.ndarray=None) -> (np.ndarray, np.ndarray):

        """

        :param input_data: 2D array (time-series) to be used for constructing the history size
        :type input_data: np.ndarray
        :param output_data:
        :type output_data: np.ndarray
        :return: (np.ndarray, np.ndarray) with shapes
        (n_samples, n_history, n_features) and (n_samples, n_horizon, n_features)

        """

        # It is expected series_data to be a set of time-series with shape
        # (n_timesteps, n_variables)

        input_batches_list = list()
        output_batches_list = list()
        data_size = input_data.shape[0]

        assert input_data.shape[0] == output_data.shape[0]

        center = self.history_size

        # Loop for covering the entire time-series dataset constructing the
        # training windows
        while center + self.skip_size <= data_size:

            input_batch = input_data[center - self.history_size:center, :]
            output_batch = output_data[center - self.history_size + self.skip_size:
                                       center+self.skip_size, :]

            input_batches_list.append(input_batch)
            output_batches_list.append(output_batch)

            center += self.skip_size

        input_data = np.stack(input_batches_list, 0)
        output_data = np.stack(output_batches_list, 0)

        return input_data, output_data

class IntersectingBatches:

    """

    IntersectingBatches is applied over a time-series array (2D array).

    See a graphical example:

    batch n
    |------------|
        batch n+1
        |------------|
    ----
    skip

    """

    def __init__(self, skip_size:int=1, batch_size:int=None, full:bool=True) -> None:

        """

        :param skip_size: number of samples to skip between two windows
        :type skip_size: int
        :param batch_size: number of samples to use in each batch
        :type batch_size: int

        """

        # Verifying if history and horizon sizes was provided
        assert batch_size, f"A value for horizon_size must be provided, not {batch_size}"

        self.skip_size = skip_size
        self.batch_size = batch_size
        self.full = full

    def get_indices(self, dim:int=None) -> np.ndarray:

        """

        It gets just the indices of the shifting

        :param dim: total dimension
        :type dim: int
        :return: the shifted indices
        :rtype: np.ndarray

        """

        center = 0
        indices = list()
        indices_m = list()

        # Loop for covering the entire time-series dataset constructing the
        # training windows
        while center + self.batch_size < dim:

            index = center + self.batch_size

            indices.append(center)
            indices_m.append(index)

            center += self.skip_size

        return np.array(indices), np.array(indices_m)

    def __call__(self, input_data:np.ndarray=None) -> Union[list, np.ndarray]:

        """

        :param input_data: 2D array (time-series) to be used for constructing the history size
        :type input_data: np.ndarray
        :param output_data:
        :type output_data: np.ndarray
        :return: (np.ndarray, np.ndarray) with shapes
        (n_samples, n_history, n_features) and (n_samples, n_horizon, n_features)

        """

        # It is expected series_data to be a set of time-series with shape
        # (n_timesteps, n_variables)

        input_batches_list = list()

        data_size = input_data.shape[0]

        center = 0

        # Loop for covering the entire time-series dataset constructing the
        # training windows
        while center + self.batch_size <= data_size:

            input_batch = input_data[center:center + self.batch_size]

            input_batches_list.append(input_batch)

            center += self.skip_size

        if self.full == True:
            return input_batches_list
        else:
            return np.vstack([item[-1] for item in input_batches_list])


class BatchwiseExtrapolation:

    """

    BatchwiseExtraplation uses a time-series regression model and inputs as generated by
    MovingWindow to continuously extrapolate a dataset

    """

    def __init__(self, op:callable=None, auxiliary_data:np.ndarray=None) -> None:

        self.op = op
        self.auxiliary_data = auxiliary_data
        self.time_id = 0

    def _simple_extrapolation(self, extrapolation_dataset:np.ndarray, history_size:int=0) -> np.ndarray:

        return extrapolation_dataset[None, -history_size:, :]

    def _forcing_extrapolation(self, extrapolation_dataset:np.ndarray, history_size:int=0) -> np.ndarray:

        return np.hstack([extrapolation_dataset[-history_size:, :],
                          self.auxiliary_data[self.time_id-history_size:self.time_id, :]])[None, :, :]

    def __call__(self, init_state:np.ndarray=None, history_size:int=None, horizon_size:int=None,
                       testing_data_size:int=None) -> np.ndarray:

        if isinstance(self.auxiliary_data, np.ndarray):
            n_series = self.auxiliary_data.shape[-1]
        else:
            n_series = 0

        current_state = init_state
        extrapolation_dataset = init_state[0, :, n_series:]
        self.time_id = history_size

        if isinstance(self.auxiliary_data, np.ndarray):

            assert self.auxiliary_data.shape[-1] + n_series == init_state.shape[-1], \
                   "Number of series in the initial state must be {}".format(self.auxiliary_data.shape[-1])

            current_state_constructor = self._forcing_extrapolation

        else:

            current_state_constructor = self._simple_extrapolation

        while extrapolation_dataset.shape[0] - history_size + horizon_size <= testing_data_size:

            extrapolation = self.op(current_state)
            extrapolation_dataset = np.concatenate([extrapolation_dataset, extrapolation[0]], 0)
            current_state = current_state_constructor(extrapolation_dataset,
                                                      history_size=history_size)

            log_str = "Extrapolation {}".format(self.time_id + 1 - history_size)
            sys.stdout.write("\r" + log_str)
            sys.stdout.flush()

            self.time_id += horizon_size

        extrapolation_dataset = extrapolation_dataset[history_size:, :]

        return extrapolation_dataset

"""
    END Time-series data preparation (MovingWindow, SlidingWindow and BatchExtrapolation)
"""

class BatchCopy:

    def __init__(self, channels_last:bool=False) -> None:

        self.channels_last = channels_last

    def _single_copy(self, data:h5py.Dataset=None, data_interval:list=None,
                           batch_size:int=None, dump_path:str=None,
                           transformation:callable=lambda data: data) -> h5py.Dataset:

        assert isinstance(data, h5py.Dataset), "The input must be h5py.Dataset"

        variables_list = data.dtype.names
        data_shape = (data_interval[1] - data_interval[0],) + data.shape[1:]

        data_file = h5py.File(dump_path, "w")
        dtype = [(var, '<f8') for var in variables_list]

        dset = data_file.create_dataset("data", shape=data_shape,
                                        dtype=dtype)

        if isinstance(batch_size, MemorySizeEval):
            n_samples = data_interval[1] - data_interval[0]
            batch_size = batch_size(max_batches=n_samples, shape=data.shape[1:])
        else:
            pass

        # Constructing the normalization  using the reference data
        batches = batchdomain_constructor(data_interval, batch_size)
        dset_batches = batchdomain_constructor([0, dset.shape[0]], batch_size)

        variables_names = data.dtype.names

        n_variables = len(data.dtype.names)

        for batch_id, (batch, d_batch) in enumerate(zip(batches, dset_batches)):

            print(f"Copying batch {batch_id+1}/{len(batches)} batch_size={batch[1]-batch[0]}")

            # The variables dimension is the last one
            if self.channels_last:
                # TODO this is a restrictive way of doing it. It must be more flexible.
                chunk_data = data[slice(*batch)].view((float, len(data.dtype.names)))#.transpose((0, 4, 2, 3, 1))
            # The variables dimension is the second one
            else:
                chunk_data = data[slice(*batch)].view((float, len(data.dtype.names)))

            chunk_data = np.core.records.fromarrays(np.split(chunk_data[...], n_variables, axis=-1),
                                                               names=variables_names,
                                                               formats=','.join(len(variables_names) * ['f8']))

            if len(chunk_data.shape) > len(dset.shape):
                chunk_data = np.squeeze(chunk_data, axis=-1)
            else:
                pass

            dset[slice(*d_batch)] = transformation(chunk_data[...])

        return dset

    def _multiple_copy(self, data:list=None, data_interval:list=None,
                             batch_size:int=None, dump_path:str=None,
                             transformation:callable=lambda data: data) -> h5py.Dataset:

        assert all([isinstance(di, h5py.Dataset) for di in data]), "All inputs must be h5py.Dataset"

        variables_list = sum([list(di.dtype.names) for di in data], [])
        data_shape = (data_interval[1] - data_interval[0],) + data[0].shape[1:]

        data_file = h5py.File(dump_path, "w")
        dtype = [(var, '<f8') for var in variables_list]

        dset = data_file.create_dataset("data", shape=data_shape,
                                        dtype=dtype)

        if isinstance(batch_size, MemorySizeEval):
            n_samples = data_interval[1] - data_interval[0]
            batch_size = batch_size(max_batches=n_samples, shape=data.shape[1:])
        else:
            pass

        # Constructing the normalization  using the reference data
        batches = batchdomain_constructor(data_interval, batch_size)
        dset_batches = batchdomain_constructor([0, dset.shape[0]], batch_size)

        variables_names = sum([list(di.dtype.names) for di in data], [])

        n_variables = sum([len(di.dtype.names) for di in data])

        for batch_id, (batch, d_batch) in enumerate(zip(batches, dset_batches)):

            print(f"Copying and concatenating the batches {batch_id+1}/{len(batches)} batch_size={batch[1] - batch[0]}")

            # The variables dimension is the last one
            if self.channels_last:
                # TODO this is a restrictive way of doing it. It must be more flexible.
                chunk_data = np.stack([di[slice(*batch)].view((float, len(di.dtype.names))).transpose((0, 4, 2, 3, 1))
                                       for di in data], axis=-1)
            # The variables dimension is the second one
            else:
                chunk_data = np.stack([di[slice(*batch)].view((float, len(di.dtype.names))) for di in data], axis=-1)

            chunk_data = np.core.records.fromarrays(np.split(chunk_data[...], n_variables, axis=-1),
                                                    names=variables_names,
                                                    formats=','.join(len(variables_names) * ['f8']))

            if len(chunk_data.shape) > len(dset.shape):
                chunk_data = np.squeeze(chunk_data, axis=-1)
            else:
                pass

            dset[slice(*d_batch)] = transformation(chunk_data[...])

        return dset

    def copy(self, data:h5py.Dataset=None, data_interval:list=None,
                   batch_size:int=None, dump_path:str=None,
                   transformation:callable=lambda data: data) -> h5py.Dataset:

        if isinstance(data, list):

            return self._multiple_copy(data=data, data_interval=data_interval, batch_size=batch_size,
                                       dump_path=dump_path, transformation=transformation)

        else:
            return self._single_copy(data=data, data_interval=data_interval, batch_size=batch_size,
                                       dump_path=dump_path, transformation=transformation)

class MakeTensor:

    def __init__(self, input_names=None, output_names=None):

        self.input_names = input_names
        self.output_names = output_names

    def _make_tensor(self, input_data:np.ndarray=None, device:str='cpu') -> List[torch.Tensor]:

        inputs_list = list(torch.split(input_data, 1, dim=-1))

        for vv, var in enumerate(inputs_list):

            var.requires_grad = True
            var = var.to(device)
            inputs_list[vv] = var
            #var = var[..., None]

        return inputs_list

    def _make_tensor_dict(self, input_data:dict=None, device:str='cpu') -> dict:

        inputs_dict = dict()

        for key, item in input_data.items():

            item.requires_grad = True
            item = item.to(device)
            inputs_dict[key] = item

        return inputs_dict

    def __call__(self, input_data:Union[np.ndarray, torch.Tensor,
                                        Dict[str, np.ndarray]]=None,
                                         device:str='cpu') -> List[torch.Tensor]:

        if type(input_data) == np.ndarray:

            input_data = torch.from_numpy(input_data.astype(np.float32))

            inputs_list = self._make_tensor(input_data=input_data, device=device)

            return inputs_list

        if type(input_data) == torch.Tensor:

            inputs_list = self._make_tensor(input_data=input_data, device=device)

            return inputs_list

        elif type(input_data) == dict:

            inputs_list = self._make_tensor_dict(input_data=input_data, device=device)

            return inputs_list

        else:
            raise Exception(f"The type {type(input_data)} for input_data is not supported.")

class GaussianNoise(Dataset):

    def __init__(self, stddev:float=0.01, input_data:Union[np.ndarray, Tensor]=None):

        super(Dataset, self).__init__()

        self.stddev = stddev

        if isinstance(input_data, np.ndarray):
            input_data_ = torch.from_numpy(input_data.astype("float32"))
        else:
            input_data_ = input_data

        self.input_data = input_data_

        self.data_shape = tuple(self.input_data.shape)

    def size(self):

        return self.data_shape

    def __call__(self):

        return (1 + self.stddev*torch.randn(*self.data_shape))*self.input_data



