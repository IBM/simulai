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

import copy
import sys
import h5py
import psutil
import numpy as np
import torch
import dask.array as da
from typing import Union

from simulai.math.integration import RK4
from simulai.abstract import DataPreparer
from simulai.batching import batchdomain_constructor

class ByPass:

    name = "no_metric"

    def __init__(self):
        pass

class L2Norm:

    name = "l2_norm"

    def __init__(self, mask:Union[str, float, None]=None, do_clean_data:bool=True,
                       large_number:float=1e15, default_data:float=0.0) -> None:

        """It evaluates a L^2 norm comparing an approximation and its reference value
        :param mask: if there are masked or missing data, it informs what kind of value is filling these gaps
        :type mask: Union[str, float, None]
        :param do_clean_data: Execute a data cleaning (removing large numbers and NaN) or not ?
        :type do_clean_data: bool
        :returns: nothing
        """

        self.large_number = large_number
        self.mask = mask
        self.do_clean_data = do_clean_data
        self.default_data = default_data

    def _clean_nan_and_large_single(self, d:np.ndarray) -> np.ndarray:

        """ It removes NaNs and large numbers from a single array
        :param d: data to be cleaned
        :type d: np.ndarray
        :returns: the data cleaned
        :rtype: np.ndarray
        """

        if self.mask is not None:
            is_mask = d == self.mask
            if np.any(is_mask):
                d = np.where(is_mask, self.default_data, d)
                print(f"There are values equal to the mask=={self.mask} in the dataset. They will be replaced with {self.default_data}.")

        isnan_data = np.isnan(d)
        if np.any(isnan_data):
            d = np.where(isnan_data, self.default_data, d)
            print(f"There are NaN's in the dataset. They will be replaced with {self.default_data}.")

        is_large = np.abs(d) >= self.large_number
        if np.any(is_large):
            d = np.where(is_large, self.default_data, d)
            print(f"There are large numbers, i.e., abs(x) > {self.large_number}, in the dataset. They will be replaced with {self.default_data}.")

        return d

    def _clean_nan_and_large(self, data:np.ndarray, reference_data:np.ndarray) -> (np.ndarray, np.ndarray):

        """It removes NaNs and large number of the input and the reference dataset
        :param data: the data to be evaluated in the norm
        :type data: np.ndarray
        :param reference_data: the data to used as comparison
        :type reference_data: np.ndarray
        :returns: both the datasets after have been cleaned
        :rtype: (np.ndarray, np.ndarray)
        """

        if self.do_clean_data:
            data = self._clean_nan_and_large_single(data)
            reference_data = self._clean_nan_and_large_single(reference_data)

        return data, reference_data

    def _batchwise_error(self, data:Union[np.ndarray, h5py.Dataset]=None,
                               reference_data:Union[np.ndarray, h5py.Dataset]=None,
                               relative_norm:bool=False, data_interval:list=None, batch_size:int=1) -> float:

        """It evaluated the error over a single batch a time
        :param data: the data to be usd for assess the norm
        :type data: Union[np.ndarray, h5py.Dataset]
        :param reference_data: the data to use as comparison
        :type data: Union[np.ndarray, h5py.Dataset]
        :param relative_norm: use relative norm or not ?
        :type relative_norm: bool
        :param data_interval: the interval over the samples' axis to use for evaluating the norm
        :type data_interval: list
        :param batch_size: the maximum size of each batch
        :type batch_size: int
        :returns: the total norm
        :rtype: float
        """

        batches = batchdomain_constructor(data_interval, batch_size)

        accumulated_error = 0
        accumulated_ref = 0

        for batch_id, batch in enumerate(batches):

            chunk_array = data[slice(*batch)]

            print(f"Evaluating error for the batch {batch_id+1}/{len(batches)} batch_size={chunk_array.shape[0]}")


            chunk_ref_array = reference_data[slice(*(batch + data_interval[0]))]

            if chunk_array.dtype.names:
                chunk_array_numeric = np.concatenate([chunk_array[var]
                                                      for var in chunk_array.dtype.names], axis=1)

                chunk_ref_array_numeric = np.concatenate([chunk_ref_array[var]
                                                         for var in chunk_ref_array.dtype.names], axis=1)

            else:
                chunk_array_numeric = chunk_array

                chunk_ref_array_numeric = chunk_ref_array

            chunk_array_numeric, chunk_ref_array_numeric = self._clean_nan_and_large(chunk_array_numeric,
                                                                                     chunk_ref_array_numeric)

            accumulated_error += np.sum(np.square(chunk_array_numeric.flatten()
                                                  - chunk_ref_array_numeric.flatten()))

            accumulated_ref += np.sum(np.square(chunk_ref_array_numeric.flatten()))

        if relative_norm:
            return np.sqrt(accumulated_error/accumulated_ref)
        else:
            return np.sqrt(accumulated_error)

    def __call__(self, data:Union[np.ndarray, da.core.Array, h5py.Dataset]=None,
                       reference_data:Union[np.ndarray, da.core.Array, h5py.Dataset]=None,
                       relative_norm:bool=False, data_interval:list=None, batch_size:int=1, ord:int=2) -> float:

        """It evaluated the error over a single batch a time
        :param data: the data to be usd for assess the norm
        :type data: Union[np.ndarray, da.core.Array, h5py.Dataset]
        :param reference_data: the data to use as comparison
        :type data: Union[np.ndarray, da.core.Array, h5py.Dataset]
        :param relative_norm: use relative norm or not ?
        :type relative_norm: bool
        :param data_interval: the interval over the samples' axis to use for evaluating the norm
        :type data_interval: list
        :param batch_size: the maximum size of each batch
        :type batch_size: int
        :returns: the total norm
        :rtype: float
        """

        # NumPy and Dask arrays have similar properties
        if isinstance(data, (np.ndarray, da.core.Array)) and isinstance(reference_data, (np.ndarray, da.core.Array)):

            data, reference_data = self._clean_nan_and_large(data, reference_data)

            # Linear algebra engine
            if isinstance(data, np.ndarray):
                lin = np
            elif isinstance(data, da.core.Array):
                lin = da
            else:
                lin = np

            # The arrays must be flattened because np.linalg.norm accept only two-dimensional
            # arrays

            data_ = data.flatten()

            reference_data_ = reference_data.flatten()

            if relative_norm:
                eval_ = lin.linalg.norm(data_ - reference_data_, ord=ord)/lin.linalg.norm(reference_data_, ord=ord)
            else:
                eval_ = lin.linalg.norm(data_-reference_data_, ord=ord)

            return float(eval_)

        if isinstance(data, h5py.Dataset) and isinstance(reference_data, h5py.Dataset):

            assert ord == 2, 'We only implemented the norm 2 for hdf5 input data'

            assert data_interval, "In using a h5py.Dataset it is necessary" \
                                  "to provide a data interval"

            return self._batchwise_error(data=data, reference_data=reference_data,
                                         relative_norm=relative_norm,
                                         data_interval=data_interval, batch_size=batch_size)

        else:

            raise Exception("Data format not supported. It must be np.ndarray, dask.core.Array or h5py.Dataset.")

class SampleWiseErrorNorm:

    name = "samplewiseerrornorm"

    def __init__(self, ):
        pass

    def _aggregate_norm(self, norms, ord):

        n = np.stack(norms, axis=0)
        if ord == 1:
            return np.sum(n, axis=0)
        elif ord == 2:
            return np.sqrt(np.sum(n*n, axis=0))
        elif ord == np.inf:
            return np.max(n, axis=0)
        else:
            raise RuntimeError(f'Norm ord={str(ord)} not supported')

    def __call__(self, data=None, reference_data=None, relative_norm=False,
                 key=None, data_interval=None, batch_size=1, ord=2):
        """

        :param data: np.ndarray
        :param reference_data: np.ndarray
        :param relative_norm: bool
        :return: None
        """

        if data_interval is None:
            data_interval = (0, data.shape[0])

        n_samples = data_interval[1] - data_interval[0]
        if isinstance(data, np.ndarray) and isinstance(reference_data, np.ndarray):

            data_ = np.reshape(data[slice(*data_interval)], [n_samples, -1])
            reference_data_ = np.reshape(reference_data[slice(*data_interval)], [n_samples, -1])

            norm = np.linalg.norm(data_-reference_data_, ord=ord, axis=0)
            if relative_norm:
                ref_norm = np.linalg.norm(reference_data_, ord=ord, axis=0)
                norm = _relative(norm, ref_norm)

        elif isinstance(data, h5py.Dataset) and isinstance(reference_data, h5py.Dataset):

            if isinstance(batch_size, MemorySizeEval):
                batch_size = batch_size(max_batches=data_interval[1] - data_interval[0], shape=data.shape[1:])

            batches = batchdomain_constructor(data_interval, batch_size)

            is_string_key = isinstance(key, str)
            if is_string_key:
                keys = [key]
            else:
                keys = key

            norms_dict = []
            ref_norms_dict = []
            for ii, batch in enumerate(batches):
                d = data[slice(*batch)]
                print(f'Computing norm for batch {ii+1}/{len(batches)}. batch_size={d.shape[0]}')

                r = reference_data[slice(*batch)]

                nb = {}
                rb = {}
                for k in keys:
                    rk = r[k]
                    nb[k] = self.__call__(data=d[k],
                                          reference_data=rk,
                                          relative_norm=False,
                                          ord=ord)

                    if relative_norm:
                        rb[k] = np.linalg.norm(np.reshape(rk, [batch[1] - batch[0], -1]), ord=ord, axis=0)

                norms_dict.append(nb)
                ref_norms_dict.append(rb)

            norms_dict = {k: self._aggregate_norm([n[k] for n in norms_dict], ord) for k in keys}
            if relative_norm:
                ref_norms_dict = {k: self._aggregate_norm([n[k] for n in ref_norms_dict], ord) for k in keys}

                norms_dict = {k: _relative(norms_dict[k], ref_norms_dict[k]) for k in keys}

            if is_string_key:
                norm = norms_dict[key]
            else:
                norm = norms_dict
        else:

            raise Exception("Data format not supported. It must be np.ndarray or"
                            "h5py.Dataset.")

        return norm

def _relative(norm, ref_norm):
    ref_norm_zero = ref_norm == 0
    norm_zero = norm == 0

    ref_not_zero = np.logical_not(ref_norm_zero)
    norm[ref_not_zero] = norm[ref_not_zero] / ref_norm[ref_not_zero]

    norm[ref_norm_zero] = np.inf
    norm[norm_zero] = 0

    return norm

class FeatureWiseErrorNorm:

    name = "featurewiseerrornorm"

    def __init__(self, ):
        pass

    def __call__(self, data=None, reference_data=None, relative_norm=False,
                 key=None, data_interval=None, reference_data_interval=None, batch_size=1, ord=2):
        """

        :param data: np.ndarray
        :param reference_data: np.ndarray
        :param relative_norm: bool
        :return: None
        """

        if data_interval is None:
            data_interval = (0, data.shape[0])
        if reference_data_interval is None:
            reference_data_interval = (0, reference_data.shape[0])

        assert (data_interval[1]-data_interval[0]) == (reference_data_interval[1]-reference_data_interval[0])

        n_samples = data_interval[1] - data_interval[0]
        if isinstance(data, np.ndarray) and isinstance(reference_data, np.ndarray):

            data_ = np.reshape(data[slice(*data_interval)], [n_samples, -1])
            reference_data_ = np.reshape(reference_data[slice(*reference_data_interval)], [n_samples, -1])

            norm = np.linalg.norm(data_-reference_data_, ord=ord, axis=1)
            if relative_norm:
                ref_norm = np.linalg.norm(reference_data_, ord=ord, axis=1)
                norm = _relative(norm, ref_norm)

        elif isinstance(data, h5py.Dataset) and isinstance(reference_data, h5py.Dataset):

            if isinstance(batch_size, MemorySizeEval):
                batch_size = batch_size(max_batches=data_interval[1] - data_interval[0], shape=data.shape[1:])

            batches = batchdomain_constructor(data_interval, batch_size)
            batches_ref = batchdomain_constructor(reference_data_interval, batch_size)

            is_string_key = isinstance(key, str)
            if is_string_key:
                keys = [key]
            else:
                keys = key

            norms_dict = []
            ref_norms_dict = []
            for ii, (batch, batch_ref) in enumerate(zip(batches, batches_ref)):

                d = data[slice(*batch)]
                print(f'Computing norm for batch {ii+1}/{len(batches)} batch_size={d.shape[0]}')

                r = reference_data[slice(*batch_ref)]

                nb = {}
                rb = {}
                for k in keys:
                    rk = r[k]
                    nb[k] = self.__call__(data=d[k],
                                          reference_data=rk,
                                          relative_norm=False,
                                          ord=ord)

                    if relative_norm:
                        rb[k] = np.linalg.norm(np.reshape(rk, [batch[1] - batch[0], -1]), ord=ord, axis=1)

                norms_dict.append(nb)
                ref_norms_dict.append(rb)

            norms_dict = {k: np.concatenate([n[k] for n in norms_dict], axis=0) for k in keys}
            if relative_norm:
                ref_norms_dict = {k: np.concatenate([n[k] for n in ref_norms_dict], axis=0) for k in keys}

                norms_dict = {k: _relative(norms_dict[k], ref_norms_dict[k]) for k in keys}

            if is_string_key:
                norm = norms_dict[key]
            else:
                norm = norms_dict
        else:

            raise Exception("Data format not supported. It must be np.ndarray or"
                            "h5py.Dataset.")

        return norm

class DeterminationCoeff:

    def __init__(self) -> None:
        pass

    def __call__(self, data:np.ndarray=None, reference_data:np.ndarray=None) -> float:

        self.mean = reference_data.mean(axis=0)

        assert isinstance(data, np.ndarray), "Error! data is not a ndarray: {}".format(type(data))
        data_ = data.flatten()
        assert isinstance(reference_data, np.ndarray), \
            "Error! reference_data is not a ndarray: {}".format(type(reference_data))

        reference_data_ = reference_data.flatten()
        _reference_data = (reference_data - self.mean).flatten()

        return 1 - np.linalg.norm(data_ - reference_data_, 2)/np.linalg.norm(_reference_data, 2)

class RosensteinKantz:

    name = 'lyapunov_exponent'

    def __init__(self, epsilon:float=None) -> None:

        self.ref_index = 30
        self.epsilon = epsilon
        self.tau_amp = 30

    def _neighborhood(self, v_ref:np.ndarray, v:np.ndarray) -> np.ndarray:

        v_epsilon = np.where(np.abs(v-v_ref) <= self.epsilon)

        return v_epsilon

    def _reference_shift(self, v:np.ndarray, ref_index:int, shift:int) -> np.ndarray:

        return v[ref_index + shift]

    def __call__(self, data:np.ndarray=None) -> None:

        # It is expected data to be an array with shape (n_timesteps, n_variables)
        n_timesteps = data.shape[0]
        n_variables = data.shape[1]

        for vv in range(n_variables):

            var_ = data[:, vv]
            S_tau = list()

            for tau in range(-self.tau_amp, self.tau_amp):

                s_tau_list = list()
                for tt in range(self.ref_index, n_timesteps-self.ref_index-1):

                    var = var_[self.ref_index:n_timesteps-self.ref_index]

                    var_ref = self._reference_shift(var_, tt, tau)
                    var_copy = copy.copy(var)

                    var_copy = np.delete(var_copy, self.ref_index)

                    v_index_epsilon = self._neighborhood(var_ref, var_copy)[0]

                    assert not v_index_epsilon.size == 0

                    v_index = v_index_epsilon + tau
                    var_tau = var_[v_index]

                    s_tau = np.mean(np.abs(var_tau - var_ref))

                    s_tau_list.append(s_tau)

                    log_str = "Timestep {}".format(tt)
                    sys.stdout.write("\r" + log_str)
                    sys.stdout.flush()

                s_tau_array = np.array(s_tau_list)
                s_tau = s_tau_array.mean()
                S_tau.append(s_tau)

                print('\n')
                log_str = "Tau {}".format(tau)
                sys.stdout.write(log_str)


            S_tau = np.array(S_tau)
            print('.')

class PerturbationMethod:

    def __init__(self, jacobian_evaluator:callable=None) -> None:

        """
        :param jacobian_evaluator: function
        """
        self.jacobian_matrix_series = None
        self.global_timestep = None
        self.jacobian_evaluator = jacobian_evaluator

    def _definition_equation(self, z:np.ndarray) -> np.ndarray:

        """
        :param z: np.ndarray
        :return: np.ndarray
        """

        jacobian_matrix = self.jacobian_matrix_series[self.global_timestep, :, :]

        return jacobian_matrix.dot(z.T).T

    def __call__(self, data:np.ndarray=None, data_residual:np.ndarray=None, step:float=None) -> float:

        """
        :param data: np.ndarray
        :param data_residual: np.ndarray
        :param step: float
        :return: float
        """

        n_timesteps = data.shape[0]

        self.jacobian_matrix_series = self.jacobian_evaluator(data, data_residual=data_residual)

        init_state = np.array([1e-150, 1e-150, 1e-150])[None, :]

        lyapunov_exponents_list = list()

        integrator = RK4(self._definition_equation)

        current_state = init_state

        for timestep in range(n_timesteps):

            self.global_timestep = timestep
            variables_state, derivatives_state = integrator.step(current_state, step)

            current_state = variables_state

            sys.stdout.write("\rIteration {}/{}".format(timestep, n_timesteps))
            sys.stdout.flush()

            lyapunov_exponent_ = derivatives_state.dot(variables_state[0].T)
            lyapunov_exponent = lyapunov_exponent_/variables_state[0].dot(variables_state[0].T)

            lyapunov_exponents_list.append(lyapunov_exponent)

        lle = np.mean(lyapunov_exponents_list)

        return lle

class MeanEvaluation:

    def __init__(self) -> None:

        pass

    def __call__(self, dataset:Union[np.ndarray, h5py.Dataset]=None,
                       data_interval:list=None, batch_size:int=None,
                       data_preparer:DataPreparer=None) -> np.ndarray:

        if isinstance(batch_size, MemorySizeEval):
            batch_size = batch_size(max_batches=data_interval[1]-data_interval[0], shape=dataset.shape[1:])
        else:
            pass

        # Constructing the normalization  using the reference data
        batches = batchdomain_constructor(data_interval=data_interval, batch_size=batch_size)

        data_size = 0
        data_mean = 0

        for batch_id, batch in enumerate(batches):

            data = dataset[slice(*batch)]

            if data_preparer is None:
                data_ = data.view(np.float)
                data_flatten = data_.reshape(-1, np.product(data_.shape[1:]))
            else:
                data_flatten = data_preparer.prepare_input_structured_data(data)

            data_mean = (data_size * data_mean +
                         data_flatten.shape[0] * data_flatten.mean(0)) / (data_size + data_flatten.shape[0])

            data_size += data.shape[0]

        return data_mean

# #valuating Minimum and Maximum values for large dataset in batch-wise mode
class MinMaxEvaluation:

    def __init__(self) -> None:

        self.data_min_ref = np.inf
        self.data_max_ref = - np.inf
        self.default_axis = None

    def __call__(self, dataset:Union[np.ndarray, h5py.Dataset]=None,
                       data_interval:list=None, batch_size:int=None,
                       data_preparer:DataPreparer=None, axis:int=-1) -> np.ndarray:

        if isinstance(batch_size, MemorySizeEval):
            batch_size = batch_size(max_batches=data_interval[1]-data_interval[0], shape=dataset.shape[1:])
        else:
            pass

        # Constructing the normalization  using the reference data
        batches = batchdomain_constructor(data_interval=data_interval, batch_size=batch_size)

        if axis is not None:

            n_dims = len(dataset.shape)
            axes = [i for i in range(n_dims)]
            axes.remove(axis)
            axes = tuple(axes)

            data_max = self.data_max_ref*np.ones(n_dims - 1)
            data_min = self.data_min_ref*np.ones(n_dims - 1)

        else:

            axes = self.default_axis
            data_max = self.data_max_ref
            data_min = self.data_min_ref

        for batch_id, batch in enumerate(batches):

            data = dataset[slice(*batch)]

            max_ = data.max(axes)
            min_ = data.min(axes)

            data_max  = np.maximum(max_, data_max)
            data_min = np.minimum(min_, data_min)

        return data_max, data_min

    def eval_h5(self, dataset: h5py.Group = None,
                      data_interval: list = None, batch_size: int = None,
                      data_preparer: DataPreparer = None, axis: int = -1, keys:list=None) -> np.ndarray:

            data_min_ = list()
            data_max_ = list()

            for k in keys:

                data = dataset[k]

                data_max, data_min = self.__call__(dataset=data, data_interval=data_interval, batch_size=batch_size,
                                                   data_preparer=data_preparer, axis=axis)

                data_min_.append(data_min)
                data_max_.append(data_max)

            return np.hstack(data_min_), np.hstack(data_max_)

class MemorySizeEval:

    def __init__(self, memory_tol_percent:float=0.5) -> None:

        self.memory_tol_percent = memory_tol_percent
        self.size_default = np.array([0]).astype('float64').itemsize
        self.available_memory = None
        self.multiplier = 1024

    @property
    def available_memory_in_GB(self) -> float:
        if self.available_memory is not None:
            return self.available_memory/(self.multiplier**3)
        else:
            raise Warning("The available was not evaluated. Execute the __call__ method.")

    def __call__(self, max_batches:int=None, shape:Union[tuple, list]=None) -> int:

        self.available_memory = self.memory_tol_percent * psutil.virtual_memory().available

        memory_size = self.size_default*np.prod(shape)

        if memory_size <= self.available_memory:

            possible_batch_size = max_batches

        else:

            possible_batch_size = np.ceil(memory_size/self.available_memory).astype(int)


        return possible_batch_size

# Cumulative error norm for time-series
class CumulativeNorm:

    def __init__(self):

        pass

    def __call__(self, data:np.ndarray=None, reference_data:np.ndarray=None,
                       relative_norm:bool=False, ord:int=2) -> np.ndarray:

        assert len(data.shape) == len(reference_data.shape) == 2, "The data and reference_data must be two-dimensional."

        cumulative_norm = np.sqrt(np.cumsum(np.square(data - reference_data), axis=0)/np.cumsum(np.square(reference_data), axis=0))

        return np.where(cumulative_norm == np.NaN, 0, cumulative_norm)

# Cumulative error norm for time-series
class PointwiseError:

    def __init__(self):

        pass

    def __call__(self, data:np.ndarray=None, reference_data:np.ndarray=None,
                       relative_norm:bool=False, ord:int=2) -> np.ndarray:

        assert len(data.shape) == len(reference_data.shape) == 2, "The data and reference_data must be two-dimensional."

        pointwise_relative_error = (data - reference_data)/reference_data

        filter_nan = np.where(pointwise_relative_error == np.NaN, 0, pointwise_relative_error)
        filter_nan_inf = np.where(filter_nan == np.inf, 0, filter_nan)

        return filter_nan_inf

# Evaluating the number of Lyapunov units using cumulative error norm
class LyapunovUnits:

    def __init__(self, lyapunov_unit:float=1, tol=0.10, time_scale=1, norm_criteria='cumulative_norm'):

        self.lyapunov_unit = lyapunov_unit
        self.tol = tol
        self.time_scale = time_scale

        if norm_criteria == 'cumulative_norm':
            self.norm = CumulativeNorm()
        elif norm_criteria == 'pointwise_error':
            self.norm = PointwiseError()
        else:
            raise Exception(f"The case {norm_criteria} is not available.")

    def __call__(self, data:np.ndarray=None, reference_data:np.ndarray=None,
                       relative_norm:bool=False, ord:int=2) -> float:

        cumulative_norm = self.norm(data=data, reference_data=reference_data, relative_norm=relative_norm)

        respect = cumulative_norm[cumulative_norm <= self.tol]

        return respect.shape[0]*self.time_scale/self.lyapunov_unit

class MahalanobisDistance:

    def __init__(self, batchwise:bool=None) -> None:

        self.batchwise = batchwise

        if self.batchwise == True:
            self.inner_dot = self.batchwise_inner_dot
        else:
            self.inner_dot = self.simple_inner_dot

    def simple_inner_dot(self, a:torch.Tensor=None, b:torch.Tensor=None,
                               metric_tensor: Union[np.ndarray, torch.Tensor]=None) -> torch.Tensor:

        return a.T @ metric_tensor @ b

    def batchwise_inner_dot(self, a:torch.Tensor=None, b:torch.Tensor=None,
                                  metric_tensor: Union[np.ndarray, torch.Tensor]=None) -> torch.Tensor:

        b_til = b @ metric_tensor

        return torch.bmm(a[:, None, :], b_til[..., None]).squeeze()

    def __call__(self, metric_tensor: Union[np.ndarray, torch.Tensor], center:Union[np.ndarray, torch.Tensor],
                       point:Union[np.ndarray, torch.Tensor]) -> torch.Tensor:

        return self.inner_dot(a = center - point, metric_tensor = metric_tensor, b = center - point)










