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
from typing import List, Optional, Tuple, Union

import dask.array as da
import h5py
import numpy as np
import psutil
import torch

from simulai.abstract import DataPreparer
from simulai.batching import batchdomain_constructor
from simulai.math.integration import RK4


def _relative(norm: np.ndarray, ref_norm: np.ndarray) -> np.ndarray:
    """
    General usage relative norm.

    Parameters:
    -----------
    norm : np.ndarray
        The norm to be normalized.
    ref_norm : np.ndarray
        The norm to be used for normalization.

    Returns:
    --------
    norm : np.ndarray
        The normalized norm.
    """

    ref_norm_zero = ref_norm == 0
    norm_zero = norm == 0

    ref_not_zero = np.logical_not(ref_norm_zero)
    norm[ref_not_zero] = norm[ref_not_zero] / ref_norm[ref_not_zero]

    norm[ref_norm_zero] = np.inf
    norm[norm_zero] = 0

    return norm


class ByPass:
    name = "no_metric"

    def __init__(self):
        pass


class L2Norm:
    name = "l2_norm"

    def __init__(
        self,
        mask: Union[str, float, None] = None,
        do_clean_data: bool = True,
        large_number: float = 1e15,
        default_data: float = 0.0,
    ) -> None:
        """
        Parameters
        __________

        mask: Union[str, float]
            A floating point number or string indicating which positions in
            the dataset are not valid, it means, are missing data.
        do_clean_data: bool
            It is necessary to execute a cleaning in the dataset (removing NaN and very large numbers)
            or not ?
        large_number: float
            Threshold for considering number as large numbers.
        default_data: float
            The defulat data used for replacing NaN and large number when the
            option `do_clean_data` is `True`.
        """

        self.large_number = large_number
        self.mask = mask
        self.do_clean_data = do_clean_data
        self.default_data = default_data

    def _clean_nan_and_large_single(self, d: np.ndarray) -> np.ndarray:
        """
        It removes NaN and large numbers in an array and replaces those by
        a defualt value.

        Parameters
        __________

        d : np.ndarray
            The array to be cleaned.

        Returns
        _______

        np.ndarray
            The array cleaned.
        """

        if self.mask is not None:
            is_mask = d == self.mask
            if np.any(is_mask):
                d = np.where(is_mask, self.default_data, d)
                print(
                    f"There are values equal to the mask=={self.mask} in the dataset. They will be replaced with {self.default_data}."
                )

        isnan_data = np.isnan(d)
        if np.any(isnan_data):
            d = np.where(isnan_data, self.default_data, d)
            print(
                f"There are NaN's in the dataset. They will be replaced with {self.default_data}."
            )

        is_large = np.abs(d) >= self.large_number
        if np.any(is_large):
            d = np.where(is_large, self.default_data, d)
            print(
                f"There are large numbers, i.e., abs(x) > {self.large_number}, in the dataset. They will be replaced with {self.default_data}."
            )

        return d

    def _clean_nan_and_large(
        self, data: np.ndarray, reference_data: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        It removes NaN and large number of the input and the reference dataset.

        Parameters
        __________

        data: np.ndarray
            The data to be evaluated in the norm.
        reference_data: np.ndarray
            the data to be used as comparison.

        Returns
        _______

        Tuple[np.ndarray, np.ndarray]
            Both the datasets after have been cleaned.

        """

        if self.do_clean_data:
            data = self._clean_nan_and_large_single(data)
            reference_data = self._clean_nan_and_large_single(reference_data)

        return data, reference_data

    def _batchwise_error(
        self,
        data: Union[np.ndarray, h5py.Dataset] = None,
        reference_data: Union[np.ndarray, h5py.Dataset] = None,
        relative_norm: bool = False,
        data_interval: List[int] = None,
        batch_size: int = 1,
    ) -> float:
        """
        It evaluates the error over a single dataset batch a time.

        Parameters
        __________

        data : Union[np.ndarray, h5py.Dataset]
            The data to be used for assessing the norm.
        reference_data : Union[np.ndarray, h5py.Dataset]
            The data to be used as comparison.
        relative_norm : bool
            Using relative norm or not ? (Dividing the error norm by the norm of reference_data)
        data_interval : list
            The interval along the samples axis to use for evaluating the norm.
        batch_size : int
            The maximum size of each mini-batch created for evaluating the norm.

        Returns
        _______

        float
            The overall norm for the batch.

        """
        batches = batchdomain_constructor(data_interval, batch_size)

        accumulated_error = 0
        accumulated_ref = 0

        for batch_id, batch in enumerate(batches):
            chunk_array = data[slice(*batch)]

            print(
                f"Evaluating error for the batch {batch_id+1}/{len(batches)} batch_size={chunk_array.shape[0]}"
            )

            chunk_ref_array = reference_data[slice(*(batch + data_interval[0]))]

            if chunk_array.dtype.names:
                chunk_array_numeric = np.concatenate(
                    [chunk_array[var] for var in chunk_array.dtype.names], axis=1
                )

                chunk_ref_array_numeric = np.concatenate(
                    [chunk_ref_array[var] for var in chunk_ref_array.dtype.names],
                    axis=1,
                )

            else:
                chunk_array_numeric = chunk_array

                chunk_ref_array_numeric = chunk_ref_array

            chunk_array_numeric, chunk_ref_array_numeric = self._clean_nan_and_large(
                chunk_array_numeric, chunk_ref_array_numeric
            )

            accumulated_error += np.sum(
                np.square(
                    chunk_array_numeric.flatten() - chunk_ref_array_numeric.flatten()
                )
            )

            accumulated_ref += np.sum(np.square(chunk_ref_array_numeric.flatten()))

        if relative_norm:
            return np.sqrt(accumulated_error / accumulated_ref)
        else:
            return np.sqrt(accumulated_error)

    def __call__(
        self,
        data: Union[np.ndarray, da.core.Array, h5py.Dataset] = None,
        reference_data: Union[np.ndarray, da.core.Array, h5py.Dataset] = None,
        relative_norm: bool = False,
        data_interval: List[int] = None,
        batch_size: int = 1,
        ord: int = 2,
    ) -> float:
        """
         It evaluates the norm error os a large dataset in a batchwise and lazzy way.

         Parameters
         __________

         data : Union[np.ndarray, da.core.Array, h5py.Dataset]
             The data to be used for assessing the norm.
         reference_data : Union[np.ndarray, da.core.Array, h5py.Dataset]
             The data to be used for comparison.
         relative_norm : bool
             Using relative norm or not ? (Dividing the error norm by the norm of reference_data)
        data_interval : list
             The interval along the samples axis to use for evaluating the norm.
         batch_size : int
             The maximum size of each mini-batch created for evaluating the norm.

         Returns
         _______

         float
             The overall norm for the dataset.

        """

        # NumPy and Dask arrays have similar properties
        if isinstance(data, (np.ndarray, da.core.Array)) and isinstance(
            reference_data, (np.ndarray, da.core.Array)
        ):
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
                eval_ = lin.linalg.norm(
                    data_ - reference_data_, ord=ord
                ) / lin.linalg.norm(reference_data_, ord=ord)
            else:
                eval_ = lin.linalg.norm(data_ - reference_data_, ord=ord)

            return float(eval_)

        if isinstance(data, h5py.Dataset) and isinstance(reference_data, h5py.Dataset):
            assert ord == 2, "We only implemented the norm 2 for hdf5 input data"

            assert data_interval, (
                "In using a h5py.Dataset it is necessary" "to provide a data interval"
            )

            return self._batchwise_error(
                data=data,
                reference_data=reference_data,
                relative_norm=relative_norm,
                data_interval=data_interval,
                batch_size=batch_size,
            )

        else:
            raise Exception(
                "Data format not supported. It must be np.ndarray, dask.core.Array or h5py.Dataset."
            )


class SampleWiseErrorNorm:
    name = "samplewiseerrornorm"

    def __init__(
        self,
    ) -> None:
        pass

    def _aggregate_norm(self, norms: List[float] = None, ord: int = None):
        """
        It stacks the list of norms (evaluated for multiple keys).
        """

        n = np.stack(norms, axis=0)
        if ord == 1:
            return np.sum(n, axis=0)
        elif ord == 2:
            return np.sqrt(np.sum(n * n, axis=0))
        elif ord == np.inf:
            return np.max(n, axis=0)
        else:
            raise RuntimeError(f"Norm ord={str(ord)} not supported")

    def __call__(
        self,
        data: Union[np.ndarray, h5py.Dataset] = None,
        reference_data: Union[np.ndarray, h5py.Dataset] = None,
        relative_norm: bool = False,
        key: str = None,
        data_interval: List[int] = None,
        batch_size: int = 1,
        ord: int = 2,
    ) -> None:
        """
        call method for the SampleWiseErrorNorm class

        Parameters:
        -----------
        data : Union[np.ndarray, h5py.Dataset]
            The data to be used for assessing the norm.
        reference_data : Union[np.ndarray, h5py.Dataset]
            The data to be used for comparison.
        relative_norm : bool
            Using relative norm or not ? (Dividing the error norm by the norm of reference_data)
        key : str
            The key to be used for accessing the data.
        data_interval : list
            The interval along the samples axis to use for evaluating the norm.
        batch_size : int
            The maximum size of each mini-batch created for evaluating the norm.
        ord : int
            The order of the norm to be used

        Returns:
        --------
        norm : float
            The overall norm for the dataset.
        """

        if data_interval is None:
            data_interval = (0, data.shape[0])

        n_samples = data_interval[1] - data_interval[0]
        if isinstance(data, np.ndarray) and isinstance(reference_data, np.ndarray):
            data_ = np.reshape(data[slice(*data_interval)], [n_samples, -1])
            reference_data_ = np.reshape(
                reference_data[slice(*data_interval)], [n_samples, -1]
            )

            norm = np.linalg.norm(data_ - reference_data_, ord=ord, axis=0)
            if relative_norm:
                ref_norm = np.linalg.norm(reference_data_, ord=ord, axis=0)
                norm = _relative(norm, ref_norm)

        elif isinstance(data, h5py.Dataset) and isinstance(
            reference_data, h5py.Dataset
        ):
            if isinstance(batch_size, MemorySizeEval):
                batch_size = batch_size(
                    max_batches=data_interval[1] - data_interval[0],
                    shape=data.shape[1:],
                )

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
                print(
                    f"Computing norm for batch {ii+1}/{len(batches)}. batch_size={d.shape[0]}"
                )

                r = reference_data[slice(*batch)]

                nb = {}
                rb = {}
                for k in keys:
                    rk = r[k]
                    nb[k] = self.__call__(
                        data=d[k], reference_data=rk, relative_norm=False, ord=ord
                    )

                    if relative_norm:
                        rb[k] = np.linalg.norm(
                            np.reshape(rk, [batch[1] - batch[0], -1]), ord=ord, axis=0
                        )

                norms_dict.append(nb)
                ref_norms_dict.append(rb)

            norms_dict = {
                k: self._aggregate_norm([n[k] for n in norms_dict], ord) for k in keys
            }
            if relative_norm:
                ref_norms_dict = {
                    k: self._aggregate_norm([n[k] for n in ref_norms_dict], ord)
                    for k in keys
                }

                norms_dict = {
                    k: _relative(norms_dict[k], ref_norms_dict[k]) for k in keys
                }

            if is_string_key:
                norm = norms_dict[key]
            else:
                norm = norms_dict
        else:
            raise Exception(
                "Data format not supported. It must be np.ndarray or" "h5py.Dataset."
            )

        return norm


class FeatureWiseErrorNorm:
    """
    Feature-wise error norm for a dataset.

    Attributes:
    -----------
    name : str
       The name of the class.
    """

    name = "featurewiseerrornorm"

    def __init__(
        self,
    ):
        pass

    def __call__(
        self,
        data: Union[np.ndarray, h5py.Dataset] = None,
        reference_data: Union[np.ndarray, h5py.Dataset] = None,
        relative_norm: bool = False,
        key: str = None,
        data_interval: List[int] = None,
        reference_data_interval=None,
        batch_size=1,
        ord=2,
    ):
        """
        Compute the L2-norm or relative L2-norm between data and reference data along the first axis.

        Parameters
        ----------
        data : numpy.ndarray or h5py.Dataset, optional
            Data to compute the norm, by default None.
        reference_data : numpy.ndarray or h5py.Dataset, optional
            Reference data to compute the norm, by default None.
        relative_norm : bool, optional
            Whether to compute the relative norm, by default False.
        key : str or sequence of str, optional
            Keys of the datasets to compute the norm when data and reference_data are h5py.Datasets,
            by default None.
        data_interval : tuple, optional
            Interval of the data to use for computing the norm, by default None.
        reference_data_interval : tuple, optional
            Interval of the reference data to use for computing the norm, by default None.
        batch_size : int or MemorySizeEval, optional
            Size of the batch to use for computing the norm when data and reference_data are h5py.Datasets,
            by default 1.
        ord : int or inf or -inf or 'fro', optional
            Order of the norm. Supported values are positive integers,
            numpy.inf, numpy.NINF and 'fro'. By default 2.

        Returns
        -------
        norm : numpy.ndarray
            L2-norm or relative L2-norm between data and reference_data along the first axis.
            The shape of the output array depends on the input data format and the value of `key`.
        """
        if data_interval is None:
            data_interval = (0, data.shape[0])
        if reference_data_interval is None:
            reference_data_interval = (0, reference_data.shape[0])

        assert (data_interval[1] - data_interval[0]) == (
            reference_data_interval[1] - reference_data_interval[0]
        )

        n_samples = data_interval[1] - data_interval[0]
        if isinstance(data, np.ndarray) and isinstance(reference_data, np.ndarray):
            data_ = np.reshape(data[slice(*data_interval)], [n_samples, -1])
            reference_data_ = np.reshape(
                reference_data[slice(*reference_data_interval)], [n_samples, -1]
            )

            norm = np.linalg.norm(data_ - reference_data_, ord=ord, axis=1)
            if relative_norm:
                ref_norm = np.linalg.norm(reference_data_, ord=ord, axis=1)
                norm = _relative(norm, ref_norm)

        elif isinstance(data, h5py.Dataset) and isinstance(
            reference_data, h5py.Dataset
        ):
            if isinstance(batch_size, MemorySizeEval):
                batch_size = batch_size(
                    max_batches=data_interval[1] - data_interval[0],
                    shape=data.shape[1:],
                )

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
                print(
                    f"Computing norm for batch {ii+1}/{len(batches)} batch_size={d.shape[0]}"
                )

                r = reference_data[slice(*batch_ref)]

                nb = {}
                rb = {}
                for k in keys:
                    rk = r[k]
                    nb[k] = self.__call__(
                        data=d[k], reference_data=rk, relative_norm=False, ord=ord
                    )

                    if relative_norm:
                        rb[k] = np.linalg.norm(
                            np.reshape(rk, [batch[1] - batch[0], -1]), ord=ord, axis=1
                        )

                norms_dict.append(nb)
                ref_norms_dict.append(rb)

            norms_dict = {
                k: np.concatenate([n[k] for n in norms_dict], axis=0) for k in keys
            }
            if relative_norm:
                ref_norms_dict = {
                    k: np.concatenate([n[k] for n in ref_norms_dict], axis=0)
                    for k in keys
                }

                norms_dict = {
                    k: _relative(norms_dict[k], ref_norms_dict[k]) for k in keys
                }

            if is_string_key:
                norm = norms_dict[key]
            else:
                norm = norms_dict
        else:
            raise Exception(
                "Data format not supported. It must be np.ndarray or" "h5py.Dataset."
            )

        return norm


class DeterminationCoeff:
    """
    Determination coefficient (R^2) between data and reference data.
    """

    def __init__(self) -> None:
        pass

    def __call__(
        self, data: np.ndarray = None, reference_data: np.ndarray = None
    ) -> float:
        """
        Call methoh to compute the determination coefficient.

        Parameters:
        -----------
        data : numpy.ndarray, optional
            Data to compute the determination coefficient, by default None.
        reference_data : numpy.ndarray, optional
            Reference data to compute the determination coefficient, by default None.

        Returns:
        --------
        determination_coeff : float
            Determination coefficient between data and reference_data.
        """
        self.mean = reference_data.mean(axis=0)

        assert isinstance(data, np.ndarray), "Error! data is not a ndarray: {}".format(
            type(data)
        )
        data_ = data.flatten()
        assert isinstance(
            reference_data, np.ndarray
        ), "Error! reference_data is not a ndarray: {}".format(type(reference_data))

        reference_data_ = reference_data.flatten()
        _reference_data = (reference_data - self.mean).flatten()

        return 1 - np.linalg.norm(data_ - reference_data_, 2) / np.linalg.norm(
            _reference_data, 2
        )


class RosensteinKantz:
    """
    Rosenstein-Kantz algorithm to compute the Lyapunov exponent.

    Attributes:
    -----------
    name : str
        Name of the algorithm.
    """

    name = "lyapunov_exponent"

    def __init__(self, epsilon: float = None) -> None:
        """
        Initialize the Rosenstein-Kantz algorithm.

        Parameters:
        -----------
        epsilon : float, optional
            Epsilon value to compute the neighborhood, by default None.
        ref_index : int, optional
            Reference index to compute the neighborhood, by default 30.
        tau_amp : int, optional
            Amplitude of the time shift, by default 30.
        """
        self.ref_index = 30
        self.epsilon = epsilon
        self.tau_amp = 30

    def _neighborhood(self, v_ref: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        neighborhood function to compute the neighborhood of a given point.

        Parameters:
        -----------
        v_ref : numpy.ndarray
            Reference point.
        v : numpy.ndarray
            Point to compute the neighborhood.

        Returns:
        --------
        v_epsilon : numpy.ndarray
            Neighborhood of the point v.
        """
        v_epsilon = np.where(np.abs(v - v_ref) <= self.epsilon)

        return v_epsilon

    def _reference_shift(self, v: np.ndarray, ref_index: int, shift: int) -> np.ndarray:
        """
        reference_shift function to compute the reference shift.

        Parameters:
        -----------
        v : numpy.ndarray
            Vector to compute the reference shift.
        ref_index : int
            Reference index.
        shift : int
            Shift to compute the reference shift.

        Returns:
        --------
        return : numpy.ndarray
            Reference shift.
        """
        return v[ref_index + shift]

    def __call__(self, data: np.ndarray = None) -> float:
        """
        Call method to compute the Lyapunov exponent. It is expected data to be an array with shape (n_timesteps, n_variables).

        Parameters:
        -----------
        data : numpy.ndarray, optional
            Data to compute the Lyapunov exponent, by default None.

        Returns:
        --------
        return : float
            Lyapunov exponent. If the algorithm fails, it returns -1.
        """
        # It is expected data to be an array with shape (n_timesteps, n_variables)
        n_timesteps = data.shape[0]
        n_variables = data.shape[1]

        for vv in range(n_variables):
            var_ = data[:, vv]
            S_tau = list()

            for tau in range(-self.tau_amp, self.tau_amp):
                s_tau_list = list()
                for tt in range(self.ref_index, n_timesteps - self.ref_index - 1):
                    var = var_[self.ref_index : n_timesteps - self.ref_index]

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

                print("\n")
                log_str = "Tau {}".format(tau)
                sys.stdout.write(log_str)

            S_tau = np.array(S_tau)

            return S_tau[-1]


class PerturbationMethod:
    """
    Class for computing the largest Lyapunov exponent of a time series.

    Parameters
    ----------
    jacobian_evaluator : callable, optional
        A function that computes the Jacobian matrix at each time step.
        If None, it must be passed as an argument to the `__call__` method.

    Attributes
    ----------
    jacobian_matrix_series : ndarray or None
        The Jacobian matrices computed by the `jacobian_evaluator` at each time step.

    global_timestep : int or None
        The current time step.

    Methods
    -------
    __call__(self, data: ndarray = None, data_residual: ndarray = None, step: float = None) -> float:
        Computes the largest Lyapunov exponent of the time series data using the
        specified step size and residual data. Returns the Lyapunov exponent as a float.

    _definition_equation(self, z: ndarray) -> ndarray:
        Computes the derivative of the state vector z. Returns an ndarray.
    """

    def __init__(self, jacobian_evaluator: callable = None) -> None:
        """
        Initialize the LyapunovExponent object.

        Parameters
        ----------
        jacobian_evaluator : callable, optional
            A function that computes the Jacobian matrix at each time step.
            If None, it must be passed as an argument to the `__call__` method.
        """
        self.jacobian_matrix_series = None
        self.global_timestep = None
        self.jacobian_evaluator = jacobian_evaluator

    def _definition_equation(self, z: np.ndarray) -> np.ndarray:
        """
        Compute the derivative of the state vector z.

        Parameters
        ----------
        z : ndarray
            The state vector.

        Returns
        -------
        ndarray
            The derivative of the state vector.
        """
        jacobian_matrix = self.jacobian_matrix_series[self.global_timestep, :, :]
        return jacobian_matrix.dot(z.T).T

    def __call__(
        self,
        data: np.ndarray = None,
        data_residual: np.ndarray = None,
        step: float = None,
    ) -> float:
        """
        Compute the largest Lyapunov exponent for a given dataset.

        Parameters
        ----------
        data : np.ndarray
            The dataset for which to compute the largest Lyapunov exponent.
        data_residual : np.ndarray, optional
            The residual data. Default is None.
        step : float, optional
            The time step to use. Default is None.

        Returns
        -------
        float
        The largest Lyapunov exponent for the given dataset.
        """
        n_timesteps = data.shape[0]

        self.jacobian_matrix_series = self.jacobian_evaluator(
            data, data_residual=data_residual
        )

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
            lyapunov_exponent = lyapunov_exponent_ / variables_state[0].dot(
                variables_state[0].T
            )

            lyapunov_exponents_list.append(lyapunov_exponent)

        lle = np.mean(lyapunov_exponents_list)

        return lle


class MeanEvaluation:
    def __init__(self) -> None:
        """
        Evaluating mean for large dataset in batch-wise mode
        """
        pass

    def __call__(
        self,
        dataset: Union[np.ndarray, h5py.Dataset] = None,
        data_interval: List[int] = None,
        batch_size: int = None,
        data_preparer: DataPreparer = None,
    ) -> np.ndarray:
        """
        It evaluates the mean values of a h5py.Dataset (lazzy) or numpy.ndarray (on memory)
        object

        Parameters
        ----------
        dataset : Union[h5py.Dataset, np.ndarray]
                  The dataset to be used for MinMax evaluation.
        data_interval : list
                  A list containing the interval along axis 0 (batches) used for
                  the MinMax evaluation.
        batch_size : int
                  The value of the batch size to be used for the incremental evaluation.
                  Usually it is chosen as smaller than the total dataset size in order to avoid
                  memory overflow.
        data_preparer : DataPreparer (simulai.io.DataPreparer), optional
                  A class for reformatting the data before executing the MinMax evaluation.

        Returns
        -------
        np.ndarray
                  An array for the mean values.
        """

        if isinstance(batch_size, MemorySizeEval):
            batch_size = batch_size(
                max_batches=data_interval[1] - data_interval[0], shape=dataset.shape[1:]
            )
        else:
            pass

        # Constructing the normalization  using the reference data
        batches = batchdomain_constructor(
            data_interval=data_interval, batch_size=batch_size
        )

        data_size = 0
        data_mean = 0

        for batch_id, batch in enumerate(batches):
            data = dataset[slice(*batch)]

            if data_preparer is None:
                data_ = data.view(float)
                data_flatten = data_.reshape(-1, np.product(data_.shape[1:]))
            else:
                data_flatten = data_preparer.prepare_input_structured_data(data)

            data_mean = (
                data_size * data_mean + data_flatten.shape[0] * data_flatten.mean(0)
            ) / (data_size + data_flatten.shape[0])

            data_size += data.shape[0]

        return data_mean


class MinMaxEvaluation:
    def __init__(self) -> None:
        """
        Evaluating Minimum and Maximum values for large dataset in batch-wise mode
        """

        self.data_min_ref = np.inf
        self.data_max_ref = -np.inf
        self.default_axis = None

    def __call__(
        self,
        dataset: Union[np.ndarray, h5py.Dataset] = None,
        data_interval: List[int] = None,
        batch_size: int = None,
        data_preparer: Optional[DataPreparer] = None,
        axis: int = -1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        It evaluates the minimum and maximum values of a h5py.Dataset (lazzy) or numpy.ndarray (on memory)
        object

        Parameters
        ----------
        dataset : Union[h5py.Dataset, np.ndarray]
                  The dataset to be used for MinMax evaluation.
        data_interval : list
                  A list containing the interval along axis 0 (batches) used for
                  the MinMax evaluation.
        batch_size : int
                  The value of the batch size to be used for the incremental evaluation.
                  Usually it is chosen as smaller than the total dataset size in order to avoid
                  memory overflow.
        data_preparer : DataPreparer (simulai.io.DataPreparer), optional
                  A class for reformatting the data before executing the MinMax evaluation.
        axis : int
                  The axis used as reference for the MinMax evaluation. Use None for global
                  values.

        Returns
        -------
        np.ndarray
                  An array for the maximum values.
        np.ndarray
                  An array for the minimum values.

        """

        if isinstance(batch_size, MemorySizeEval):
            batch_size = batch_size(
                max_batches=data_interval[1] - data_interval[0], shape=dataset.shape[1:]
            )
        else:
            pass

        # Constructing the normalization  using the reference data
        batches = batchdomain_constructor(
            data_interval=data_interval, batch_size=batch_size
        )

        if axis is not None:
            n_dims = len(dataset.shape)
            axes = [i for i in range(n_dims)]

            axes.remove(axis)

            axes = tuple(axes)

            data_max = self.data_max_ref * np.ones(n_dims - 1)
            data_min = self.data_min_ref * np.ones(n_dims - 1)

        else:
            axes = self.default_axis
            data_max = self.data_max_ref
            data_min = self.data_min_ref

        for batch_id, batch in enumerate(batches):
            data = dataset[slice(*batch)]

            max_ = data.max(axes)
            min_ = data.min(axes)

            data_max = np.maximum(max_, data_max)
            data_min = np.minimum(min_, data_min)

        return data_max, data_min

    def eval_h5(
        self,
        dataset: Union[h5py.Group, h5py.File] = None,
        data_interval: List[int] = None,
        batch_size: int = None,
        data_preparer: DataPreparer = None,
        axis: int = -1,
        keys: list = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        It evaluates the minimum and maximum values of a h5py.Dataset (lazzy) or numpy.ndarray (on memory)
        object

        Parameters
        ----------
        dataset : Union[h5py.Group, h5py.File]
                  The dataset to be used for MinMax evaluation. The type h5py.File is supported
                  only when the datasets are directly placed inside it, without intermediary
                  h5py.Group objects.
        data_interval : list
                  A list containing the interval along axis 0 (batches) used for
                  the MinMax evaluation.
        batch_size : int
                  The value of the batch size to be used for the incremental evaluation.
                  Usually it is chosen as smaller than the total dataset size in order to avoid
                  memory overflow.
        data_preparer : DataPreparer (simulai.io.DataPreparer), optional
                  A class for reformatting the data before executing the MinMax evaluation.
        axis : int
                  The axis used as reference for the MinMax evaluation. Use None for global
                  values.
        keys : list
                  The list of keys (variables or fields) to be used during the evaluation.
        Returns
        -------
        np.ndarray
                  An array for the maximum values.
        np.dnarray
                  An array for the minimum values.

        """

        data_min_ = list()
        data_max_ = list()

        for k in keys:
            data = dataset[k]

            data_max, data_min = self.__call__(
                dataset=data,
                data_interval=data_interval,
                batch_size=batch_size,
                data_preparer=data_preparer,
                axis=axis,
            )

            data_min_.append(data_min)
            data_max_.append(data_max)

        return np.hstack(data_max_), np.hstack(data_min_)


class MemorySizeEval:
    def __init__(self, memory_tol_percent: float = 0.5) -> None:
        """
        It determine a size for the batches in order to respect some
        used memory limit defined by the user

        Parameters
        ----------
        memory_tol_percent: float
            Maximum fraction of the available RAM memory to be used.
        """

        self.memory_tol_percent = memory_tol_percent
        self.size_default = np.array([0]).astype("float64").itemsize
        self.available_memory = None
        self.multiplier = 1024

    @property
    def available_memory_in_GB(self) -> float:
        """
        It evaluated the maximum available memory in Gigabytes.

        Returns
        -------
        float
            The available memory in GB.
        """

        if self.available_memory is not None:
            return self.available_memory / (self.multiplier**3)
        else:
            raise Warning(
                "The available was not evaluated. Execute the __call__ method."
            )

    def __call__(
        self, max_batches: int = None, shape: Union[tuple, list] = None
    ) -> int:
        """
        It determines the maximum batch size based on the dataset size and
        memory availability.

        Parameters
        ----------
        max_batches: int
            An estimative of the number of batches to be used.
        shape: Union[tuple, list]
            The shape of the dataset which will be processed in a
            batchwise way.

        Returns
        -------
        int
            The maximum value for the batch considering the memory availability.
        """

        self.available_memory = (
            self.memory_tol_percent * psutil.virtual_memory().available
        )

        memory_size = self.size_default * np.prod(shape)

        if memory_size <= self.available_memory:
            possible_batch_size = max_batches

        else:
            possible_batch_size = np.ceil(memory_size / self.available_memory).astype(
                int
            )

        return possible_batch_size


class CumulativeNorm:
    def __init__(self):
        """
        It evaluates cumulative error norms for time-series.
        """
        pass

    def __call__(
        self, data: np.ndarray = None, reference_data: np.ndarray = None
    ) -> np.ndarray:
        """
        Parameters
        ----------
        data: np.ndarray
            The data used for evaluating the norm.
        reference_data: np.ndarray
            The reference data used as comparison.

        Returns
        -------
        np.ndarray
            The cumulative norm along axis 0 of data.
        """

        assert (
            len(data.shape) == len(reference_data.shape) == 2
        ), "The data and reference_data must be two-dimensional."

        cumulative_norm = np.sqrt(
            np.cumsum(np.square(data - reference_data), axis=0)
            / np.cumsum(np.square(reference_data), axis=0)
        )

        return np.where(cumulative_norm == np.NaN, 0, cumulative_norm)


# Cumulative error norm for time-series
class PointwiseError:
    def __init__(self):
        """
        It evaluates the difference between each entry of a
        data array and its corresponding entry in the reference data.
        """

        pass

    def __call__(
        self, data: np.ndarray = None, reference_data: np.ndarray = None
    ) -> np.ndarray:
        """
        Parameters
        ----------
        data: np.ndarray
            The data used for evaluating the norm.
        reference_data: np.ndarray
            The reference data used as comparison.

        Returns
        -------
        np.ndarray
            The cumulative norm along axis 0 of data.
        """

        assert (
            len(data.shape) == len(reference_data.shape) == 2
        ), "The data and reference_data must be two-dimensional."

        pointwise_relative_error = (data - reference_data) / reference_data

        filter_nan = np.where(
            pointwise_relative_error == np.NaN, 0, pointwise_relative_error
        )
        filter_nan_inf = np.where(filter_nan == np.inf, 0, filter_nan)

        return filter_nan_inf


class LyapunovUnits:
    """
    Class for computing the Lyapunov units of a time-series. The Lyapunov units are computed as the cumulative norm of the error between the data and the reference data.
    """

    def __init__(
        self,
        lyapunov_unit: float = 1,
        tol=0.10,
        time_scale=1,
        norm_criteria="cumulative_norm",
    ):
        """
        Method for initializing the LyapunovUnits class.

        Parameters:
        -----------
        lyapunov_unit : float
            The Lyapunov unit of the time-series.
        tol : float
            The tolerance for the Lyapunov units.
        time_scale : float
            The time scale of the time-series.
        norm_criteria : str
            The criteria for computing the norm. The options are "cumulative_norm" and "pointwise_error".
        """
        self.lyapunov_unit = lyapunov_unit
        self.tol = tol
        self.time_scale = time_scale

        if norm_criteria == "cumulative_norm":
            self.norm = CumulativeNorm()
        elif norm_criteria == "pointwise_error":
            self.norm = PointwiseError()
        else:
            raise Exception(f"The case {norm_criteria} is not available.")

    def __call__(
        self,
        data: np.ndarray = None,
        reference_data: np.ndarray = None,
        relative_norm: bool = False,
    ) -> float:
        """
        Call method for the LyapunovUnits class.

        Parameters:
        -----------
        data : np.ndarray
            The data to be evaluated.
        reference_data : np.ndarray
            The reference data.
        relative_norm : bool
            A boolean variable to define if the relative norm will be computed or not.

        Returns:
        --------
        respect : float
            The Lyapunov units of the data.
        """
        cumulative_norm = self.norm(
            data=data, reference_data=reference_data, relative_norm=relative_norm
        )

        respect = cumulative_norm[cumulative_norm <= self.tol]

        return respect.shape[0] * self.time_scale / self.lyapunov_unit


class MahalanobisDistance:
    def __init__(self, batchwise: bool = None) -> None:
        """
        It evaluates the Mahalanobis distance metric

        Parameters
        ----------
        batchwise: bool
                A boolean variable to define if a batchwise loop will
                be executed to evaluate the metric or not.

        """

        self.batchwise = batchwise

        if self.batchwise == True:
            self.inner_dot = self.batchwise_inner_dot
        else:
            self.inner_dot = self.simple_inner_dot

    def simple_inner_dot(
        self,
        a: torch.Tensor = None,
        b: torch.Tensor = None,
        metric_tensor: Union[np.ndarray, torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Simple inner dot between tensors

        Parameters
        ----------
        a : torch.Tensor
            A one-dimensional tensor.
        b: torch.Tensor
            Another one-dimensional tensor.
        metric_tensor: Union[np.ndarray, torch.Tensor]
            A two-dimensional matrix (array or tensor) with
            the metric paramters.
        Returns
        -------
        torch.Tensor
            the result of the inner product
        """

        return a.T @ metric_tensor @ b

    def batchwise_inner_dot(
        self,
        a: torch.Tensor = None,
        b: torch.Tensor = None,
        metric_tensor: Union[np.ndarray, torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Inner dot between higher-dimensional tensors

        Parameters
        ----------
        a : torch.Tensor
            A one-dimensional tensor.
        b: torch.Tensor
            Another one-dimensional tensor.
        metric_tensor: Union[np.ndarray, torch.Tensor]
            A two-dimensional matrix (array or tensor) with
            the metric paramters.
        Returns
        -------
        torch.Tensor
            The result of the inner product
        """

        b_til = b @ metric_tensor

        return torch.bmm(a[:, None, :], b_til[..., None]).squeeze()

    def __call__(
        self,
        metric_tensor: Union[np.ndarray, torch.Tensor],
        center: Union[np.ndarray, torch.Tensor],
        point: Union[np.ndarray, torch.Tensor],
    ) -> torch.Tensor:
        """
        Evaluating the Mahalanobis distance.

        Parameters
        ----------
        metric_tensor: Union[np.ndarray, torch.Tensor]
            A two-dimensional matrix (array or tensor) with
            the metric paramters.
        center:  Union[np.ndarray, torch.Tensor]
            Centroids used as reference.
        points: Union[np.ndarray, torch.Tensor]
            Points for what the metric will be evaluated.
        Returns
        -------
        torch.Tensor
            The value of the metric for each point.
        """

        return self.inner_dot(
            a=center - point, metric_tensor=metric_tensor, b=center - point
        )
