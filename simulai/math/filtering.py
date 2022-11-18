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

import math

import numpy as np
import h5py
from scipy.integrate import quadrature
from typing import Union

from simulai.metrics import MemorySizeEval
from simulai.batching import batchdomain_constructor
from simulai.abstract import ROM

class HardPositivityLimiting:

    def __init__(self, tol:float=1e-10) -> None:

        """"Positivity limiting for avoiding negative values when they are
            physically inconsistent.
            It simply applies value >= tol for avoiding negative values.

        :param tol: The minimum acceptable value to ensure positivity
        :type tol: float
        """

        self.tol = tol

    def _apply_limiting(self, data:np.ndarray=None) -> (np.ndarray, int):

        """Effectively applying the limiting

        :param data: the data to be limited
        :type data: np.ndarray
        :return: the data limited
        :rtype: np.ndarray
        """

        indices = np.where(data < self.tol)
        data[indices] = self.tol

        return data, len(indices)

    def __call__(self, data:np.ndarray=None,
                       batch_size:Union[int, MemorySizeEval]=None) -> (np.ndarray, int):

        """The main call method

        :param data: the data to be processed by the limiting
        :type data: np.ndarray
        :param batch_size: Size of each batch to be processed
        :type batch_size:int
        :return: A tuple containing the processed batch_size and the number of point in which
                 the limiting was actioned.
        :rtype: (np.ndarray, int)
        """

        if isinstance(data, np.ndarray):
            return self._apply_limiting(data=data)

        elif isinstance(data, h5py.Dataset):

            assert batch_size, "It is necessary to provide a way for estimating the " \
                               "batch size."

            if isinstance(batch_size, MemorySizeEval):
                n_samples = data.shape[0]
                batch_size = batch_size(max_batches=n_samples, shape=data.shape[1:])
            else:
                pass

            # Constructing the normalization  using the reference data
            batches = batchdomain_constructor([0, n_samples], batch_size)

            variables_names = data.dtype.names
            n_variables = len(variables_names)
            total_number_of_limited = 0

            for batch_id , batch in enumerate(batches):

                chunk_data = data[slice(*batch)].view(float)
                limited_chunk_data, number_of_applied = self._apply_limiting(data=chunk_data)

                chunk_data = np.core.records.fromarrays(np.split(limited_chunk_data, n_variables, axis=1),
                                                                   names=variables_names,
                                                                   formats=','.join(len(variables_names) * ['f8']))

                data[slice(*batch)] = chunk_data
                total_number_of_limited += number_of_applied

            return data, total_number_of_limited

class TimeAveraging:

    def __init__(self, batch_size:int=None, axis:int=None) -> None:

        """Time-averaging along an axis

        :param batch_size: the size of the batch to be processed at each time
        :type batch_size: int
        :param axis: the axis along to apply time-averaging
        :type axis: int
        :return: nothing
        """

        self.batch_size = batch_size

        if not axis:
            self.axis = 1
        else:
            self.axis = axis

    def exec(self, dataset:Union[np.ndarray, h5py.Dataset], path:str=None) -> h5py.Dataset:

        """Executing time-averaging

        :param dataset: The dataset to be time-averaged
        :type dataset: Union[np.ndarray, h5py.Dataset]
        :param path: path to save the output into a HDF5 file
        :type path: str
        :return: HDF5 dataset with the time-averaged original data
        :rtype: h5py.Dataset
        """

        fp = h5py.File(path, 'w')

        output_n_samples = int(dataset.shape[0]/self.batch_size)
        other_dimensions = dataset.shape[1:]
        final_shape = (output_n_samples, ) + other_dimensions
        dtype = [(name, 'f8') for name in dataset.dtype.names]
        fp.create_dataset('data', shape=final_shape, dtype=dtype)
        output_dataset = fp.get('data')

        # The dataset is considered a hdf5 file
        n_variables = len(output_dataset.dtype.names)
        formats = n_variables*['f8']

        for ii in range(0, output_n_samples):

            mini_batch = dataset[ii*self.batch_size:(ii+1)*self.batch_size]

            mean_value = [mini_batch[name].mean(0) for name in dataset.dtype.names]

            mean_value_structured = np.core.records.fromarrays(mean_value, names=dataset.dtype.names,
                                                               formats=formats)

            output_dataset[ii] = mean_value_structured

        return output_dataset

class SVDThreshold:

    def __init__(self) -> None:

        """Methodology for defining a threshold for de-noising data via SVD decomposition.
        :return: nothing
        """
        self.limit_aspect_ratio = 0.1

    def lambda_special(self, beta:float=None) -> float:

        """Evaluating special lambda parameter

        :param beta: parameter beta
        :type beta: float
        :return: special lambda
        :rtype: float
        """

        return np.sqrt(2 * (beta + 1) + 8 * beta / ((beta + 1) + math.sqrt(beta ** 2 + 14 * beta + 1)))

    def lambda_function(self, beta:float=None,  shape:Union[tuple, list]=None) -> float:
        if shape[0] == shape[1]:
            return beta
        elif shape[1]/shape[0] < self.limit_aspect_ratio:
            return self.lambda_special(beta)
        else:
            raise Exception(f"This case is not covered, The data matrix in nor square,"
                            f" nor m/n < {self.limit_aspect_ratio} = m << n.")

    def beta_parameter(self, shape:Union[list, tuple]=None) -> float:
        n, m = shape
        # In case of square data matrices
        if n == m:
            return 4/np.sqrt(3)
        elif n < m:
            return n/m
        else:
            return m/n

    def integrand(self, t:float, beta:float=None) -> np.ndarray:

        monomial_upper = float((1 + np.sqrt(beta))**2)
        monomial_lower = float((1 - np.sqrt(beta))**2)

        out = np.where((monomial_upper - t)*(t - monomial_lower)>0,
                        np.sqrt((monomial_upper - t)*(t - monomial_lower))/(2*np.pi*beta*t),
                        0)

        return out

    def Marcenko_Pastur_integral(self, t:float, beta:float=None) -> float:

        """ Marcenko-Pastur Integral

        :param t: integration variable
        :type t: float
        :param beta: parameter beta
        :type beta: float
        :return: evaluation of the Marcenko-Pastur integral
        :rtype: float
        """

        upper_lim = (1 + np.sqrt(beta)) ** 2
        val, err = quadrature(self.integrand, t, upper_lim, args=(beta,))
        print(f"Quadrature error: {err}")

        return val

    def MedianMarcenkoPastur(self, beta:float) -> float:

        """ Marcenko-Pastur Median

        :param beta: parameter beta
        :type beta: float
        :return: evaluation of the Marcenko-Pastur median
        :rtype: float
        """

        MarPas = lambda t: 1 - self.Marcenko_Pastur_integral(beta, t)
        lobnd = (1 - np.sqrt(beta))**2
        hibnd = (1 + np.sqrt(beta))**2

        change = 1
        count = 0
        while change & (hibnd - lobnd > .001):
            print(f"Median Marcenko-Pastur, iteration {count}")
            change = 0
            x = np.linspace(lobnd, hibnd, 5)
            y = np.zeros(x.shape)
            for i in range(x.shape[0]):
                y[i] = MarPas(x[i])

                if any(y < 0.5):
                    lobnd = max(x[y < 0.5])
                    change = 1
                else:
                    pass

                if any(y > 0.5):
                    hibnd = min(x[y > 0.5])
                    change = 1
                else:
                    pass
            count += 1

        med = (hibnd + lobnd)/ 2

        return med

    def exec(self, singular_values:np.ndarray=None,
                   data_shape:Union[tuple, list]=None, gamma:float=None) -> np.ndarray:

        print("Executing SVD filtering.")

        n, m = data_shape

        beta = self.beta_parameter(shape=data_shape)

        if gamma is not None:
            lambda_parameter = self.lambda_function(beta=beta, shape=data_shape)
            tau_parameter = lambda_parameter*np.sqrt(n)*gamma
        else:
            sigma_med = np.median(singular_values)
            lambda_parameter = self.lambda_special(beta=beta)
            mu_parameter = self.MedianMarcenkoPastur(beta)
            tau_parameter = (lambda_parameter/mu_parameter)*sigma_med

        sv = singular_values[singular_values>=tau_parameter]

        print(f"SVD threshold: {tau_parameter}."
              f" Truncating component {len(sv)} of {len(singular_values)}")

        return sv

    def apply(self, pca:ROM=None, data_shape:Union[tuple, list]=None) -> ROM:

        assert hasattr(pca, "singular_values"), f"The object {type(pca)} has bo attribute singular_values."

        singular_values_truncated = self.exec(singular_values=pca.singular_values, data_shape=data_shape)
        pca.singular_values = singular_values_truncated
        pca.modes = pca.modes[:len(singular_values_truncated), :]

        return pca

class TimeSeriesExtremes:

    """
        Getting up the indices corresponding to the extremes of time-series
    """

    def __init__(self):
        pass

    def _curvature_change_indicator(self, data:np.ndarray=None, index:int=None) -> np.ndarray:

        assert type(index) == int, f"index must be an integer but received {type(index)}."

        previous = data[:-2, index]
        next = data[2:, index]

        return previous*next

    def _get_indices_for_extremes(self, data:np.ndarray=None, index:int=None):

        z = np.zeros(data.shape[0])
        z[1:-1] = self._curvature_change_indicator(data=data, index=index)

        indices = np.where(z < 0)[0]

        return indices

    def apply(self, gradient_input_data:np.ndarray=None, column:int=None):

        if column is not None:
            indices = self._get_indices_for_extremes(data=gradient_input_data, index=column)
            return (indices,)
        else:
            indices_list = list()
            for cln in range(gradient_input_data.shape[1]):
                indices = self._get_indices_for_extremes(data=gradient_input_data, index=cln)
                indices_list.append(indices)

            return tuple(indices_list)



