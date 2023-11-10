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

import random
import sys
from typing import Dict, List, Tuple, Union

import h5py
import numpy as np
import torch
from numpy.lib import recfunctions
from torch import Tensor

from simulai import ARRAY_DTYPE
from simulai.abstract import DataPreparer, Dataset
from simulai.batching import batchdomain_constructor, indices_batchdomain_constructor
from simulai.metrics import MemorySizeEval

"""
    BEGIN DataPreparer children
"""

# This class does nothing


class ByPassPreparer(DataPreparer):
    """ByPass class, it fills the DataPreparer blank, but does nothing."""

    name = "no_preparer"

    def __init__(self, channels_last: bool = False) -> None:
        super().__init__()

        self.channels_last = channels_last
        self.collapsible_shapes = None
        self.dtype = None

    def prepare_input_data(self, data: np.ndarray) -> np.ndarray:
        """Prepare input data.

        Args:
            data (np.ndarray):

        Returns:
            numpy.ndarray:
        Example:

            >>> import numpy as np
            >>> data = np.random.rand(5, 3, 4, 2)
            >>> preparer = ByPassPreparer()
            >>> prepared_data = preparer.prepare_input_data(data)
            >>> prepared_data.shape
            (5, 3, 4, 2)
        """
        self.collapsible_shapes = data.shape[1:]
        return data

    def prepare_output_data(self, data: np.ndarray) -> np.ndarray:
        """Prepare output data.

        Args:
            data (np.ndarray):

        Returns:
            numpy.ndarray: The output data in the original format

        Example:

            >>> import numpy as np
            >>> data = np.random.rand(5, 3)
            >>> preparer = ByPassPreparer()
            >>> prepared_data = preparer.prepare_output_data(data)
            >>> prepared_data.shape
            (5, 3)
        """

        return data

    def prepare_input_structured_data(self, data: np.recarray) -> np.ndarray:
        """Prepare structured input data by converting it to an ndarray.

        Args:
            data (np.recarray):

        Returns:
            np.ndarray: numpy ndarray version of the input data.

        Note:
            This function is used when the input data is in the form of a structured array and needs to be converted to a regular numpy ndarray.
        Example:

            >>> import numpy as np
            >>> data = np.array([(1, 'a', 0.5), (2, 'b', 0.6)], dtype=[('a', int), ('b', '|S1'), ('c', float)])
            >>> preparer = ByPassPreparer()
            >>> preparer.prepare_input_structured_data(data)
            array([[1, 'a', 0.5],
                   [2, 'b', 0.6]])
        """

        return data

    def prepare_output_structured_data(self, data: np.ndarray) -> np.recarray:
        """Prepare structured output data by converting it to a recarray.

        Args:
            data (np.ndarray):

        Returns:
            np.recarray: numpy recarray version of the output data.

        Note:
            This function is used when the output data needs to be in the form of a structured array and is currently in the form of a regular numpy ndarray.
        Example:

            >>> import numpy as np
            >>> data = np.array([[1, 'a', 0.5], [2, 'b', 0.6]])
            >>> preparer = ByPassPreparer()
            >>> preparer.prepare_output_structured_data(data)
            rec.array([(1, 'a', 0.5), (2, 'b', 0.6)],
            dtype=[('f0', '<i4'), ('f1', 'S1'), ('f2', '<f8')])
        """
        return data


class Reshaper(DataPreparer):
    """Reshaper converts n-dimensional arrays to two-dimensional ones, performing a simple reshaping operation F: (n0, n1, ..., nm) -> (n0, prod(n1, ..., nm))"""

    name = "reshaper"

    def __init__(self, channels_last: bool = False) -> None:
        super().__init__()
        self.channels_last = channels_last
        self.collapsible_shapes = None
        self.collapsed_shape = None
        self.dtype = None
        self.n_features = None

    def _set_shapes_from_data(self, data: np.ndarray = None) -> None:
        """

        Args:
            data (np.ndarray, optional): The input data to reshape. (Default value = None)
        Example:

            >>> reshaper = Reshaper()
            >>> reshaper._set_shapes_from_data(np.random.random((10,3,4,5)))
            >>> reshaper.collapsible_shapes
            (3, 4, 5)
        """

        self.collapsible_shapes = data.shape[1:]
        self.collapsed_shape = np.prod(self.collapsible_shapes).astype(int)
        self._is_recarray = data.dtype.names is not None
        if self._is_recarray:
            self.n_features = len(data.dtype.names) * self.collapsed_shape
        else:
            self.n_features = self.collapsed_shape

    def _prepare_input_data(self, data: np.ndarray = None) -> np.ndarray:
        """

        Args:
            data (np.ndarray, optional):  (Default value = None)

        Returns:
            np.ndarray:

        Note:
            This function reshapes the input data to (n0, prod(n1, ..., nm)) shape.
        Example:

            >>> reshaper = Reshaper()
            >>> data = np.random.random((10,3,4,5))
            >>> reshaper.prepare_input_data(data)
            array([[0.527, 0.936, ... , 0.812],
                  [0.947, 0.865, ... , 0.947],
                  ...,
                  [0.865, 0.947, ... , 0.865],
                  [0.947, 0.865, ... , 0.947]])
        """

        assert len(data.shape) > 1, "Error! data must have at least two dimensions"
        return data.reshape((data.shape[0], self.n_features))

    def prepare_input_data(self, data: Union[np.ndarray, np.recarray]) -> np.ndarray:
        """Prepare input data for reshaping.

        Args:
            data (Union[np.ndarray, np.recarray]):

        Returns:
            np.ndarray:

        Note:
            - If `data` is a structured numpy array, it will be passed to `_prepare_input_structured_data` function.
            - If `data` is a plain numpy array, it will be passed to `_prepare_input_data` function.
        Example:

            >>> reshaper = Reshaper()
            >>> input_data = np.random.rand(2, 3, 4)
            >>> reshaper.prepare_input_data(input_data)
            array([[ 0.948...,  0.276...,  0.967...,  0.564...],
                [ 0.276...,  0.948...,  0.564...,  0.967...],
                [ 0.276...,  0.948...,  0.564...,  0.967...],
                [ 0.948...,  0.276...,  0.967...,  0.564...],
                [ 0.276...,  0.948...,  0.564...,  0.967...],
                [ 0.276...,  0.948...,  0.564...,  0.967...]])
        """

        self._set_shapes_from_data(data)
        if self._is_recarray:
            return self._prepare_input_structured_data(data)
        else:
            return self._prepare_input_data(data)

    def _reshape_to_output(self, data: np.ndarray) -> np.ndarray:
        """Reshape the data to its original shape before reshaping.

        Args:
            data (np.ndarray):

        Returns:
            np.ndarray:

        Note:
            The original shape of the data is stored in `collapsible_shapes` attribute.
        Example:

            >>> reshaper = Reshaper()
            >>> input_data = np.random.rand(2, 3, 4)
            >>> reshaper._set_shapes_from_data(input_data)
            >>> reshaped_data = reshaper._reshape_to_output(input_data.flatten())
            >>> reshaped_data.shape
            (2, 3, 4)
        """

        return data.reshape((data.shape[0],) + self.collapsible_shapes)

    def _prepare_output_data(
        self, data: np.ndarray = None, single: bool = False
    ) -> np.ndarray:
        """Prepare the input data to be in the shape and format expected by the model.

        Args:
            data (np.ndarray, optional): The input data to be prepared, by default None
            single (bool, optional):  (Default value = False)

        Returns:
            np.ndarray: The prepared input data

        """
        if self._is_recarray:
            return self._prepare_output_structured_data(data)
        else:
            return self._reshape_to_output(data)

    def prepare_output_data(self, data: np.ndarray, single: bool = False) -> np.ndarray:
        """Prepare the input data to be in the shape and format expected by the model.

        Args:
            data (np.ndarray): The input data to be prepared
            single (bool, optional):  (Default value = False)

        Returns:
            np.ndarray: The prepared input data

        """
        return self._prepare_output_data(data)

    def _prepare_input_structured_data(self, data: np.recarray = None) -> np.ndarray:
        """Prepare the input structured data to be in the shape and format expected by the model.

        Args:
            data (np.recarray, optional):  (Default value = None)

        Returns:
            np.ndarray: The prepared input structured data

        """
        self.dtype = data.dtype
        self._set_shapes_from_data(data)
        data_ = recfunctions.structured_to_unstructured(data)
        reshaped_data_ = self._prepare_input_data(data_)
        return reshaped_data_

    def prepare_input_structured_data(self, data: np.recarray = None) -> np.ndarray:
        """Prepare the input structured data to be in the shape and format expected by the model.

        Args:
            data (np.recarray, optional):  (Default value = None)

        Returns:
            np.ndarray: The prepared input structured data

        """
        return self._prepare_input_structured_data(data)

    def prepare_output_structured_data(self, data: np.ndarray = None) -> np.recarray:
        """Prepare the output data to be in the shape and format expected by the user.

        Args:
            data (np.ndarray, optional):  (Default value = None)

        Returns:
            np.recarray: The prepared output structured data

        """
        return self._prepare_output_structured_data(data)

    def _prepare_output_structured_data(self, data: np.ndarray = None) -> np.recarray:
        """Prepare the output data to be in the shape and format expected by the user.

        Args:
            data (np.ndarray, optional):  (Default value = None)

        Returns:
            np.recarray: The prepared output structured data

        """
        data = data.reshape(
            (data.shape[0],) + self.collapsible_shapes + (len(self.dtype),)
        )
        output_data = recfunctions.unstructured_to_structured(data, self.dtype)
        output_data = self._reshape_to_output(output_data)
        return output_data


class ScalerReshaper(Reshaper):

    """ScalerReshaper is a class that inherits from the Reshaper class and performs additional scaling on the input data."""

    name = "scalerreshaper"

    def __init__(
        self, bias: float = 0.0, scale: float = 1.0, channels_last: bool = False
    ) -> None:
        """Reshaper converts n-dimensional arrays to two-dimensional ones, performing a
        simple reshaping operation F: (n0, n1, ..., nm) -> (n0, prod(n1, ..., nm))

        Args:
            bias (float, optional):  (Default value = 0.0)
            scale (float, optional):  (Default value = 1.0)
            channels_last (bool, optional):  (Default value = False)
        """
        super().__init__(channels_last=channels_last)
        self.bias = bias
        self.scale = scale

    def prepare_input_data(
        self, data: Union[np.ndarray, np.recarray] = None, *args, **kwargs
    ) -> np.ndarray:
        """Prepare the input data by subtracting the bias and scaling the data.

        Args:
            data (Union[np.ndarray, np.recarray], optional): The input data to be prepared (Default value = None)
            *args:
            **kwargs:

        Returns:
            np.ndarray: The prepared input data

        Note:
            If the input data is a structured array, the method 'prepare_input_structured_data' will be called instead.
        Example:

            >>> reshaper = ScalerReshaper(bias=10, scale=2)
            >>> reshaper.prepare_input_data(np.array([1, 2, 3]))
            array([-4.5, -3.5, -2.5])
        """

        if data.dtype.names is None:
            return super(ScalerReshaper, self).prepare_input_data(
                (data - self.bias) / self.scale, *args, **kwargs
            )
        else:
            return self.prepare_input_structured_data(data, *args, **kwargs)

    def prepare_output_data(
        self, data: Union[np.ndarray, np.recarray] = None, *args, **kwargs
    ) -> np.ndarray:
        """Prepare the output data by scaling it and adding the bias.

        Args:
            data (Union[np.ndarray, np.recarray], optional): The output data to be prepared (Default value = None)
            *args:
            **kwargs

        Returns:
            np.ndarray: The prepared output data

        Note:
            If the input data is a structured array, the method 'prepare_output_structured_data' will be called
        Example:

            >>> reshaper = ScalerReshaper(bias=10, scale=2)
            >>> reshaper.prepare_output_data(np.array([1, 2, 3]))
            array([12., 14., 16.])
        """

        if not self._is_recarray:
            return super(ScalerReshaper, self).prepare_output_data(
                data * self.scale + self.bias, *args, **kwargs
            )
        else:
            return self.prepare_output_structured_data(data)

    def _get_structured_bias_scale(self, dtype: np.dtype = None) -> Tuple[dict, dict]:
        """Get the bias and scale values for each field of a structured array.

        Args:
            dtype (np.dtype, optional):  (Default value = None)

        Returns:
            Tuple[dict, dict]: A tuple of two dictionaries, the first containing the bias values for each field and the second

        Note:
            If the bias and scale attributes are floats, they will be used for all fields.
        Example:

            >>> reshaper = ScalerReshaper(bias=10, scale=2)
            >>> reshaper._get_structured_bias_scale(np.dtype([('a', float), ('b', float)]))
            ({'a': 10, 'b': 10}, {'a': 2, 'b': 2})
        """

        bias = self.bias
        if isinstance(self.bias, float):
            bias = {n: self.bias for n in dtype.names}
        scale = self.scale
        if isinstance(self.scale, float):
            scale = {n: self.scale for n in dtype.names}

        return bias, scale

    def prepare_input_structured_data(
        self, data: np.recarray = None, *args, **kwargs
    ) -> np.ndarray:
        """Scale and reshape structured data (np.recarray) before passing it to the next layer.

        Args:
            data (np.recarray, optional): structured data to be transformed (Default value = None)
            *args (Additional arguments passed to the parent class):
            **kwargs

        Returns:
            np.ndarray:

        Note:
            The bias and scale parameters are expected to be provided in the form of dictionaries, where keys are field names and values are the corresponding bias and scale values for those fields.
        Example:

            >>> data = np.array([(1, 2, 3), (4, 5, 6)], dtype=[("a", int), ("b", int), ("c", int)])
            >>> reshaper = ScalerReshaper(bias={'a': 1, 'b': 2, 'c': 3}, scale={'a': 2, 'b': 3, 'c': 4})
            >>> reshaper.prepare_input_structured_data(data)
                array([[-0.5, 0.33333333, 0.75      ],
                    [ 1.5, 1.66666667, 2.        ]])
        """

        bias, scale = self._get_structured_bias_scale(data.dtype)
        data = data.copy()
        names = data.dtype.names
        for name in names:
            data[name] = (data[name] - bias[name]) / scale[name]
        return super(ScalerReshaper, self).prepare_input_structured_data(
            data, *args, **kwargs
        )

    def prepare_output_structured_data(
        self, data: np.ndarray = None, *args, **kwargs
    ) -> np.recarray:
        """Scale and reshape structured data (np.recarray) before passing it to the next layer.

        Args:
            data (np.ndarray, optional): structured data to be transformed (Default value = None)
            *args (Additional arguments passed to the parent class):
            **kwargs:

        Returns:
            np.recarray:

        Note:
            - The bias and scale parameters are expected to be provided in the form of dictionaries, where keys are field names and values are the corresponding bias and scale values for those fields.
        Example:

            >>> data = np.array([[-0.5, 0.33333333, 0.75      ],
            >>>                  [ 1.5, 1.66666667, 2.        ]])
            >>> reshaper = ScalerReshaper(bias={'a': 1, 'b': 2, 'c': 3}, scale={'a': 2, 'b': 3, 'c': 4})
            >>> reshaper.prepare_output_structured_data(data)
            rec.array([(0., 2.,  6.), (6., 8., 12.)],
                dtype=[('a', '<f8'), ('b', '<f8'), ('c', '<f8')])
        """

        bias, scale = self._get_structured_bias_scale(self.dtype)
        data = super(ScalerReshaper, self).prepare_output_structured_data(
            data, *args, **kwargs
        )
        data = data.copy()
        for name in self.dtype.names:
            data[name] = data[name] * scale[name] + bias[name]
        return data


class MapValid(Reshaper):
    """MapValid is a reshaper class that converts n-dimensional arrays to two-dimensional ones performing a valid values
    mapping operation F: F: data.shape = (n0, n1, ..., nm) -> data'.shape = (n0, n_valids)
    where n_valids is the number of valid elements in the data array.
    This class is useful for datasets in which there are invalid data.

    """

    name = "map_valid"

    def __init__(
        self, config: dict = None, mask=None, channels_last: bool = True
    ) -> None:
        """Initialize the MapValid class with the configurations and mask passed as parameters.

        Args:
            config (dict, optional): configurations dictionary, by default None
            mask (int, np.NaN, np.inf, optional, optional): mask to select the invalid values, by default None
            channels_last (bool, optional): if set to True, move the channel dimension to the last, by default True

        """
        super().__init__()

        self.default_dtype = "float64"

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

    def prepare_input_data(self, data: np.ndarray = None) -> np.ndarray:
        """Internal input data preparer, executed for each label of the structured array

        Args:
            data (np.ndarray, optional):  (Default value = None)

        Returns:
            np.ndarray:

        Note:
            - MapValid converts n-dimensional arrays to two-dimensional ones performing a valid values
            mapping operation F: F: data.shape = (n0, n1, ..., nm) -> data'.shape = (n0, n_valids)
            n_valids = dim([k in data[0, ...] if k != mask])
            - WARNING: the invalid positions are expected to be static in relation to n0.
        Example:

            >>> data = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
            >>> prepare_input_data(data)
            array([[1, 2, 3],
                   [5, 6, 7],
                   [9, 10, 11]])
        """

        data = super(MapValid, self).prepare_input_data(data)

        if self.mask == self.large_number:
            self.valid_indices_ = np.where(data[0, ...] < self.mask)

        elif not str(self.mask).isnumeric() or isinstance(self.mask, int):
            self.valid_indices_ = np.where(data[0, ...] != self.mask)

        else:
            raise Exception(
                "The chosen mask {} does not fit in any supported case".format(
                    self.mask
                )
            )

        samples_dim = data.shape[0]

        valid_indices = (slice(0, samples_dim),) + self.valid_indices_

        return data[valid_indices]

    def prepare_output_data(self, data: np.ndarray = None) -> np.ndarray:
        """Prepare output data for the MapValid operation.

        Args:
            data (np.ndarray, optional):  (Default value = None)

        Returns:
            np.ndarray:

        Note:
            - The reshaped data will have shape (n0, n_valids) where n0 is the number of samples and n_valids are the number of valid values in the data.
            - If the return_the_same_mask attribute is set to True, the mask used to select the invalid values will be returned. Otherwise, the reshaped data will be filled with NaN.
        Example:

            >>> import numpy as np
            >>> reshaper = MapValid()
            >>> data = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
            >>> reshaper.prepare_output_data(data)
            array([[[ 1.,  2.,  3.],
                    [ 4.,  5.,  6.]],
        """

        immutable_shape = data.shape[0]

        final_shape = (
            immutable_shape,
            self.n_features,
        )

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

    def prepare_input_structured_data(self, data: np.recarray = None) -> np.ndarray:
        """This function is used to prepare structured input data for further processing.

        Args:
            data (np.recarray, optional):  (Default value = None)

        Returns:
            np.ndarray:

        Note:
            This function is a wrapper function that calls the 'prepare_input_data' function internally.
        Example:

            >>> import numpy as np
            >>> data = np.array([(1, 2, 3), (4, 5, 6)], dtype=[('a', int), ('b', int), ('c', int)])
            >>> model = MapValid()
            >>> prepared_data = MapValid.prepare_input_structured_data(data)
            >>> prepared_data
            array([[1, 2, 3],
                  [4, 5, 6]])
        """

        return self.prepare_input_data(data)

    def prepare_output_structured_data(self, data: np.ndarray = None) -> np.ndarray:
        """This function is used to prepare structured output data for further processing.

        Args:
            data (np.ndarray, optional):  (Default value = None)

        Returns:
            np.ndarray:

        Note:
            This function is a wrapper function that calls the 'prepare_output_data' function internally.
        Example:

            >>> import numpy as np
            >>> data = np.array([[1, 2, 3], [4, 5, 6]])
            >>> model = MapValid()
            >>> prepared_data = MapValid.prepare_output_structured_data(data)
            >>> prepared_data
            array([[1, 2, 3],
                [4, 5, 6]])
        """

        return self.prepare_output_data(data)


class Sampling(DataPreparer):
    """This class is used for sampling data from the input dataset."""

    name = "sampling"

    def __init__(self, choices_fraction: float = 0.1, shuffling: bool = False) -> None:
        """Initializes the Sampling class.

        Args:
            choices_fraction (float, optional): The fraction of the dataset to be sampled, by default 0.1
            shuffling (bool, optional): Whether to shuffle the data before sampling, by default False

        """

        super().__init__()
        self.choices_fraction = choices_fraction
        self.shuffling = shuffling
        self.global_indices = None
        self.sampled_indices = None

    @property
    def indices(self) -> list:
        """Returns the indices of the data that have been sampled.

        Returns:
            list: The indices of the data that have been sampled.

        Raises:
            AssertionError: If the indices have not been generated yet.

        Note:
            The indices are generated by calling the 'prepare_input_data' or 'prepare_input_structured_data' functions.
        Example:

            >>> import numpy as np
            >>> data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
            >>> sampler = Sampling(choices_fraction=0.5, shuffling=True)
            >>> sampler.prepare_input_data(data)
            >>> sampler.indices
            [0, 1]
        """

        assert self.sampled_indices is not None, (
            "The indices still were not generate."
            "Run prepare_input_data or prepare_input_structured_data for getting them."
        )
        return sorted(self.sampled_indices.tolist())

    def prepare_input_data(
        self, data: np.ndarray = None, data_interval: list = None
    ) -> np.ndarray:
        """Prepare input data for sampling.

        Args:
            data (np.ndarray, optional): The input data. Default is None.
            data_interval (list, optional): The interval of data that should be selected. Default is None,

        Returns:
            numpy.ndarray: The sampled data.
        Note:
            The `data_interval` parameter must be a list of two integers, specifying the start and end of the interval.
         Example:

            >>> data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
            >>> data_interval = [3, 7]
            >>> input_data = sampler.prepare_input_data(data, data_interval)
        """

        if data_interval is None:
            data_interval = [0, data.shape[0]]
        n_samples = data_interval[1] - data_interval[0]

        self.global_indices = np.arange(start=data_interval[0], stop=data_interval[1])

        n_choices = int(self.choices_fraction * n_samples)

        self.sampled_indices = self.global_indices.copy()
        if self.shuffling:
            np.random.shuffle(self.sampled_indices)
        else:
            self.sampled_indices = self.sampled_indices

        self.sampled_indices = np.random.choice(self.sampled_indices, n_choices)

        return data[self.sampled_indices]

    def prepare_input_structured_data(
        self,
        data: h5py.Dataset = None,
        data_interval: list = None,
        batch_size: int = None,
        dump_path: str = None,
    ) -> np.recarray:
        """Prepares structured data for further processing.

        Args:
            data (h5py.Dataset, optional): Structured array to be prepared, the default shape is (n_samples, 1, *other_dimensions)
            data_interval (list, optional): The interval of the data to be prepared, the default shape is [0, data.shape[0]]
            batch_size (int, optional): The size of the batches to be processed, defaults to None
            dump_path (str, optional):  (Default value = None)

        Returns:
            np.recarray:

        Note:
            - The features dimensions of the input data should be 1 in NumPy structured arrays.
            - When using a h5py.Dataset as input, a dump_path must be provided
        Example:

            >>> data = h5py.File("path/to/data.h5", 'r')['data']
            >>> data_interval = [0, data.shape[0]]
            >>> batch_size = 32
            >>> dump_path = "path/to/dump.h5"
            >>> obj = PrepareInputStructuredData()
            >>> prepared_data = obj.prepare_input_structured_data(data, data_interval, batch_size, dump_path)
        """

        if data_interval is None:
            data_interval = [0, data.shape[0]]

        n_samples = data_interval[1] - data_interval[0]
        self.global_indices = np.arange(start=data_interval[0], stop=data_interval[1])

        n_sampled_preserved = int(self.choices_fraction * n_samples)
        self.sampled_indices = np.random.choice(
            self.global_indices, n_sampled_preserved, replace=False
        )

        if isinstance(data, h5py.Dataset):
            if isinstance(batch_size, MemorySizeEval):
                batch_size = batch_size(
                    max_batches=n_sampled_preserved, shape=data.shape[1:]
                )
            else:
                pass

            assert (
                dump_path
            ), "Using a h5py.Dataset as input data a dump_path must be provided."

            fp = h5py.File(dump_path, "w")
            sampled_data = fp.create_dataset(
                "data", shape=(n_sampled_preserved,) + data.shape[1:], dtype=data.dtype
            )

            # Constructing the normalization  using the reference data
            batches = indices_batchdomain_constructor(
                indices=self.sampled_indices, batch_size=batch_size
            )

            start_ix = 0
            for batch_id, batch in enumerate(batches):
                print(
                    f"Sampling batch {batch_id+1}/{len(batches)} batch_size={len(batch)}"
                )
                finish_ix = start_ix + len(batch)
                sampled_data[start_ix:finish_ix] = data[sorted(batch)]
                start_ix = finish_ix

            if self.shuffling:
                random.shuffle(sampled_data)

        else:
            raise Exception("Others cases are still not implemented.")

        return sampled_data


class MovingWindow:
    r"""MovingWindow is applied over a time-series array (2D array), and it is used for
    creating the necessary augmented data used for LSTM networks, replicating the training
    windows for each sample in the dataset.

    See a graphical example:

    Example:

        batch n
        ---------|---
        history  | horizon

            batch n+1
            ---------|---
            history  | horizon
        ----
        skip

    Example:

        >>> import numpy as np
        >>> data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        >>> window = MovingWindow(history_size=3, horizon_size=1)
        >>> window.transform(data)
        array([[1, 2, 3],
               [2, 3, 4],
               [3, 4, 5],
               [4, 5, 6],
               [5, 6, 7],
               [6, 7, 8],
               [7, 8, 9],
               [8, 9, 10]])
    """

    def __init__(
        self,
        history_size: int = None,
        skip_size: int = 1,
        horizon_size: int = None,
        full_output: bool = True,
    ) -> None:
        r"""Initializes the MovingWindow class

        Args:
            history_size (int, optional): the size of the history window, by default None
            skip_size (int, optional): the number of steps to skip between windows, by default 1
            horizon_size (int, optional): the size of the horizon window, by default None
            full_output (bool, optional): flag to use the full output or only the last item, by default True

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
        assert (
            history_size
        ), f"A value for history_size must be provided, not {history_size}"
        assert (
            horizon_size
        ), f"A value for horizon_size must be provided, not {horizon_size}"

    def transform(self, time_series: np.ndarray) -> np.ndarray:
        r"""Applies the moving window over the time_series array.

        Args:
            time_series (np.ndarray):

        Returns:
            np.ndarray: the transformed array with the windows.

        """
        return np.ndarray(time_series)

    def bypass(self, batch: np.ndarray) -> np.ndarray:
        r"""Does nothing, returns the input batch.

        Args:
            batch (np.ndarray):

        Returns:
            np.ndarray: the input array

        """
        return batch

    def get_last_item(self, batch: np.ndarray) -> np.ndarray:
        r"""Get the last item of a batch

        Args:
            batch (np.ndarray):

        Returns:
            np.ndarray:

        Note:
            - This method is used internally by the MovingWindow class
        Example:

            >>> data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
            >>> mw.get_last_item(data)
            array([[7, 8, 9]])
        """

        return batch[-1:]

    def __call__(
        self, input_data: np.ndarray = None, output_data: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        r"""Apply Moving Window over the input data

        Args:
            input_data (np.ndarray, optional): 2D array (time-series) to be used for constructing the history size (Default value = None)
            output_data (np.ndarray, optional):  (Default value = None)

        Returns:
            Tuple of np.ndarray: The tuple contains two arrays with shapes (n_samples, n_history, n_features) and

        Note:
            - It is expected that the input_data and output_data have the same shape
            - This method is used internally by the MovingWindow class
        Example:

            >>> data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]])
            >>> mw = MovingWindow(history_size=2, horizon_size=2, skip_size=1)
            >>> input_data, output_data = mw(data, data)
            >>> input_data
            array([[[1, 2, 3],
                    [4, 5, 6]],
                   [[4, 5, 6],
                    [7, 8, 9]],
                   [[7, 8, 9],
                    [10, 11, 12]]])
            >>> output_data
            array([[[ 7,  8,  9],
                    [10, 11, 12]],
                   [[10, 11, 12],
                    [13, 14, 15]]])
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
            input_batch = input_data[center - self.history_size : center, :]
            output_batch = output_data[center : center + self.horizon_size, :]

            input_batches_list.append(input_batch)
            output_batches_list.append(self.process_batch(batch=output_batch))

            center += self.skip_size

        input_data = np.stack(input_batches_list, 0)
        output_data = np.stack(output_batches_list, 0)

        return input_data, output_data


class SlidingWindow:
    r"""SlidingWindow is applied over a time-series array (2D array), and it is used for
    creating the necessary augmented data used for LSTM networks, replicating the training
    windows for each sample in the dataset. The difference between SlidingWindow and MovingWindow
    is that here there is no intersection between two sequential batches

    Attributes:
        history_size : int
            The number of history samples to include in each window.
        skip_size : int
            The number of samples to skip between each window.
    Note:
        - The difference between SlidingWindow and MovingWindow is that here there is no intersection between two sequential batches.

    See a graphical example:

    Example:

        batch n
        ---------|---
        history  | horizon

                        batch n+1
                        ---------|---
                            history  | horizon

    Example:

        >>> window = SlidingWindow(history_size=3, skip_size=1)
        >>> time_series = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        >>> windows = window.apply(time_series)
        >>> windows
        [[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7], [6, 7, 8], [7, 8, 9]]
    """

    def __init__(self, history_size: int = None, skip_size: int = None) -> None:
        r"""Initialize the SlidingWindow object.

        Args:
            history_size (int, optional): The number of history samples to include in each window. (Default value = None)
            skip_size (int, optional): The number of samples to skip between each window. (Default value = None)

        """

        self.history_size = history_size
        self.skip_size = skip_size

        # Verifying if history and horizon sizes was provided
        assert (
            history_size
        ), f"A value for history_size must be provided, not {history_size}"
        assert skip_size, f"A value for horizon_size must be provided, not {skip_size}"

    def apply(self, time_series: List[int]) -> List[List[int]]:
        r"""Applies the sliding window to the given time series.

        Args:
            time_series (List[int]):

        Returns:
            List[List[int]]:
        Example:

            >>> window = SlidingWindow(history_size=3, skip_size=1)
            >>> time_series = [1, 2, 3, 4, 5, 6, 7, 8, 9]
            >>> windows = window.apply(time_series)
            >>> windows
            [[[1, 2, 3], [4, 5, 6]], [[4, 5, 6], [7, 8, 9]], [[7, 8, 9], [10, 11, 12]]]
        """

        windowed_samples = []
        for i in range(0, len(time_series) - self.history_size - self.skip_size + 1):
            window = time_series[i : i + self.history_size + self.skip_size]
            windowed_samples.append(window)
        return windowed_samples

    def __call__(
        self, input_data: np.ndarray = None, output_data: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        r"""Applies a sliding window operation on the given time series and returns the windowed samples.

        Args:
            input_data (np.ndarray, optional): 2D array (time-series) to be used for constructing the history size (Default value = None)
            output_data (np.ndarray, optional):  (Default value = None)

        Returns:
            Tuple[np.ndarray, np.ndarray]: tuple of np.ndarray with shapes (n_samples, n_history, n_features) and (n_samples, n_horizon, n_features)

        Note:
            - history_size and horizon_size should be positive integers
            - history_size should be less than the length of input_data
            - input_data and output_data should have the same number of rows
        Example:

            >>> data = np.random.rand(10,3)
            >>> history_size = 3
            >>> horizon_size = 2
            >>> window = Window(history_size, horizon_size)
            >>> input_data, output_data = window(data)
            >>> input_data.shape
            (4, 3, 3)
            >>> output_data.shape
            (4, 2, 3)
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
            input_batch = input_data[center - self.history_size : center, :]
            output_batch = output_data[
                center - self.history_size + self.skip_size : center + self.skip_size, :
            ]

            input_batches_list.append(input_batch)
            output_batches_list.append(output_batch)

            center += self.skip_size

        input_data = np.stack(input_batches_list, 0)
        output_data = np.stack(output_batches_list, 0)

        return input_data, output_data


class IntersectingBatches:
    r"""IntersectingBatches is a class that is applied over a time-series array (2D array) to create batches of input data for training or testing purposes."""

    def __init__(
        self, skip_size: int = 1, batch_size: int = None, full: bool = True
    ) -> None:
        r"""Initializes the IntersectingBatches class

        Args:
            skip_size (int, optional): Number of samples to skip between two windows. (Default value = 1)
            batch_size (int, optional): Number of samples to use in each batch. (Default value = None)
            full (bool, optional): Whether to include the last batch or not, even if it's not full. (Default value = True)

        """
        assert (
            batch_size
        ), f"A value for horizon_size must be provided, not {batch_size}"

        self.skip_size = skip_size
        self.batch_size = batch_size
        self.full = full

    def get_indices(self, dim: int = None) -> np.ndarray:
        r"""It gets just the indices of the shifting

        Args:
            dim (int, optional): total dimension (Default value = None)

        Returns:
            np.ndarray: the shifted indices

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

    def __call__(self, input_data: np.ndarray = None) -> Union[list, np.ndarray]:
        r"""Applies the batching strategy to the input data.

        Args:
            input_data (np.ndarray, optional):  (Default value = None)

        Returns:
            Union[list, np.ndarray]: A list of batches or a single batch if `full` attribute is set to False.
        Note:
            - If the `full` attribute is set to True, the last batch will be included even if it's not full.
        Example:

            >>> input_data = np.array([[1,2,3], [4,5,6], [7,8,9], [10,11,12]])
            >>> batches = IntersectingBatches(skip_size=1, batch_size=2)
            >>> batches(input_data)
            [array([[1, 2, 3],
                [4, 5, 6]]),
            array([[4, 5, 6],
                [7, 8, 9]]),
            array([[ 7,  8,  9],
                [10, 11, 12]])]
        """

        input_batches_list = list()
        data_size = input_data.shape[0]
        center = 0

        # Loop for covering the entire time-series dataset constructing the
        # training windows
        while center + self.batch_size <= data_size:
            input_batch = input_data[center : center + self.batch_size]

            input_batches_list.append(input_batch)

            center += self.skip_size

        if self.full == True:
            return input_batches_list
        else:
            return np.vstack([item[-1] for item in input_batches_list])


class BatchwiseExtrapolation:
    r"""BatchwiseExtraplation uses a time-series regression model and inputs as generated by
    MovingWindow to continuously extrapolate a dataset.

    Attributes:
        time_id : int
    Examples::

        >>> import numpy as np
        >>> from sklearn.linear_model import LinearRegression
        >>> model = LinearRegression()
        >>> op = lambda state: model.predict(state)
        >>> auxiliary_data = np.random.rand(100, 10)
        >>> batchwise_extrapolation = BatchwiseExtrapolation(op=op, auxiliary_data=auxiliary_data)
        >>> init_state = np.random.rand(1, 10, 20)
        >>> history_size = 3
        >>> horizon_size = 2
        >>> testing_data_size = 10
        >>> extrapolation_dataset = batchwise_extrapolation(init_state, history_size, horizon_size, testing_data_size)
        >>> extrapolation_dataset.shape
    """

    def __init__(self, op: callable = None, auxiliary_data: np.ndarray = None) -> None:
        self.op = op
        self.auxiliary_data = auxiliary_data
        self.time_id = 0

    def _simple_extrapolation(
        self, extrapolation_dataset: np.ndarray, history_size: int = 0
    ) -> np.ndarray:
        r"""Given the current extrapolation dataset, use the last history_size number of rows to create the next state of the dataset.

        Args:
            extrapolation_dataset (np.ndarray): The current state of the extrapolation dataset.
            history_size (int, optional):  (Default value = 0)

        Returns:
            np.ndarray: The next state of the extrapolation dataset.

        """
        return extrapolation_dataset[None, -history_size:, :]

    def _forcing_extrapolation(
        self, extrapolation_dataset: np.ndarray, history_size: int = 0
    ) -> np.ndarray:
        return np.hstack(
            [
                extrapolation_dataset[-history_size:, :],
                self.auxiliary_data[self.time_id - history_size : self.time_id, :],
            ]
        )[None, :, :]

    def __call__(
        self,
        init_state: np.ndarray = None,
        history_size: int = None,
        horizon_size: int = None,
        testing_data_size: int = None,
    ) -> np.ndarray:
        r"""A function that performs the extrapolation of the time series.

        Args:
            init_state (np.ndarray, optional): initial state of the time series. It should have the shape (batch_size, history_size, n_series) (Default value = None)
            history_size (int, optional): the size of the history window used in the extrapolation. (Default value = None)
            horizon_size (int, optional): the size of the horizon window used in the extrapolation. (Default value = None)
            testing_data_size (int, optional):  (Default value = None)

        Returns:
            np.ndarray:

        Note:
            The number of series in the initial state must be equal to the number of series in the auxiliary data, if it is provided.
        Example:

            >>> model = BatchwiseExtrapolation()
            #Init state of the time series
            >>> init_state = np.random.random((1,20,3))
            >>> history_size = 10
            >>> horizon_size = 5
            >>> testing_data_size = 50
            #Calling the function
            >>> output = model(init_state, history_size, horizon_size, testing_data_size)
            >>> print(output.shape)
            #(50,3)
        """

        if isinstance(self.auxiliary_data, np.ndarray):
            n_series = self.auxiliary_data.shape[-1]
        else:
            n_series = 0

        current_state = init_state
        extrapolation_dataset = init_state[0, :, n_series:]
        self.time_id = history_size

        if isinstance(self.auxiliary_data, np.ndarray):
            assert (
                self.auxiliary_data.shape[-1] + n_series == init_state.shape[-1]
            ), "Number of series in the initial state must be {}".format(
                self.auxiliary_data.shape[-1]
            )

            current_state_constructor = self._forcing_extrapolation

        else:
            current_state_constructor = self._simple_extrapolation

        while (
            extrapolation_dataset.shape[0] - history_size + horizon_size
            <= testing_data_size
        ):
            extrapolation = self.op(current_state)
            extrapolation_dataset = np.concatenate(
                [extrapolation_dataset, extrapolation[0]], 0
            )
            current_state = current_state_constructor(
                extrapolation_dataset, history_size=history_size
            )

            log_str = "Extrapolation {}".format(self.time_id + 1 - history_size)
            sys.stdout.write("\r" + log_str)
            sys.stdout.flush()

            self.time_id += horizon_size

        extrapolation_dataset = extrapolation_dataset[history_size:, :]

        return extrapolation_dataset


# ND Time-series data preparation (MovingWindow, SlidingWindow and BatchExtrapolation)


class BatchCopy:
    r"""A class for copying data in batches and applying a transformation function."""

    def __init__(self, channels_last: bool = False) -> None:
        self.channels_last = channels_last

    def _single_copy(
        self,
        data: h5py.Dataset = None,
        data_interval: list = None,
        batch_size: int = None,
        dump_path: str = None,
        transformation: callable = lambda data: data,
    ) -> h5py.Dataset:
        r"""Copy data from a single h5py.Dataset to another h5py.Dataset in batches.

        Args:
            data (h5py.Dataset, optional):  (Default value = None)
            data_interval (list, optional): The interval of the data to be copied. (Default value = None)
            batch_size (int, optional): The size of the batch to be copied. (Default value = None)
            dump_path (str, optional): The path where the new h5py.Dataset will be saved. (Default value = None)
            transformation (callable, optional):  (Default value = lambda data: data)

        Returns:
            h5py.Dataset: The new h5py.Dataset after the copy process.

        Note:
            - Copy data from data_file.h5/data to data_copy.h5/data with a batch size of 1000:
            - The input must be an h5py.Dataset.
        Example:

            >>> data = h5py.File("data_file.h5", "r")
            >>> batch_copy = BatchCopy()
            >>> dset = batch_copy._single_copy(data=data["data"], data_interval=[0, 100000], batch_size=1000, dump_path="data_copy.h5")
        """

        assert isinstance(data, h5py.Dataset), "The input must be h5py.Dataset"

        variables_list = data.dtype.names
        data_shape = (data_interval[1] - data_interval[0],) + data.shape[1:]

        data_file = h5py.File(dump_path, "w")
        dtype = [(var, "<f8") for var in variables_list]

        dset = data_file.create_dataset("data", shape=data_shape, dtype=dtype)

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
            print(
                f"Copying batch {batch_id+1}/{len(batches)} batch_size={batch[1]-batch[0]}"
            )

            # The variables dimension is the last one
            if self.channels_last:
                # TODO this is a restrictive way of doing it. It must be more flexible.
                # .transpose((0, 4, 2, 3, 1))
                chunk_data = data[slice(*batch)].view((float, len(data.dtype.names)))
            # The variables dimension is the second one
            else:
                chunk_data = data[slice(*batch)].view((float, len(data.dtype.names)))

            chunk_data = np.core.records.fromarrays(
                np.split(chunk_data[...], n_variables, axis=-1),
                names=variables_names,
                formats=",".join(len(variables_names) * ["f8"]),
            )

            if len(chunk_data.shape) > len(dset.shape):
                chunk_data = np.squeeze(chunk_data, axis=-1)
            else:
                pass

            dset[slice(*d_batch)] = transformation(chunk_data[...])

        return dset

    def _multiple_copy(
        self,
        data: list = None,
        data_interval: list = None,
        batch_size: int = None,
        dump_path: str = None,
        transformation: callable = lambda data: data,
    ) -> h5py.Dataset:
        r"""Copy and concatenate multiple h5py.Dataset objects into a single h5py.Dataset object.

        Args:
            data (list, optional): A list of h5py.Dataset objects to be concatenated. (Default value = None)
            data_interval (list, optional): A list of two integers indicating the start and end index of the data to be concatenated. (Default value = None)
            batch_size (int, optional): The number of samples to be processed at a time. (Default value = None)
            dump_path (str, optional): The file path where the concatenated h5py.Dataset object will be saved. (Default value = None)
            transformation (callable, optional):  (Default value = lambda data: data)

        Returns:
            h5py.Dataset: The concatenated h5py.Dataset object.

        """

        assert all(
            [isinstance(di, h5py.Dataset) for di in data]
        ), "All inputs must be h5py.Dataset"

        variables_list = sum([list(di.dtype.names) for di in data], [])
        data_shape = (data_interval[1] - data_interval[0],) + data[0].shape[1:]

        data_file = h5py.File(dump_path, "w")
        dtype = [(var, "<f8") for var in variables_list]

        dset = data_file.create_dataset("data", shape=data_shape, dtype=dtype)

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
            print(
                f"Copying and concatenating the batches {batch_id+1}/{len(batches)} batch_size={batch[1] - batch[0]}"
            )

            # The variables dimension is the last one
            if self.channels_last:
                # TODO this is a restrictive way of doing it. It must be more flexible.
                chunk_data = np.stack(
                    [
                        di[slice(*batch)]
                        .view((float, len(di.dtype.names)))
                        .transpose((0, 4, 2, 3, 1))
                        for di in data
                    ],
                    axis=-1,
                )
            # The variables dimension is the second one
            else:
                chunk_data = np.stack(
                    [
                        di[slice(*batch)].view((float, len(di.dtype.names)))
                        for di in data
                    ],
                    axis=-1,
                )

            chunk_data = np.core.records.fromarrays(
                np.split(chunk_data[...], n_variables, axis=-1),
                names=variables_names,
                formats=",".join(len(variables_names) * ["f8"]),
            )

            if len(chunk_data.shape) > len(dset.shape):
                chunk_data = np.squeeze(chunk_data, axis=-1)
            else:
                pass

            dset[slice(*d_batch)] = transformation(chunk_data[...])

        return dset

    def copy(
        self,
        data: h5py.Dataset = None,
        data_interval: list = None,
        batch_size: int = None,
        dump_path: str = None,
        transformation: callable = lambda data: data,
    ) -> h5py.Dataset:
        r"""Copies the data from h5py.Dataset to a new h5py.Dataset file.
        It allows to apply a transformation function to the data.

        Args:
            data (h5py.Dataset, optional): input data to be copied (Default value = None)
            data_interval (list, optional): the range of the data to be copied (Default value = None)
            batch_size (int, optional): the size of the batches to be used to copy the data (Default value = None)
            dump_path (str, optional): the path of the file where the data will be copied (Default value = None)
            transformation (callable, optional):  (Default value = lambda data: data)

        Returns:
            h5py.Dataset: The copied data
        Note:
            - If the data is a list of h5py.Dataset, it will call the `_multiple_copy` function.
        Example:

            >>> data = h5py.File('data.h5', 'r')
            >>> data_interval = [0, 100]
            >>> batch_size = 1000
            >>> dump_path = 'copied_data.h5'
            >>> transformation = lambda x: x*2
            >>> copied_data = copy(data, data_interval, batch_size, dump_path, transformation)
        """

        if isinstance(data, list):
            return self._multiple_copy(
                data=data,
                data_interval=data_interval,
                batch_size=batch_size,
                dump_path=dump_path,
                transformation=transformation,
            )

        else:
            return self._single_copy(
                data=data,
                data_interval=data_interval,
                batch_size=batch_size,
                dump_path=dump_path,
                transformation=transformation,
            )


class MakeTensor:
    r"""This class is used to make torch tensors from numpy arrays or dictionaries.

    Args:
        input_names (List[str]): list of input names.
        output_names (List[str]): list of output names.

    Note:
        - input_tensors will be a list of tensors in case of numpy array and dictionary inputs.
        - The input_data should be numpy array with shape (batch_size, features_size) or dictionary with keys from input_names and values with shape (batch_size, features_size) if input_names and output_names are provided.
        - The input_data will be converted to float32 dtype.
        - The input_data will be put on the device specified by the device parameter, which defaults to 'cpu'.
        - If input_data is None, it will raise an exception.
    Example:

        # Creating a MakeTensor object with input and output names

        # Converting numpy array to torch tensor

        # Converting dictionary to torch tensors

        >>> mt = MakeTensor(input_names=["input_1", "input_2"], output_names=["output"])

        >>> input_data = np.random.randn(10, 3)
        >>> input_tensors = mt(input_data)

        >>> input_data = {"input_1": np.random.randn(10, 3), "input_2": np.random.randn(10, 4)}
        >>> input_tensors = mt(input_data)
    """

    def __init__(self, input_names=None, output_names=None):
        self.input_names = input_names
        self.output_names = output_names

    def _make_tensor(
        self, input_data: np.ndarray = None, device: str = "cpu"
    ) -> List[torch.Tensor]:
        r"""Convert input_data to a list of torch tensors.

        Args:
            input_data (np.ndarray, optional):  (Default value = None)
            device (str, optional):  (Default value = "cpu")

        Returns:
            List[torch.Tensor]: list of tensors.

        """
        inputs_list = list(torch.split(input_data, 1, dim=-1))

        for vv, var in enumerate(inputs_list):
            var.requires_grad = True
            var = var.to(device)
            inputs_list[vv] = var
            # var = var[..., None]

        return inputs_list

    def _make_tensor_dict(self, input_data: dict = None, device: str = "cpu") -> dict:
        r"""Convert input_data to a dictionary of torch tensors.

        Args:
            input_data (dict, optional):  (Default value = None)
            device (str, optional):  (Default value = "cpu")

        Returns:
            dict: dictionary of tensors.

        """
        inputs_dict = dict()

        for key, item in input_data.items():
            item.requires_grad = True
            item = item.to(device)
            inputs_dict[key] = item

        return inputs_dict

    def __call__(
        self,
        input_data: Union[np.ndarray, torch.Tensor, Dict[str, np.ndarray]] = None,
        device: str = "cpu",
    ) -> List[torch.Tensor]:
        r"""Make tensors from input_data.

        Args:
            input_data (Union[np.ndarray, torch.Tensor, Dict[str, np.ndarray]], optional): input data to be converted. (Default value = None)
            device (str, optional):  (Default value = "cpu")

        Returns:
            Union[List[torch.Tensor], dict]:

        Raises:
            - Exception:


        """

        if type(input_data) == np.ndarray:
            input_data = torch.from_numpy(input_data.astype(ARRAY_DTYPE))

            inputs_list = self._make_tensor(input_data=input_data, device=device)

            return inputs_list

        if type(input_data) == torch.Tensor:
            inputs_list = self._make_tensor(input_data=input_data, device=device)

            return inputs_list

        elif type(input_data) == dict:
            inputs_list = self._make_tensor_dict(input_data=input_data, device=device)

            return inputs_list

        else:
            raise Exception(
                f"The type {type(input_data)} for input_data is not supported."
            )


class GaussianNoise(Dataset):
    r"""GaussianNoise(stddev=0.01, input_data=None)
    A dataset that applies Gaussian noise to input data.

    Example:

        >>> import numpy as np
        >>> input_data = np.random.rand(100,100)
        >>> dataset = GaussianNoise(stddev=0.05, input_data=input_data)
        >>> dataset.size()
        (100, 100)
    """

    def __init__(
        self, stddev: float = 0.01, input_data: Union[np.ndarray, Tensor] = None
    ):
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
        return (1 + self.stddev * torch.randn(*self.data_shape)) * self.input_data

class Tokenizer:

    """Wrapper for multiple tokenization approaches"""

    def __init__(self, kind: str = "time_indexer"):
        """
        Args:
            kind (str): The kind of tokenization to be used. (Default value = "time_indexer")
        """
        self.kind = kind

        # Tokenizer selection
        if self.kind == "time_indexer":
            self.input_tokenizer = self._make_time_input_sequence
            self.target_tokenizer = self._make_time_target_sequence
        else:
            raise Exception(f"The tokenization option {self.kind} is not available.")

    def generate_input_tokens(self, input_data: Union[np.ndarray, torch.Tensor], **kwargs) -> torch.Tensor:

        """Generating the input sequence of tokens."""

        return self.input_tokenizer(input_data, **kwargs)
    
    def generate_target_tokens(self, target_data: Union[np.ndarray, torch.Tensor], **kwargs) -> torch.Tensor: 

        """Generating the target sequence of tokens."""

        return self.target_tokenizer(target_data, **kwargs)

    def _make_time_input_sequence(self,
        src: Union[np.ndarray, torch.Tensor], num_step:int=None, step:float=None
    ) -> Union[np.ndarray, torch.Tensor]:
        """Simple tokenization based on repeating samples
           and time-indexing them.
        Args:
            src (Union[np.ndarray, torch.Tensor]): The dataset to be tokenized.
            num_step (int): number of timesteps for each batch. (Default value: None)
            step (float): Size of the timestep. (Default value: None)
        Returns:
            Union[np.ndarray, torch.Tensor]: The tokenized input dataset.
        """

        dim = num_step
        src = np.repeat(np.expand_dims(src, axis=1), dim, axis=1)
        src_shape = src.shape
        src_shape_list = list(src_shape)
        src_shape_list[-1] += 1

        src_final = np.zeros(tuple(src_shape_list))
        src_final[:, :, :-1] = src

        for i in range(num_step):
            src_final[:, i, -1] += step * i

        return src_final[:-num_step + 1]

    def _make_time_target_sequence(self, 
        src: Union[np.ndarray, torch.Tensor], num_step:int=None) ->  Union[np.ndarray, torch.Tensor]:
        """Simple tokenization based on repeating samples
           and time-indexing them.
        Args:
            src (Union[np.ndarray, torch.Tensor]): The dataset to be tokenized.
            num_step (int): number of timesteps for each batch. (Default value: None)
        Returns:
            Union[np.ndarray, torch.Tensor]: The tokenized target dataset.
        """
        moving_window = MovingWindow(history_size=1, skip_size=1, horizon_size=num_step - 1) 
        input_data, output_data = moving_window(input_data=src, output_data=src)

        return np.concatenate([input_data, output_data], axis=1)
