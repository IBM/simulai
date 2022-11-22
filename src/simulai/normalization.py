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
import h5py
import copy
from typing import Union

from simulai.batching import batchdomain_constructor
from simulai.abstract import Normalization
from simulai.metrics import MemorySizeEval

# It rescales to the interval [0, 1]
class UnitaryNormalization(Normalization):

    name = "unitary"

    def __init__(self, value_max:float=None, value_min:float=None) -> None:

        super().__init__()

        self.purpose = "normalization"

        if value_max and value_min:
            self.value_max = np.array(value_max)
            self.value_min = np.array(value_min)
        else:
            self.value_max = value_max
            self.value_min = value_min

        self.data_info_dict = dict()

        self.data_info_dict['input'] = {'max': self.value_max,
                                        'min': self.value_min}

        self.max_values = None
        self.min_values = None

    def _rescale(self, data_array:np.ndarray, tag:Union[str, int]) -> np.ndarray:

        if not self.value_max and not self.value_min:

            if len(data_array.shape) <= 2:
                data_array_max = data_array.max(0)
                data_array_min = data_array.min(0)
            else:
                indices = np.arange(len(data_array.shape)).tolist()
                ind = len(indices)-1
                indices.remove(ind)
                data_array_max = data_array.max(axis=tuple(indices))
                data_array_min = data_array.min(axis=tuple(indices))

            self.data_info_dict[tag] = {'max': data_array_max, 'min': data_array_min}

        else:

            data_array_max = self.value_max
            data_array_min = self.value_min

        data_array_transformed = (data_array - data_array_min)/(data_array_max - data_array_min)

        return data_array_transformed

    def _rescale_structured(self, data_array:np.recarray) -> np.recarray:

        arrays_list = list()

        for name in data_array.dtype.names:

            if not name in self.data_info_dict.keys():

                data_array_max = data_array[name].max()
                data_array_min = data_array[name].min()

                self.data_info_dict[name] = {'max': data_array_max, 'min': data_array_min}
            else:
                data_array_max = self.data_info_dict[name]['max']
                data_array_min = self.data_info_dict[name]['min']

            data_array_transformed = (data_array[name] - data_array_min)/(data_array_max - data_array_min)

            arrays_list.append(data_array_transformed)

        output_vars_names = ', '.join(data_array.dtype.names)

        data_array_transformed = np.core.records.fromarrays(arrays_list,
                                    names=output_vars_names,
                                    formats=','.join(len(data_array.dtype.names) * ['f8']))

        return data_array_transformed

    def rescale(self, map_dict:dict=None) -> dict:

        data_array_info_dict = dict()

        for key, data_array in map_dict.items():
            if data_array.dtype.names:
                data_array_transformed = self._rescale_structured(data_array)
            else:
                data_array_transformed = self._rescale(data_array, key)
            data_array_info_dict[key] = data_array_transformed

        return data_array_info_dict

    def apply_rescaling(self, map_dict:dict=None, eval:bool=False, axis:int=4) -> dict:

        data_rescaled_dict = dict()
        for key, data_array in map_dict.items():

            rescale_parameters = self.data_info_dict[key]

            if not eval:
                max_value = rescale_parameters['max']
                min_value = rescale_parameters['min']
            else:
                axis_list = np.arange(len(data_array.shape)).tolist()
                max_value = np.max(data_array, axis=axis_list.remove(axis))
                min_value = np.max(data_array, axis=axis_list.remove(axis))

            data_array_ = (data_array - min_value)/(max_value - min_value)

            data_rescaled_dict[key] = data_array_

        return data_rescaled_dict

    def apply_descaling(self, map_dict:dict=None) -> dict:

        data_rescaled_dict = dict()
        for key, data_array in map_dict.items():

            rescale_parameters = self.data_info_dict[key]

            max_value = rescale_parameters['max']
            min_value = rescale_parameters['min']

            data_array_ = data_array*(max_value - min_value) + min_value

            data_rescaled_dict[key] = data_array_

        return data_rescaled_dict

    def transform(self, data:np.ndarray=None, eval:bool=False, axis:int=4) -> np.ndarray:

        map_dict = {'input': data}
        data_transformed = self.apply_rescaling(map_dict=map_dict, eval=eval, axis=axis)

        return data_transformed['input']

    def apply_transform(self, data:np.ndarray=None) -> np.ndarray:

        return (data - self.min_values)/(self.max_values - self.min_values)

    def update_global_parameters(self, data:np.ndarray=None) -> None:

        indices = np.arange(len(data.shape)).tolist().remove(1)
        max_values = data.max(axis=indices)
        min_values = data.min(axis=indices)

        if not isinstance(self.max_values, np.ndarray):
            self.max_values = max_values
        elif max_values > self.max_values:
            self.max_values = max_values
        else:
            pass

        if not isinstance(self.min_values, np.ndarray):
            self.min_values = min_values
        elif min_values < self.min_values:
            self.min_values = min_values
        else:
            pass

# It rescales to the interval [-1, 1]
class UnitarySymmetricalNormalization(UnitaryNormalization):

    name = "unitary_symmetrical"

    def __init__(self, value_max:float=None, value_min:float=None) -> None:

        super().__init__(value_max=value_max, value_min=value_min)

    def rescale(self, map_dict:dict=None) -> dict:

        data_array_info_dict = dict()

        for key, data_array in map_dict.items():
            if data_array.dtype.names:
                data_array_transformed = self._rescale_structured(data_array)
            else:
                data_array_transformed = self._rescale(data_array, key)
            data_array_info_dict[key] = 2*data_array_transformed -1

        return data_array_info_dict

    def transform(self, data:np.ndarray=None, eval:bool=False, axis:int=4) -> np.ndarray:

        if eval:

            axis_ = np.arange(len(data.shape)).tolist()
            axis_.remove(axis)
            axis_ = tuple(axis_)

            if not isinstance(self.value_max, np.ndarray) and not isinstance(self.value_min, np.ndarray):

                self.value_max = data.max(axis_)
                self.value_min = data.min(axis_)
            else:
                pass

            return 2*(data - self.value_min)/(self.value_max - self.value_min) - 1

        else:
            data_transformed = super().transform(data=data, eval=eval, axis=axis)

            return 2*data_transformed - 1

    def transform_inv(self, data:np.ndarray=None) -> np.ndarray:

        inv_simmetrical_transform = (data + 1 )/2

        return (inv_simmetrical_transform + self.value_min)*(self.value_max - self.value_min)

class StandardNormalization(Normalization):

    name = "standard"

    def __init__(self):

        self.purpose = "normalization"

        self.data_info_dict = dict()

    def _rescale(self, data_array, tag):

        data_array_mean = data_array.mean(0)
        data_array_std = data_array.std(0)

        self.data_info_dict[tag] = {'mean': data_array_mean, 'std': data_array_std}
        data_array_transformed = (data_array - data_array_mean) / data_array_std

        return data_array_transformed

    def _rescale_structured(self, data_array):

        arrays_list = list()

        for name in data_array.dtype.names:

            if not name in self.data_info_dict.keys():

                data_array_mean = data_array[name].mean(0)
                data_array_std = data_array[name].std(0)

                self.data_info_dict[name] = {'mean': data_array_mean, 'std': data_array_std}
            else:
                data_array_mean = self.data_info_dict[name]['mean']
                data_array_std = self.data_info_dict[name]['std']

            data_array_transformed = (data_array[name] - data_array_mean) / data_array_std

            arrays_list.append(data_array_transformed)

        output_vars_names = ', '.join(data_array.dtype.names)

        data_array_transformed = np.core.records.fromarrays(arrays_list,
                                                            names=output_vars_names,
                                                            formats=','.join(len(data_array.dtype.names) * ['f8']))

        return data_array_transformed

    def rescale(self, map_dict=None):

        data_array_info_dict = dict()

        for key, data_array in map_dict.items():
            if data_array.dtype.names:
                data_array_transformed = self._rescale_structured(data_array)
            else:
                data_array_transformed = self._rescale(data_array, key)
            data_array_info_dict[key] = data_array_transformed

        return data_array_info_dict

    def apply_rescaling(self, map_dict=None):

        data_rescaled_dict = dict()
        for key, data_array in map_dict.items():
            rescale_parameters = self.data_info_dict[key]

            mean_value = rescale_parameters['mean']
            std_value = rescale_parameters['std']

            data_array_ = (data_array - mean_value) / std_value

            data_rescaled_dict[key] = data_array_

        return data_rescaled_dict

    def apply_descaling(self, map_dict=None):

        data_rescaled_dict = dict()
        for key, data_array in map_dict.items():
            rescale_parameters = self.data_info_dict[key]

            mean_value = rescale_parameters['mean']
            std_value = rescale_parameters['std']

            data_array_ = data_array * std_value + mean_value

            data_rescaled_dict[key] = data_array_

        return data_rescaled_dict


class BatchNormalization:

    def __init__(self, norm=None, channels_last=False):

        assert isinstance(norm, Normalization), "The norm must be a " \
                                                "simulai.normalization.Normalization object"
        self.norm = norm
        self.channels_last = channels_last

    def transform(self, data=None, data_interval=None,
                        batch_size=None, dump_path=None):

        assert isinstance(data, h5py.Dataset), "The input must be h5py.Dataset"

        variables_list = data.dtype.names
        data_shape = data.shape

        data_shape_ = list(copy.copy(data_shape))
        dims_ = list(range(0, len(data_shape_)))

        if not self.channels_last:

            channels = data_shape_[1]
            data_shape = tuple([data_shape_[0]] + data_shape_[2:] + [channels])
            dims = tuple([dims_[0]] + dims_[2:] + [dims_[1]])

        else:
            data_shape = data_shape_
            dims = dims_

        n_variables = data_shape[-1]

        data_file = h5py.File(dump_path, "w")
        dtype = [(var, '<f8') for var in variables_list]

        dset = data_file.create_dataset("normalized_data", shape=data_shape,
                                        dtype=dtype)

        if isinstance(batch_size, MemorySizeEval):
            n_samples = data_interval[1] - data_interval[0]
            batch_size = batch_size(max_batches=n_samples, shape=data.shape[1:])
        else:
            pass

        # Constructing the normalization  using the reference data
        batches = batchdomain_constructor(data_interval, batch_size)

        variables_names = data.dtype.names

        for batch_id, batch in enumerate(batches):

            chunk_data = data[slice(*batch)]
            print(f"Reading batch {batch_id+1}/{len(batches)} batch_size={chunk_data.shape[0]}. Updating global parameters.")

            chunk_data = chunk_data.view((float, len(data.dtype.names)))
            self.norm.update_global_parameters(data=chunk_data)

        for batch_id, batch in enumerate(batches):

            chunk_data = data[slice(*batch)].view((float, len(data.dtype.names)))

            if not self.channels_last:
                chunk_data = chunk_data.transpose(dims)
            else:
                pass

            normalized_chunk_data = self.norm.apply_transform(chunk_data)
            normalized_chunk_data = np.core.records.fromarrays(np.split(normalized_chunk_data[...],
                                                                        n_variables, axis=-1),
                                                                names=variables_names,
                                                                formats=','.join(len(variables_names) * ['f8']))
            dset[slice(*batch)] = normalized_chunk_data

        return dset


