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

from typing import Tuple, Union, Optional
import importlib
import warnings
import scipy.sparse as sparse

from simulai.abstract import Regression
from ._pytorch_network import NetworkTemplate

class ReservoirComputing(Regression):

    def __init__(self, reservoir_dim=None, sparsity_level=None):

        super().__init__()

        self.sparsity_tolerance = 0.0025  # Default choice

        self.reservoir_dim = reservoir_dim

        self.sparsity_level = sparsity_level

    @property
    def _reservoir_dim_corrected_sparsity_level(self):
        # Guaranteeing a minimum sparsity tolerance
        dim = max(self.sparsity_level, self.sparsity_tolerance) if self.reservoir_dim == 0 else self.reservoir_dim
        effective_sparsity = self.sparsity_level / dim
        if effective_sparsity < self.sparsity_tolerance:
            return self.sparsity_tolerance
        else:
            return self.sparsity_level / dim

    # It creates a sparse and randomly distributed reservoir matrix
    def create_reservoir(self, reservoir_dim=None):
        return sparse.rand(self.reservoir_dim, self.reservoir_dim,
                           density=self._reservoir_dim_corrected_sparsity_level)

class NetworkInstanceGen:

    def __init__(self,
                 architecture : str,
                 dim : str = None,
                 reduce_dimensionality : bool = True) -> None:

        self.reduce_dimensionality = reduce_dimensionality

        if architecture == 'dense':
            self.architecture = 'DenseNetwork'
        elif architecture == 'cnn':
            self.architecture = 'ConvolutionalNetwork'
        else:
            raise Exception(f"Option {architecture} for architecture is not supported." +
                            f" It must be 'dense' or 'cnn'")

        self.engine = 'simulai.regression'

        self.engine_module = importlib.import_module(self.engine)
        self.architecture_class = getattr(self.engine_module, self.architecture)

        self.divisor = 2
        self.multiplier = 2

        # Selecting the architecture generator function to be used.
        if architecture == 'cnn':
            if reduce_dimensionality == True:
                self.method_tag = '_reduce'
            else:
                self.method_tag = '_increase'
        else:
            self.method_tag = ''

        gen_network_name = f'_gen_{architecture}_network{self.method_tag}'

        self.gen_network = getattr(self, gen_network_name)

        # It is still hard-coded
        self.interp_tag = {'1d': 'linear',
                           '2d': 'bicubic',
                           '3d': 'trilinear'}

        if architecture == 'cnn':

            assert dim in ['1d', '2d'], "dim must be '1d' or '2d'. "

            self.dim = dim

            self.after_conv = 'maxpool' + self.dim
            self.before_conv = 'upsample'

            ### CNN specificities
            # Default number of channels used for the first layer of a convolutional
            # neural network
            self.channels = 16
            self.channels_multiplier = 2
            self.kernel_size = 3
            self.stride = 1
            self.pool_kernel_size = 2
            self.pool_stride = 2
            self.padding = 1
            self.scale_factor = 2
            self.mode = self.interp_tag.get(dim)
            self.channels_position = 1

        self.architecture_str = architecture

    def _gen_dense_network(self, input_dim : int = None,
                                 output_dim : int = None,
                                 activation : str = None,
                                 name : str = None) -> dict:

        assert type(input_dim) == int
        assert type(output_dim) == int
        assert type(activation) == str

        # Creating the list of units
        ref = input_dim
        result = input_dim
        units_list = list()

        if input_dim > output_dim:

            while (ref % self.divisor < ref) and (result > self.divisor * output_dim) :

                result, remainder = divmod(ref, self.divisor)
                ref = result
                units_list.append(result)

        else:

            while (result < int(output_dim / self.multiplier)):

                result *= self.multiplier

                units_list.append(result)

        units_list.append(output_dim)

        config_dict = {'layers_units' : units_list,
                       'activations' : activation,
                       'input_size' : input_dim,
                       'output_size' : output_dim,
                       'name' : name}

        return config_dict

    def _is_div_cnn_dims(self, dim: Tuple[int, ...]):

        reduce_dims = dim[2:]

        return all([idim % self.divisor == 0 for idim in reduce_dims])

    def _div_cnn_dims(self, dim: Tuple[int, ...]):

        reduce_dims = dim[2:]

        return dim[:2] + tuple([int(idim/self.divisor) for idim in reduce_dims])

    def _multiply_cnn_dims(self, dim: Tuple[int, ...]):

        reduce_dims = dim[2:]

        return dim[:2] + tuple([int(idim * self.multiplier) for idim in reduce_dims])

    def _gen_cnn_layer_increase_dimensionality(self, channels_in : int = None,
                                               channels_out: int = None) -> dict:

        if channels_out == None:
            channels_out =  int(channels_in / self.channels_multiplier)

        layer_input = {'in_channels': channels_in,
                       'out_channels': channels_out,
                       'kernel_size': self.kernel_size,
                       'stride': self.stride,
                       'padding': self.padding,
                       'before_conv': {'type': self.before_conv,
                                      'scale_factor': self.scale_factor,
                                      'mode': self.mode}
                       }

        return layer_input

    def _gen_cnn_layer_reduce_dimensionality(self, channels_in : int = None,
                                             channels_out: int = None) -> dict:

        if channels_out == None:
            channels_out = channels_in * self.channels_multiplier

        layer_input = {'in_channels': channels_in,
                       'out_channels': channels_out,
                       'kernel_size': self.kernel_size,
                       'stride': self.stride,
                       'padding': self.padding,
                       'after_conv': {'type': self.after_conv,
                                      'kernel_size': self.pool_kernel_size,
                                      'stride': self.pool_stride}
                       }

        return layer_input

    def _gen_cnn_network_reduce(self, input_dim : Tuple[int, ...] = None,
                                output_dim: Optional[Tuple[int, ...]] = None,
                                activation : str = None,
                                name : str = None,
                                **kwargs) -> dict:

        assert type(input_dim) == tuple
        assert type(activation) == str

        layers_list = list()
        ref_dim = input_dim
        channels = input_dim[self.channels_position]
        layer_count = 0

        while self._is_div_cnn_dims(ref_dim):

            if layer_count == 0:
                channels_out = self.channels
            else:
                channels_out = None

            layer =   self._gen_cnn_layer_reduce_dimensionality(channels_in=channels,
                                                                channels_out=channels_out)

            channels = layer['out_channels']

            layers_list.append(layer)

            ref_dim = self._div_cnn_dims(ref_dim)

            layer_count += 1

        config_dict = {'layers' : layers_list,
                       'activations' : activation,
                       'case' : self.dim,
                       'name' : name}

        config_dict.update(kwargs)

        return  config_dict

    def _gen_cnn_network_increase(self, input_dim : Tuple[int, ...] = None,
                                  output_dim : Tuple[int, ...] = None,
                                  activation : str = None,
                                  name : str = None,
                                  **kwargs) -> dict:

        assert type(input_dim) == tuple
        assert type(output_dim) == tuple
        assert type(activation) == str

        layers_list = list()
        ref_dim = input_dim
        channels = input_dim[self.channels_position]

        while not all([ii > jj for ii, jj in zip(self._multiply_cnn_dims(ref_dim)[2:], output_dim[2:])]):

            layer =  self._gen_cnn_layer_increase_dimensionality(channels_in=channels)

            channels = layer['out_channels']

            layers_list.append(layer)

            ref_dim = self._multiply_cnn_dims(ref_dim)

        config_dict = {'layers' : layers_list,
                       'activations' : activation,
                       'case' : self.dim,
                       'name' : name}

        config_dict.update(kwargs)

        return  config_dict

    def __call__(self, input_dim : Union[int, Tuple[int, ...]] = None,
                 output_dim : Union[int, Tuple[int, ...]] = None,
                 activation : str = None,
                 channels: int = None,
                 name : str = None,
                 **kwargs) -> NetworkTemplate:

        if name == None:
            name = 'net' + str(id(self))
        else:
            pass

        if self.architecture_str == 'cnn':

            if channels == None:
                warnings.warn("As no value was provided for 'channels'," +
                              f" the default value of {self.channels} is being used. ")
            else:
                self.channels = channels

        config_dict  =  self.gen_network(input_dim = input_dim,
                                         output_dim = output_dim,
                                         activation = activation,
                                         name = name, **kwargs)

        return self.architecture_class(**config_dict)


