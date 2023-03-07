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
import os
from typing import List, Union

import numpy as np
import torch
from torch.nn.parameter import Parameter

import simulai.activations as simulact


# Template for a generic neural network
class NetworkTemplate(torch.nn.Module):
    def __init__(self, name: str = None) -> None:
        super(NetworkTemplate, self).__init__()

        # Default choice for the model name
        if name == None:
            name = "nnet"

        self.name = name
        self.engine = torch.nn
        self.suplementary_engines = [torch, simulact]

        self.default_last_activation = None

        self.input_size = None
        self.output_size = None
        self.layers = None
        self.activations = None
        self.initializations = None

        self.shapes_dict = None

    @property
    def weights_l2(self) -> torch.Tensor:
        return sum([torch.norm(weight, p=2) for weight in self.weights])

    @property
    def weights_l1(self) -> torch.Tensor:
        return sum([torch.norm(weight, p=1) for weight in self.weights])

    @property
    def n_parameters(self):
        if hasattr(self, "weights"):
            return int(sum([np.product(i.shape) for i in self.weights]))
        else:
            raise Exception(f"Class {self} has no attribute self.weights.")

    def _set_device(self, devices: Union[str, list] = "cpu") -> str:
        device = None
        if type(devices) == list():
            raise Exception("In construction.")
        elif type(str):
            if devices == "gpu":
                if torch.cuda.is_available():
                    try:
                        device = "cuda:" + os.environ["LOCAL_RANK"]
                    except KeyError:
                        device = "cuda"
                else:
                    device = "cpu"
            else:
                device = "cpu"

        return device

    def _get_from_guest(self, **kwargs) -> None:
        search_dict = copy.copy(self.__dict__)
        search_dict.update(kwargs)

        for key, value in search_dict.items():
            if hasattr(value, "share_to_host"):
                share_to_host = value.share_to_host

                for k, v in share_to_host.items():
                    print(f"Sharing the key {k} to host with value {v}")
                    setattr(self, k, v)

    # Getting up activation if it exists
    def _get_operation(
        self, operation: str = None, is_activation: bool = True
    ) -> callable:
        mod_items = dir(self.engine)
        mod_items_lower = [item.lower() for item in mod_items]

        if operation in mod_items_lower:
            operation_name = mod_items[mod_items_lower.index(operation)]
            operation_class = getattr(self.engine, operation_name)

            if is_activation is True:
                return operation_class()
            else:
                return operation_class
        else:
            try:
                for engine in self.suplementary_engines:
                    res_ = getattr(engine, operation, None)

                    if hasattr(res_, "__mro__"):
                        if torch.nn.Module in res_.__mro__:
                            res = res_
                            print(f"Module {operation} found in {engine}.")
                            return res()
                        else:
                            print(f"Module {operation} not found in {engine}.")
                    else:
                        print(f"Module {operation} not found in {engine}.")

            except AssertionError:
                raise Exception(
                    f"There is no correspondent to {operation} in {self.engine}"
                )

    def _setup_activations(
        self, activation: Union[str, list] = None, n_layers: int = None
    ) -> (list, Union[str, list]):
        if n_layers == None:
            assert self.n_layers, "n_layers is not defined."
            n_layers = self.n_layers

        # It instantiates an operation x^l = \sigma(y^l), in which y^l
        # is the output of the previous linear operation.
        if isinstance(activation, str):
            activation_op = self._get_operation(operation=activation)

            return (
                (n_layers - 1) * [activation_op]
                + [self._get_operation(operation=self.default_last_activation)],
                (n_layers - 1) * [activation] + [self.default_last_activation],
            )

        elif isinstance(activation, list) and all(
            [isinstance(el, str) for el in activation]
        ):
            activations_list = list()
            for activation_name in activation:
                activation_op = self._get_operation(operation=activation_name)

                activations_list.append(activation_op())

            return activations_list, activation

        elif isinstance(activation, torch.nn.Module):
            return (
                (n_layers - 1) * [activation]
                + [self._get_operation(operation=self.default_last_activation)],
                (n_layers - 1) * [activation.name] + [self.default_last_activation],
            )

        else:
            raise Exception(
                "The activation argument must be str"
                f"or list(str), not {type(activation)}"
            )

    # Instantiating all the linear layers.
    def _setup_hidden_layers(self, last_bias: bool = True) -> List[torch.nn.Module]:
        layers = list()
        input_layer = self._setup_layer(
            self.input_size,
            self.layers_units[0],
            initialization=self.initializations[0],
            first_layer=True,
        )

        layers.append(input_layer)
        self.add_module(self.name + "_" + "input", input_layer)
        self.weights.append(input_layer.weight)

        for li, layer_units in enumerate(self.layers_units[:-1]):
            layer_op = self._setup_layer(
                layer_units,
                self.layers_units[li + 1],
                initialization=self.initializations[li + 1],
            )

            layers.append(layer_op)
            self.add_module(self.name + "_" + str(li), layer_op)
            self.weights.append(layer_op.weight)

        output_layer = self._setup_layer(
            self.layers_units[-1],
            self.output_size,
            initialization=self.initializations[-1],
            bias=last_bias,
        )

        layers.append(output_layer)
        self.add_module(self.name + "_" + str(li + 1), output_layer)
        self.weights.append(output_layer.weight)

        return layers

    # It converts torch.tensor to np.ndarray. It is used for applications
    # employing SciPy optimizers.
    def _numpy_layers(self) -> List[np.ndarray]:
        numpy_layers = [
            [layer.weight.detach().numpy(), layer.bias.detach().numpy()]
            for layer in self.layers
        ]

        return numpy_layers

    # It converts torch.tensor to np.ndarray. It is used for applications
    # employing SciPy optimizers in which gradients are required.
    def _numpy_grad_layers(self) -> List[np.ndarray]:
        numpy_layers = [
            [
                layer.weight.grad.detach().numpy(),
                layer.bias.grad.detach().numpy()[None, :],
            ]
            for layer in self.layers
        ]

        return numpy_layers

    # It stores all the model parameters into a single flatten array, a
    # requirement from the SciPy optimizers.
    def _make_stitch_idx(self) -> np.ndarray:
        stitch_indices = list()
        partitions = list()
        self.n_tensors = len(self.shapes)

        all_dof = 0
        for ii, shape in enumerate(self.shapes):
            n_dof = np.prod(shape)
            idx = np.reshape(np.arange(all_dof, all_dof + n_dof, dtype=np.int32), shape)
            stitch_indices.append(idx)
            all_dof += n_dof
            partitions += [ii] * n_dof

        return stitch_indices

    # It returns all the model parameters in a single array.
    def get_parameters(self) -> np.ndarray:
        layers = self._numpy_layers()

        return np.hstack([item.flatten() for item in sum(layers, [])])

    # It returns all the gradients of the model parameters in a single array.
    def get_gradients(self) -> np.ndarray:
        grads = self._numpy_grad_layers()

        return np.hstack([item.flatten() for item in sum(grads, [])])

    # Setting up values for the model parameters.
    def set_parameters(self, parameters=None) -> None:
        for ll, layer in enumerate(self.layers_map):
            self.layers[ll].weight = Parameter(
                data=torch.from_numpy(
                    parameters[self.stitch_idx[layer[0]]].astype("float32")
                ),
                requires_grad=True,
            )

            self.layers[ll].bias = Parameter(
                data=torch.from_numpy(
                    parameters[self.stitch_idx[layer[1]]].astype("float32")
                ),
                requires_grad=True,
            )

        # Making evaluation using the trained fully-connected network

    def eval(self, input_data: Union[np.ndarray, torch.Tensor] = None) -> np.ndarray:
        output_tensor = self.forward(input_data=input_data)

        # Guaranteeing the dataset location as CPU
        output_tensor = output_tensor.to("cpu")

        return output_tensor.detach().numpy()

    # It prints a summary of the network architecture.
    def summary(self, **kwargs) -> None:
        import pprint

        pprinter = pprint.PrettyPrinter(indent=2)

        print("Summary of the network properties:")

        print("Linear operations layers:\n")
        pprinter.pprint(self.layers)
        print("\n")
        print("Activations layers:\n")
        pprinter.pprint(self.activations_str)
        print("\n")
        print("Initializations at each layer:\n")
        pprinter.pprint(self.initializations)

        self.shapes_dict = {"layers": self.layers}

    def save(self, save_dir: str = None, name: str = None, device: str = None) -> None:
        # Moving all the tensors to the destiny device if necessary
        if device is not None:
            print(f"Trying to move all the tensors to the destiny device {device}.")
            for key, value in self.state_dict().items():
                self.state_dict()[key] = value.to(device)
            print("Moving concluded.")
        else:
            pass

        try:
            torch.save(self.state_dict(), os.path.join(save_dir, name + ".pth"))
        except Exception:
            print(f"It was not possible to save {self}")

    def load(self, save_dir: str = None, name: str = None, device: str = None) -> None:
        print(f"Trying to load for {device}")

        try:
            if device != None:
                self.load_state_dict(
                    torch.load(
                        os.path.join(save_dir, name + ".pth"),
                        map_location=torch.device(device),
                    )
                )
            else:
                self.load_state_dict(torch.load(os.path.join(save_dir, name + ".pth")))
        except Exception:
            print(
                f"It was not possible to load from {os.path.join(save_dir, name + '.pth')}"
            )


# Decorators
def as_tensor(method):
    def inside(self, input_data=None, **kwargs):
        if isinstance(input_data, torch.Tensor):
            return method(self, input_data, **kwargs)

        elif isinstance(input_data, np.ndarray):
            input_data_ = torch.from_numpy(input_data.astype("float32"))

            return method(self, input_data_, **kwargs)

        elif isinstance(input_data, list):
            input_data_ = torch.cat(input_data, dim=-1)

            return method(self, input_data_, **kwargs)

        else:
            raise Exception("The input data must be numpy.ndarray or torch.tensor.")

    return inside


def as_array(method):
    def inside(self, input_data=None):
        if isinstance(input_data, np.ndarray):
            return method(self, input_data)

        elif isinstance(input_data, torch.Tensor):
            input_data_ = input_data.detach().numpy()

            return method(self, input_data_)

        elif type(input_data) == list:
            input_data_ = torch.cat(input_data, dim=-1).detach().numpy()

            return method(self, input_data_)

        else:
            raise Exception("The input data must be numpy.ndarray or" "torch.tensor.")

    return inside


def guarantee_device(method):
    def inside(self, **kwargs) -> callable:
        kwargs_ = {
            key: torch.from_numpy(value.astype("float32")).to(self.device)
            for key, value in kwargs.items()
            if isinstance(value, np.ndarray) == True
        }

        kwargs_.update(
            {
                key: value
                for key, value in kwargs.items()
                if isinstance(value, np.ndarray) == False
            }
        )

        return method(self, **kwargs_)

    return inside


def channels_dim(method):
    def inside(self, input_data=None):
        if len(input_data.shape) < self.n_dimensions:
            return method(self, input_data=input_data[:, None, ...])
        else:
            return method(self, input_data=input_data)

    return inside


class ConvNetworkTemplate(NetworkTemplate):
    def __init__(self, name: str = None, flatten: bool = None) -> None:
        super(ConvNetworkTemplate, self).__init__()

        self.name = name
        self.flatten = flatten

        if flatten == True:
            self.flattener = self._flatten
        else:
            self.flattener = self._no_flatten

        # When no name is provided, it will employ a random number
        # as model name
        if self.name == None:
            self.name = id(self)

        self.args = []

        # The operation coming before or in the sequence of each convolution layer can be
        # a pooling ou a sampling
        self.before_conv_tag = ""
        self.after_conv_tag = ""

        self.samples_dim = None

        self.case = None

        self.interpolation_prefix = {"1d": "", "2d": "bi", "3d": "tri"}

        self.output_shape = None

        self.shapes_dict = None

    def _no_flatten(self, input_data: torch.Tensor = None) -> torch.Tensor:
        return input_data

    def _flatten(self, input_data: torch.Tensor = None) -> torch.Tensor:
        n_samples, n_channels = input_data.shape[:2]
        collapsible_dimensions = np.prod(input_data.shape[2:])

        return torch.reshape(
            input_data, (n_samples, n_channels * collapsible_dimensions)
        )

    def _setup_layers(self, layers_config: dict = None) -> (list, list, list):
        before_conv_layers = list()
        conv_layers = list()
        after_conv_layers = list()
        weights = list()

        # Configuring each layer
        for ll, layer_config in enumerate(layers_config):
            assert isinstance(layer_config, dict), (
                "Each entry of the layers list must be a dictionary,"
                f" but received {type(layer_config)}."
            )

            assert all(
                [i in layer_config for i in self.args]
            ), f"The arguments {self.args} must be defined."

            # If a post-conv operation is defined, instantiate it
            if self.before_conv_tag in layer_config:
                assert type(layer_config[self.before_conv_tag]) == dict, (
                    f"If the argument {self.before_conv_tag}"
                    f" is present, it must be dict,"
                    f" but received {type(self.before_conv_tag)}"
                )

                before_conv_layer_config = layer_config.pop(self.before_conv_tag)

                type_ = before_conv_layer_config.pop("type")
                after_conv_layer_ll_template = self._get_operation(
                    operation=type_, is_activation=False
                )

                before_conv_layer_ll = after_conv_layer_ll_template(
                    **before_conv_layer_config
                )

            # By contrast, it must be an identity operation
            else:
                before_conv_layer_ll = torch.nn.Identity()

            # If a post-conv operation is defined, instantiate it
            if self.after_conv_tag in layer_config:
                assert type(layer_config[self.after_conv_tag]) == dict, (
                    f"If the argument {self.after_conv_tag}"
                    f" is present, it must be dict,"
                    f" but received {type(self.after_conv_tag)}"
                )

                after_conv_layer_config = layer_config.pop(self.after_conv_tag)

                type_ = after_conv_layer_config.pop("type")
                after_conv_layer_ll_template = self._get_operation(
                    operation=type_, is_activation=False
                )

                after_conv_layer_ll = after_conv_layer_ll_template(
                    **after_conv_layer_config
                )

            # By contrast, it must be an identity operation
            else:
                after_conv_layer_ll = torch.nn.Identity()

            args = [layer_config.pop(arg) for arg in self.args]

            # The convolution layer itself
            conv_layer_ll = self.layer_template(*args, **layer_config)

            before_conv_layers.append(before_conv_layer_ll)
            conv_layers.append(conv_layer_ll)
            after_conv_layers.append(after_conv_layer_ll)

            # Setting up the individual modules to the global one
            self.add_module(self.name + "_before_conv_" + str(ll), before_conv_layer_ll)
            self.add_module(self.name + "_" + str(ll), conv_layer_ll)
            self.add_module(self.name + "_after_conv_" + str(ll), after_conv_layer_ll)

            weights.append(conv_layer_ll.weight)

        return before_conv_layers, conv_layers, after_conv_layers, weights

    def _corrrect_first_dim(self, shapes: list = None) -> list:
        shapes[0] = None

        return shapes

    # Merging the layers into a reasonable sequence
    def _merge(
        self,
        before_conv: list = None,
        conv: list = None,
        act: list = None,
        after_conv: list = None,
    ) -> list:
        merged_list = list()

        for h, i, j, k in zip(before_conv, conv, act, after_conv):
            merged_list.append(h)
            merged_list.append(i)
            merged_list.append(j)
            merged_list.append(k)

        return merged_list

    # It prints an overview of the network architecture
    def summary(
        self,
        input_data: Union[torch.Tensor, np.ndarray] = None,
        input_shape: list = None,
        device: str = "cpu",
        display: bool = True,
    ) -> None:
        import pprint
        from collections import OrderedDict

        # When no input data is provided, a list containing the shape is used for creating an
        # overview of the network architecture
        if type(input_data) == type(None):
            assert type(input_shape) == list, (
                "If no input data is provided, it is necessary" " to have input_shape."
            )

            input_shape[0] = 1

            input_data = torch.ones(*input_shape).to(device)

        else:
            pass

        if isinstance(input_data, np.ndarray):
            input_data = torch.from_numpy(input_data.astype("float32")).to(device)

        else:
            pass

        input_tensor_ = input_data
        shapes_dict = OrderedDict()

        for layer_id in range(len(self.conv_layers)):
            # Applying operations before convolution
            output_tensor_before_conv = self.before_conv_layers[layer_id](input_tensor_)

            if hasattr(self.after_conv_layers[layer_id], "_get_name"):
                input_shape = self._corrrect_first_dim(list(input_tensor_.shape))
                output_shape = self._corrrect_first_dim(
                    list(output_tensor_before_conv.shape)
                )

                shapes_dict[
                    f"{self.before_conv_layers[layer_id]._get_name()}_{layer_id}"
                ] = {"Input shape": input_shape, "Output shape": output_shape}

            # Applying  convolution operations
            output_tensor_conv = self.conv_layers[layer_id](output_tensor_before_conv)

            input_shape = self._corrrect_first_dim(
                list(output_tensor_before_conv.shape)
            )
            output_shape = self._corrrect_first_dim(list(output_tensor_conv.shape))

            shapes_dict[f"{self.conv_layers[layer_id]._get_name()}_{layer_id}"] = {
                "Input shape": input_shape,
                "Output shape": output_shape,
            }

            shapes_dict[f"Activation_{layer_id}"] = self.activations[layer_id]

            output_tensor_after_conv = self.after_conv_layers[layer_id](
                output_tensor_conv
            )

            # Applying operations before convolution
            if hasattr(self.after_conv_layers[layer_id], "_get_name"):
                input_shape = self._corrrect_first_dim(list(output_tensor_conv.shape))
                output_shape = self._corrrect_first_dim(
                    list(output_tensor_after_conv.shape)
                )

                shapes_dict[
                    f"{self.after_conv_layers[layer_id]._get_name()}_{layer_id}"
                ] = {"Input shape": input_shape, "Output shape": output_shape}

            input_tensor_ = output_tensor_after_conv

        if display == True:
            pprint.pprint(shapes_dict, indent=2)

        self.shapes_dict = shapes_dict

        output_size = list(shapes_dict.values())[-1]["Output shape"]
        self.input_size = list(shapes_dict.values())[0]["Input shape"]

        # When the network output is reshaped, it is necessary to correct the value of self.output_size
        if self.flatten == True:
            self.output_shape = tuple(output_size)
            self.output_size = int(np.prod(output_size[1:]))
        else:
            self.output_shape = tuple(output_size)
            self.output_size = output_size


# Template used for defining a hyperparameter training.
class HyperTrainTemplate:
    def __init__(self, trial_config: dict = None, set_type="hard"):
        self.trial_config = trial_config

        # The duality model/optimizer
        self.model = None
        self.optimizer = None

        self.set_type = set_type
        self._raw = True

        self.hard_set()
        self.soft_set()

    @property
    def raw(self):
        return self._raw

    def set_trial(self, trial_config=None):
        self.trial_config = trial_config
        set_method = getattr(self, self.set_type + "_set")
        set_method()

    def soft_set(self):
        self._set_optimizer()
        self._raw = False  # Now the model is already initialized

    def hard_set(self):
        self._set_model()
        self._set_optimizer()
        self._raw = False  # Now the model is already initialized

    def eval(self, input_data=None):
        return self.model.eval(input_data=input_data)

    def save_model(self, path: str = None):
        torch.save(self.model.state_dict(), path)

    def _set_model(self):
        # This method must be properly defined in children classes of
        # HyperTrainTemplate.
        pass

    def _set_optimizer(self):
        # This method must be properly defined in children classes of
        # HyperTrainTemplate.
        pass
