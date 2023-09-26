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
import importlib
import math
import os
import warnings
from collections import OrderedDict
from functools import reduce
from typing import List, Tuple, Union

import numpy as np
import torch
from torch.nn.parameter import Parameter

from simulai import ARRAY_DTYPE
from simulai.abstract import Dataset, Regression
from simulai.file import SPFile
from simulai.residuals import SymbolicOperator
from simulai.templates import NetworkTemplate

# Basic built-in optimization toolkit for SimulAI


# Enforcing the input in the correct format, torch.Tensor or an iterable in which
# each element has this format.
def _convert_tensor_format(method):
    def inside(
        self,
        input_data: Union[dict, torch.Tensor, np.ndarray] = None,
        target_data: Union[torch.Tensor, np.ndarray] = None,
        validation_data: Tuple[Union[torch.Tensor, np.ndarray]] = None,
        **kwargs,
    ):
        input_data_ = None

        if isinstance(input_data, torch.Tensor):
            input_data_ = input_data

        elif callable(input_data):
            input_data_ = input_data

        elif isinstance(input_data, np.ndarray):
            input_data_ = torch.from_numpy(input_data.astype(ARRAY_DTYPE))

        elif isinstance(input_data, dict):
            input_data_ = dict()

            for key, item in input_data.items():
                if type(item) == np.ndarray:
                    input_data_[key] = torch.from_numpy(item.astype(ARRAY_DTYPE))

                else:
                    input_data_[key] = item

        else:
            raise Exception(
                "The input data must be numpy.ndarray, dict[np.ndarray], torch.tensor or h5py.Group."
            )

        if isinstance(target_data, np.ndarray):
            target_data_ = torch.from_numpy(target_data.astype(ARRAY_DTYPE))
        else:
            target_data_ = target_data

        if validation_data is not None:
            if isinstance(validation_data[0], np.ndarray):
                validation_data_ = tuple(
                    [torch.from_numpy(j.astype(ARRAY_DTYPE)) for j in validation_data]
                )
            else:
                validation_data_ = validation_data
        else:
            validation_data_ = validation_data

        return method(
            self,
            input_data=input_data_,
            target_data=target_data_,
            validation_data=validation_data_,
            **kwargs,
        )

    return inside


# It verifies if the loss function provided to Optimizer
# is the most proper for the given model
def _adjust_loss_function_to_model(
    model: NetworkTemplate = None, loss: str = None, physics_informed: bool = False
) -> None:
    import simulai.models as simmod

    special_losses = ["pirmse", "opirmse", "vaermse", "kaermse"]

    if physics_informed == True:
        if isinstance(model, simmod.DeepONet):
            recommended_loss = "opirmse"

        else:
            recommended_loss = "pirmse"

    else:
        if isinstance(model, simmod.AutoencoderVariational):
            recommended_loss = "vaermse"

        elif isinstance(model, simmod.AutoencoderKoopman):
            recommended_loss = "kaermse"

        elif isinstance(model, simmod.AutoencoderCNN) or isinstance(
            model, simmod.AutoencoderMLP
        ):
            recommended_loss = "wrmse"
        elif isinstance(model, simmod.MultiScaleAutoencoder):
            if model.kind_of_ae == "variational":
                    recommended_loss = "vaermse"
            else:
                    recommended_loss = "wrmse"
        else:
            if loss not in special_losses:
                recommended_loss = loss
            else:
                recommended_loss = ["rmse", "wrmse"]

    if recommended_loss != loss:
        raise Exception(
            f"The loss function used for this case ({loss})"
            + f" is not the recommended ({recommended_loss}). Please, redefine it."
        )


# Wrapper for basic back-propagation optimization
# algorithms
class Optimizer:
    def __init__(
        self,
        optimizer: str = None,
        early_stopping: bool = False,
        summary_writer: bool = False,
        shuffle: bool = True,
        lr_decay_scheduler_params: dict = None,
        params: dict = None,
        early_stopping_params: dict = None,
        checkpoint_params: dict = None,
    ) -> None:
        if "n_samples" in list(params.keys()):
            self.n_samples = params.pop("n_samples")
        else:
            self.n_samples = None

        self.optimizer = optimizer
        self.params = params

        self.early_stopping = early_stopping
        self.early_stopping_params = early_stopping_params
        self.checkpoint_params = checkpoint_params

        self.summary_writer = summary_writer

        self.shuffle = shuffle

        self.lr_decay_scheduler_params = lr_decay_scheduler_params
        self.lr_decay_scheduler = None

        self.optim_module_names = [
            "torch.optim",
            "simulai.optimization._builtin_pytorch",
        ]

        self.input_data_name = "input_data"
        self.optim_modules = [
            importlib.import_module(module) for module in self.optim_module_names
        ]
        self.optim_class = self._get_optimizer(optimizer=optimizer)
        self.get_data = self._get_vector_data

        self.losses_module = importlib.import_module("simulai.optimization")

        # Using early_stopping or not
        if self.early_stopping is True:
            self.stop_handler = self._early_stopping_handler

        else:
            self.stop_handler = self._bypass_stop_handler

        # Using summary writing (necessary for tensorboard), or not
        if self.summary_writer is True:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except:
                raise Exception(
                    "It is necessary to have tensorboard installed to use summary writing."
                )
            self.writer = SummaryWriter()
            self.summary_writer = self._summary_writer
        else:
            self.summary_writer = self._bypass_summary_writer

        # Determining the kind of sampling will be executed
        if self.shuffle:
            self.sampler = self._exec_shuffling

        else:
            self.sampler = self._no_shuffling

        # Using lr decay or not
        if self.lr_decay_scheduler_params is not None:
            self.lr_decay_handler = self._lr_decay_handler

        else:
            self.lr_decay_handler = self._bypass_lr_decay_handler

        # Using checkpoint or not
        if self.checkpoint_params is not None:
            if "checkpoint_frequency" in self.checkpoint_params.keys():
                self.checkpoint_frequency = self.checkpoint_params.pop(
                    "checkpoint_frequency"
                )
            else:
                raise Exception(
                    "Checkpoint frequency not defined. Please give a value for it."
                )

            self.checkpoint_handler = self._checkpoint_handler

        else:
            self.checkpoint_params = dict()
            self.checkpoint_handler = self._bypass_checkpoint_handler

        # When checkpoints are used, it is possible to overwrite them or
        # creating multiple checkpoints in different states
        overwrite_savepoint = lambda epoch: ""
        not_overwrite_savepoint = lambda epoch: f"_ckp_epoch_{epoch}"

        # Rules for overwritting or not checkpoints
        if "overwrite" in self.checkpoint_params.keys():
            overwrite = self.checkpoint_params.pop("overwrite")

            if overwrite == True:
                self.overwrite_rule = overwrite_savepoint
            else:
                self.overwrite_rule = not_overwrite_savepoint
        else:
            self.overwrite_rule = overwrite_savepoint

        self.validation_score = np.inf
        self.awaited_steps = 0
        self.accuracy_str = ""
        self.decay_frequency = None
        self.loss_states = None
        self.is_physics_informed = False

    def _verify_GPU_memory_availability(self, device: str = None):
        total = torch.cuda.get_device_properties(device).total_memory
        reserved = torch.cuda.memory_reserved(device)
        allocated = torch.cuda.memory_allocated(device)

        return total - reserved - allocated

    def _try_to_transfer_to_GPU(
        self, data: Union[dict, torch.Tensor], device: str = None
    ) -> None:
        available_GPU_memory = self._verify_GPU_memory_availability(device=device)

        if isinstance(data, dict):
            data_size = sum([t.element_size() * t.nelement() for t in data.values()])

            if data_size < available_GPU_memory:
                data_ = {k: t.to(device) for k, t in data.items()}
                print("Data transferred to GPU.")
                return data_
            else:
                print("It was not possible to move data to GPU: insufficient memory.")
                print(f"{available_GPU_memory} < {data_size}, in bytes")
                return data

        elif isinstance(data, torch.Tensor):
            data_size = data.element_size() * data.nelement()

            if data_size < available_GPU_memory:
                data_ = data.to(device)
                print("Data transferred to GPU.")
                return data_
            else:
                print("It was not possible to move data to GPU: insufficient memory.")
                print(f"{available_GPU_memory} < {data_size}, in bytes")
                return data
        else:
            return data

    def _seek_by_extra_trainable_parameters(
        self, residual: SymbolicOperator = None
    ) -> Union[list, None]:
        if hasattr(residual, "constants"):
            extra_parameters = [
                c
                for c in residual.trainable_parameters.values()
                if isinstance(c, Parameter)
            ]
            if extra_parameters:
                print("There are extra trainable parameters.")
            return extra_parameters
        else:
            return None

    def _get_lr_decay(self) -> Union[callable, None]:
        if self.lr_decay_scheduler_params is not None:
            name = self.lr_decay_scheduler_params.pop("name")
            self.decay_frequency = self.lr_decay_scheduler_params.pop("decay_frequency")

            lr_class = getattr(torch.optim.lr_scheduler, name)

            return lr_class

        else:
            return None

    def _exec_shuffling(self, size: int = None) -> torch.Tensor:
        return torch.randperm(size)

    def _summary_writer(self, loss_states: dict = None, epoch: int = None) -> None:
        for k, v in loss_states.items():
            loss = v[epoch]
            self.writer.add_scalar(k, loss, epoch)

    # It handles early-stopping for the optimization loop
    def _early_stopping_handler(self, val_loss_function: callable = None) -> None:
        loss = val_loss_function()
        self.accuracy_str = "acc: {}".format(loss)

        if loss < self.validation_score:
            self.validation_score = loss
            self.awaited_steps = 0
            return False

        elif (loss > self.validation_score) and (
            self.awaited_steps <= self.early_stopping_params["patience"]
        ):
            self.validation_score = loss
            self.awaited_steps += 1
            return False

        else:
            print("Early-stopping was actioned.")
            return True

    def _lr_decay_handler(self, epoch: int = None):
        if (epoch % self.decay_frequency == 0) and (epoch > 0):
            self.lr_decay_scheduler.step()

    def _checkpoint_handler(
        self,
        save_dir: str = None,
        name: str = None,
        model: NetworkTemplate = None,
        template: callable = None,
        compact: bool = False,
        epoch: int = None,
    ) -> None:
        if epoch % self.checkpoint_frequency == 0:
            tag = self.overwrite_rule(epoch)
            saver = SPFile(compact=compact)
            saver.write(
                save_dir=save_dir, name=name + tag, model=model, template=template
            )

    def _no_shuffling(self, size: int = None) -> torch.Tensor:
        return torch.arange(size)

    def _bypass_summary_writer(self, **kwargs) -> None:
        pass

    # Doing nothing to early-stopping
    def _bypass_stop_handler(self, **kwargs):
        return False

    # Doing nothing with lr
    def _bypass_lr_decay_handler(self, **kwargs):
        pass

    # Doing nothing to checkpoint
    def _bypass_checkpoint_handler(self, **kwargs):
        pass

    # When data is a NumPy array
    def _get_vector_data(
        self,
        dataset: Union[np.ndarray, torch.Tensor] = None,
        indices: np.ndarray = None,
    ) -> torch.Tensor:
        if dataset is None:
            return None
        elif isinstance(dataset, Dataset):
            return dataset()[indices]
        else:
            return dataset[indices]

    # When data is stored in a HDF5 dataset
    def _get_ondisk_data(
        self, dataset: callable = None, indices: np.ndarray = None
    ) -> torch.Tensor:
        return dataset(indices=indices)

    # Preparing the batches (converting format and moving to the correct device)
    # in a single batch optimization loop
    def _make_input_data(
        self, input_data: Union[dict, torch.Tensor], device="cpu"
    ) -> dict:
        if type(input_data) is dict:
            input_data_dict = {key: item.to(device) for key, item in input_data.items()}
        else:
            input_data_dict = {self.input_data_name: input_data.to(device)}

        return input_data_dict

    # Preparing the batches (converting format and moving to the correct device)
    def _batchwise_make_input_data(
        self,
        input_data: Union[dict, torch.Tensor],
        device="cpu",
        batch_indices: torch.Tensor = None,
    ) -> dict:
        if type(input_data) is dict:
            input_data_dict = {
                key: self.get_data(dataset=item, indices=batch_indices).to(device)
                for key, item in input_data.items()
            }
        else:
            input_data_dict = {
                self.input_data_name: self.get_data(
                    dataset=input_data, indices=batch_indices
                ).to(device)
            }

        return input_data_dict

    # Getting up optimizer from the supported engines
    def _get_optimizer(self, optimizer: str = None) -> torch.nn.Module:
        try:
            for optim_module in self.optim_modules:
                mod_items = dir(optim_module)
                mod_items_lower = [item.lower() for item in mod_items]

                if optimizer in mod_items_lower:
                    print(f"Optimizer {optimizer} found in {optim_module.__name__}.")
                    optimizer_name = mod_items[mod_items_lower.index(optimizer)]

                    return getattr(optim_module, optimizer_name)

                else:
                    print(
                        f"Optimizer {optimizer} not found in {optim_module.__name__}."
                    )
        except:
            raise Exception(
                f"There is no correspondent to {optimizer} in any known optimization module."
            )

    # Getting up loss function from the correspondent module
    def _get_loss(self, loss: str = None) -> callable:
        if type(loss) == str:
            name = loss.upper()
            return getattr(self.losses_module, name + "Loss")
        elif callable(loss):
            return loss
        else:
            return f"loss must be str or callable, but received {type(loss)}"

    # Single batch optimization loop
    def _optimization_loop(
        self,
        n_epochs: int = None,
        loss_function: callable = None,
        op: NetworkTemplate = None,
        loss_states: dict = None,
        validation_loss_function: callable = None,
    ) -> None:
        for epoch in range(n_epochs):
            self.optimizer_instance.zero_grad()
            self.optimizer_instance.step(loss_function)

            self.checkpoint_handler(model=op, epoch=epoch, **self.checkpoint_params)

            self.summary_writer(loss_states=loss_states, epoch=epoch)

            self.lr_decay_handler(epoch=epoch)

        self.loss_states = loss_states


    # Basic version of the mini-batch optimization loop
    # TODO It could be parallelized
    def _batchwise_optimization_loop(
        self,
        n_epochs: int = None,
        batch_size: int = None,
        loss: Union[str, type] = None,
        op: NetworkTemplate = None,
        input_data: torch.Tensor = None,
        target_data: torch.Tensor = None,
        validation_data: Tuple[torch.Tensor] = None,
        params: dict = None,
        device: str = "cpu",
    ) -> None:
        print("Executing batchwise optimization loop.")

        if isinstance(loss, str):
            loss_class = self._get_loss(loss=loss)
            loss_instance = loss_class(operator=op)
        else:
            assert isinstance(
                loss, type
            ), "The object provided is not a LossBasics object."
            loss_class = loss

            try:
                loss_instance = loss_class(operator=op)
            except:
                raise Exception(f"It was not possible to instantiate the class {loss}.")

        if validation_data is not None:
            validation_input_data, validation_target_data = validation_data
            validation_input_data = self._make_input_data(
                validation_input_data, device=device
            )
            validation_target_data = validation_target_data.to(device)

            val_loss_function = loss_instance(
                input_data=validation_input_data,
                target_data=validation_target_data,
                **params,
            )
        else:
            val_loss_function = None

        batches = np.array_split(
            np.arange(self.n_samples), int(self.n_samples / batch_size)
        )

        # Number of batchwise optimization epochs
        n_batch_epochs = len(batches)

        epoch = 0  # Outer loop iteration
        b_epoch = 0  # Total iteration
        stop_criterion = False

        # When using mini-batches, it is necessary to
        # determine the number of iterations for the outer optimization
        # loop
        if n_batch_epochs > n_epochs:
            n_epochs_global = 1
        else:
            n_epochs_global = int(math.ceil(n_epochs / n_batch_epochs))

        while epoch < n_epochs_global and stop_criterion == False:
            # For each batch-wise realization it is possible to determine a
            # new permutation for the samples
            samples_permutation = self.sampler(size=self.n_samples)

            for ibatch in batches:
                self.optimizer_instance.zero_grad()

                indices = samples_permutation[ibatch]
                input_batch = self._batchwise_make_input_data(
                    input_data, device=device, batch_indices=indices
                )
                target_batch = self.get_data(dataset=target_data, indices=indices)

                if target_batch is not None:
                    target_batch = target_batch.to(device)

                # Instantiating the loss function
                loss_function = loss_instance(
                    input_data=input_batch,
                    target_data=target_batch,
                    call_back=self.accuracy_str,
                    **params,
                )

                self.optimizer_instance.step(loss_function)

                self.summary_writer(
                    loss_states=loss_instance.loss_states, epoch=b_epoch
                )

                self.checkpoint_handler(
                    model=op, epoch=b_epoch, **self.checkpoint_params
                )

                self.lr_decay_handler(epoch=b_epoch)

                stop_criterion = self.stop_handler(val_loss_function=val_loss_function)

                b_epoch += 1

            epoch += 1

        if hasattr(loss_instance, "loss_states"):
            if all(
                [isinstance(item, list) for item in loss_instance.loss_states.values()]
            ):
                self.loss_states = {
                    key: np.hstack(value)
                    for key, value in loss_instance.loss_states.items()
                }

            else:
                self.loss_states = loss_instance.loss_states

    # Main fit method
    @_convert_tensor_format
    def fit(
        self,
        op: NetworkTemplate = None,
        input_data: Union[dict, torch.Tensor, np.ndarray, callable] = None,
        target_data: Union[torch.Tensor, np.ndarray, callable] = None,
        validation_data: Tuple[Union[torch.Tensor, np.ndarray, callable]] = None,
        n_epochs: int = None,
        loss: str = "rmse",
        params: dict = None,
        batch_size: int = None,
        device: str = "cpu",
        distributed: bool = False,
        use_jit: bool = False,
    ) -> None:
        # Verifying if the params dictionary contains Physics-informed
        # attributes
        extra_parameters = None
        if "residual" in params:
            self.is_physics_informed = True

            extra_parameters = self._seek_by_extra_trainable_parameters(
                residual=params["residual"]
            )

            if use_jit:
                try:
                    params["residual"] = torch.compile(params["residual"])
                except AttributeError:
                    pass
            else:
                pass

        _adjust_loss_function_to_model(
            model=op, loss=loss, physics_informed=self.is_physics_informed
        )

        # When using inputs with the format h5py.Dataset
        if callable(input_data) and callable(target_data):
            assert batch_size, (
                "When the input and target datasets are in disk, it is necessary to provide a "
                " value for batch_size."
            )

            self.get_data = self._get_ondisk_data
        else:
            pass

        # When target is None, it is expected a residual (Physics-Informed) training
        if target_data is None:
            assert "residual" in params, (
                "If target_data are not provided, residual must be != None "
                "in order to generate it."
            )

            assert callable(params["residual"]), (
                f"operator must be callable,"
                f" but received {type(params['operator'])}."
            )
        else:
            pass

        if "causality_preserving" in params.keys():
            assert self.shuffle == False, (
                "If the causality preserving algorithm is being used,"
                " no shuffling must be allowed when creating the mini-batches."
            )

        # When early-stopping is used, it is necessary to provide a validation dataset
        if self.early_stopping is True:
            assert validation_data is not None, (
                "If early-stopping is being used, it is necessary to provide a"
                "validation dataset via validation_data."
            )
        else:
            pass

        # Configuring the device to be used during the fitting process
        device_label = device
        if device == "gpu":
            if not torch.cuda.is_available():
                print("Warning: There is no GPU available, using CPU instead.")
                device = "cpu"
                device_label = "cpu"
            else:
                try:
                    device = "cuda:" + os.environ["LOCAL_RANK"]
                except KeyError:
                    device = "cuda"
                device_label = "gpu"
                print("Using GPU.")
        elif device == "cpu":
            print("Using CPU.")
        elif not device:
            device="cpu"
            print("Received None, but using cpu instead.")
        else:
            raise Exception(f"The device must be cpu or gpu, the device {device} is not supported.")

        if not "device" in params:
            params["device"] = device

        # In a multi-device execution, the optimizer must be properly instantiated to execute distributed tasks.
        if distributed == True:
            from torch.distributed.optim import DistributedOptimizer
            from torch.distributed.rpc import RRef

            optimizer_params = list()
            for param in op.parameters():
                optimizer_params.append(RRef(param))

            if extra_parameters is not None:
                optimizer_params += extra_parameters

            self.optimizer_instance = DistributedOptimizer(
                self.optim_class, optimizer_params, **self.params
            )

        else:
            # Guaranteeing the correct operator placement when using a single device
            op = op.to(device)

            # Trying to use the PyTorch JIT compilation
            if use_jit:
                try:
                    op = torch.compile(op)
                except AttributeError:
                    pass

            if extra_parameters is not None:
                optimizer_params = list(op.parameters()) + extra_parameters
                self.optimizer_instance = self.optim_class(
                    optimizer_params, **self.params
                )
            else:
                self.optimizer_instance = self.optim_class(
                    op.parameters(), **self.params
                )

        # Configuring LR decay, when necessary
        lr_scheduler_class = self._get_lr_decay()

        if lr_scheduler_class is not None:
            print(f"Using LR decay {lr_scheduler_class}.")
            self.lr_decay_scheduler = lr_scheduler_class(
                self.optimizer_instance, **self.lr_decay_scheduler_params
            )
        else:
            pass

        # If GPU is being used, try to completely allocate the dataset there.
        if device_label == "gpu":
            input_data = self._try_to_transfer_to_GPU(data=input_data, device=device)
            target_data = self._try_to_transfer_to_GPU(data=target_data, device=device)

        else:
            pass

        # Determining the kind of execution to be performed, batch-wise or not
        if batch_size is not None:
            # Determining the number of samples for each case
            # dictionary
            if type(input_data) is dict:
                key = list(input_data.keys())[0]
                self.n_samples = input_data[key].size()[0]

            # When using h5py.Group, the number of samples must be informed in the instantiation
            elif callable(input_data):
                assert self.n_samples is not None, (
                    "If the dataset is on disk, it is necessary"
                    "to inform n_samples using the dictionary params."
                )

            # other cases: torch.Tensor, np.ndarray
            else:
                self.n_samples = input_data.size()[0]

            self._batchwise_optimization_loop(
                n_epochs=n_epochs,
                batch_size=batch_size,
                loss=loss,
                op=op,
                input_data=input_data,
                target_data=target_data,
                validation_data=validation_data,
                params=params,
                device=device,
            )

        else:
            # In this case, the entire datasets are placed in the same device, CPU or GPU
            # The datasets are initially located on CPU
            input_data = self._make_input_data(input_data, device=device)

            # Target data is optional for some cases
            if target_data is not None:
                target_data = target_data.to(device)

            loss_class = self._get_loss(loss=loss)
            loss_instance = loss_class(operator=op)

            # Instantiating the loss function
            loss_function = loss_instance(
                input_data=input_data, target_data=target_data, **params
            )

            # Instantiating the validation loss function, if necessary
            if self.early_stopping is True:
                validation_input_data, validation_target_data = validation_data
                validation_loss_function = loss_instance(
                    input_data=validation_input_data,
                    target_data=validation_target_data,
                    **params,
                )
            else:
                validation_loss_function = None

            # Executing the optimization loop
            self._optimization_loop(
                n_epochs=n_epochs,
                loss_function=loss_function,
                op=op,
                loss_states=loss_instance.loss_states,
                validation_loss_function=validation_loss_function,
            )


# Interface for using scipy implemented optimizers
class ScipyInterface:
    def __init__(
        self,
        fun: NetworkTemplate = None,
        optimizer: str = None,
        optimizer_config: dict = dict(),
        loss: callable = None,
        loss_config: dict = None,
        device: str = "cpu",
        jac: str = None,
    ) -> None:
        # Configuring the device to be used during the fitting process
        device_label = device
        if device == "gpu":
            if not torch.cuda.is_available():
                print("Warning: There is no GPU available, using CPU instead.")
                device = "cpu"
                device_label = "cpu"
            else:
                try:
                    device = "cuda:" + os.environ["LOCAL_RANK"]
                except KeyError:
                    device = "cuda"
                device_label = "gpu"
                print("Using GPU.")
        elif device == "cpu":
            print("Using CPU.")
        else:
            raise Exception(f"The device must be cpu or gpu, but received: {device}")

        self.device = device
        self.engine = "scipy.optimize"
        self.engine_module = importlib.import_module(self.engine)
        self.minimization_method = "minimize"

        self.optimizer = getattr(self.engine_module, self.minimization_method)

        self.optimizer_config = optimizer_config or dict()
        self.optimizer_config["method"] = optimizer

        self.fun = fun
        self.loss = loss
        self.loss_config = loss_config or dict()

        self.operators_names = list(self.fun.state_dict().keys())

        self.operators_shapes = OrderedDict(
            {k: list(v.shape) for k, v in self.fun.state_dict().items()}
        )

        self.state_0 = self.fun.state_dict()

        intervals = np.cumsum(
            [0] + [np.prod(shape) for shape in self.operators_shapes.values()]
        )

        self.operators_intervals = [
            intervals[i : i + 2].tolist() for i in range(len(intervals) - 1)
        ]

        if jac:
            self.optimizer_config["jac"] = jac
            self.objective = self._fun_num
        else:
            self.optimizer_config["jac"] = True
            self.objective = self._fun

        # Determining default type
        if torch.get_default_dtype() == torch.float32:
            self.default_dtype = np.float32
        else:
            self.default_dtype = np.float64

    def _stack_and_convert_parameters(
        self, parameters: List[Union[torch.Tensor, np.ndarray]]
    ) -> np.ndarray:
        return np.hstack(
            [
                param.detach().numpy().astype(np.float64).flatten()
                for param in parameters.values()
            ]
        )

    def _update_and_set_parameters(self, parameters: np.ndarray) -> None:

        operators = [
            torch.from_numpy(
                parameters[slice(*interval)].reshape(shape).astype(self.default_dtype)
            ).to(self.device)
            for interval, shape in zip(
                self.operators_intervals, self.operators_shapes.values()
            )
        ]

        for opi, parameter in enumerate(self.fun.parameters()):
            parameter.data.copy_(operators[opi])

    def _exec_kwargs_forward(self, input_data: dict = None):
        return self.fun.forward(**input_data)

    def _exec_forward(self, input_data: Union[np.ndarray, torch.Tensor] = None):
        return self.fun.forward(input_data=input_data)

    def _fun_num(self, parameters: np.ndarray) -> Tuple[float]:
        self._update_and_set_parameters(parameters)

        closure = self.loss(self.input_data, self.target_data, **self.loss_config)
        loss = closure()

        return loss.detach().cpu().numpy().astype(np.float64)

    def _fun(self, parameters: np.ndarray) -> Tuple[float, np.ndarray]:
        # Setting the new values for the model parameters
        self._update_and_set_parameters(parameters)

        closure = self.loss(self.input_data, self.target_data, **self.loss_config)
        loss = closure()

        grads = [v.grad.detach().cpu().numpy() for v in self.fun.parameters()]

        gradients = np.hstack(
            [
                v.flatten()
                for v, shape in zip(grads, list(self.operators_shapes.values()))
            ]
        )

        return loss.detach().cpu().numpy().astype(np.float64), gradients.astype(
            np.float64
        )

    def fit(
        self,
        input_data: Union[dict, torch.Tensor, np.ndarray] = None,
        target_data: Union[torch.Tensor, np.ndarray] = None,
    ) -> None:
        parameters_0 = self._stack_and_convert_parameters(self.state_0)

        print(
            f"\nStarting ScipyInterface with method: {self.optimizer_config['method']}\n"
        )

        if isinstance(input_data, dict):
            self.exec_forward = self._exec_kwargs_forward
        else:
            self.exec_forward = self._exec_forward

        self.input_data = input_data

        self.target_data = target_data

        self.closure = self.loss(self.input_data, self.target_data, **self.loss_config)

        solution = self.optimizer(self.objective, parameters_0, **self.optimizer_config)

        self._update_and_set_parameters(solution.x)
