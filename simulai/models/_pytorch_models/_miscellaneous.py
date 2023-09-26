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

import warnings
from typing import List, Optional, Union

import numpy as np
import torch

from simulai.regression import SLFNN, ConvexDenseNetwork
from simulai.templates import NetworkTemplate, guarantee_device


############################
#### Abstract
############################
class ModelMaker:
    def __init__(self):
        pass


class MetaModel:
    def __init__(self):
        pass


############################
#### Improved Dense Network
############################


# Dense network with hidden encoders aimed at improving convergence
class ImprovedDenseNetwork(NetworkTemplate):
    name = "improveddense"
    engine = "torch"

    def __init__(
        self,
        network: ConvexDenseNetwork,
        encoder_u: NetworkTemplate = None,
        encoder_v: NetworkTemplate = None,
        devices: Union[str, list] = "cpu",
    ):
        """

        Improved DenseNetwork

        It uses auxiliary encoder networks in order to enrich
        the hidden spaces

        Parameters
        ----------

        network: ConvexDenseNetwork
            A convex dense network (it supports convex sum operations in the hidden spaces).
        encoder_u: NetworkTemplate
            First auxiliary encoder.
        encode_v: NetworkTemplate
            Second auxiliary encoder.
        devices: str
            Devices in which the model will be executed.

        """

        super(ImprovedDenseNetwork, self).__init__(devices=devices)

        # Guaranteeing the compatibility between the encoders and the branch and trunk networks
        n_hs = network.hidden_size
        eu_os = encoder_u.output_size
        ev_os = encoder_v.output_size

        self.output_size = network.output_size

        # Determining the kind of device to be used for allocating the
        # subnetworks used in the DeepONet model
        self.device = self._set_device(devices=devices)

        assert n_hs == eu_os == ev_os, (
            "The output of the encoders must have the same dimension"
            " of the network hidden size, but got"
            f" {eu_os}, {ev_os} and {n_hs}."
        )

        self.network = self.to_wrap(entity=network, device=self.device)
        self.encoder_u = self.to_wrap(entity=encoder_u, device=self.device)
        self.encoder_v = self.to_wrap(entity=encoder_v, device=self.device)

        self.add_module("network", self.network)
        self.add_module("encoder_u", self.encoder_u)
        self.add_module("encoder_v", self.encoder_v)

        self.weights = list()
        self.weights += self.network.weights
        self.weights += self.encoder_u.weights
        self.weights += self.encoder_v.weights

    def forward(
        self, input_data: Union[np.ndarray, torch.Tensor] = None
    ) -> torch.Tensor:
        """

        Forward step

        Parameters
        ----------

        input_data: Union[np.ndarray, torch.Tensor]
            Input dataset.

        Returns
        -------

        torch.Tensor
            The output after the network evaluation.

        """

        # Forward method execution
        v = self.encoder_v.forward(input_data=input_data)
        u = self.encoder_u.forward(input_data=input_data)

        output = self.to_wrap(entity=self.network.forward(input_data=input_data, u=u, v=v),
                              device=self.device)

        return output

    @guarantee_device
    def eval(self, input_data: Union[np.ndarray, torch.Tensor] = None) -> np.ndarray:
        """

        Forward step

        Parameters
        ----------

        input_data: Union[np.ndarray, torch.Tensor]
            Input dataset.

        Returns
        -------

        torch.Tensor
            The output after the network evaluation.

        """

        output = self.forward(input_data=input_data)

        return output.detach().cpu().numpy()

    def summary(self) -> None:
        """
        It prints a general view of the architecture.
        """

        print(self)

# Prototype for multi-fidelity network applied to time-series
class MultiNetwork(NetworkTemplate):

    def __init__(self, models_list:List[NetworkTemplate]=None,
                 delta_t:float=None, device:str='cpu') -> None:

        super(MultiNetwork, self).__init__()

        for i, model in enumerate(models_list):
            self.set_network(net=model, index=i)

        self.delta_t = delta_t
        self.device = device

    def set_network(self, net:NetworkTemplate=None, index:int=None) -> None:

        setattr(self, f"worker_{index}", net)

        self.add_module(f"worker_{index}", net)

    def _eval_interval(self, index:int=None, input_data:torch.Tensor=None) -> torch.Tensor:

        input_data = input_data[:, None]
        return getattr(self, f"worker_{index}").eval(input_data=input_data)

    def eval(self, input_data:np.ndarray=None) -> np.ndarray:

        eval_indices_float = input_data/delta_t
        eval_indices = np.where(eval_indices_float > 0,
                                np.floor(eval_indices_float - 1e-13).astype(int),
                                eval_indices_float.astype(int))

        eval_indices = eval_indices.flatten().tolist()

        input_data = input_data - self.delta_t*np.array(eval_indices)[:, None]

        return np.vstack([self._eval_interval(index=i, input_data=idata) \
                                             for i, idata in zip(eval_indices, input_data)])

    def summary(self):

        print(self)


# Mixture of Experts POC
class MoEPool(NetworkTemplate):
    def __init__(
        self,
        experts_list: List[NetworkTemplate],
        gating_network: Union[NetworkTemplate, callable] = None,
        input_size: int = None,
        devices: Union[list, str] = None,
        binary_selection: bool = False,
        hidden_size: Optional[int] = None,
    ) -> None:
        super(MoEPool, self).__init__()

        """
        Mixture of Experts

        Parameters
        ----------

        experts_list: List[NetworkTemplate]
            The list of neural networks used as experts. 
        gating_network: Union[NetworkTemplate, callable]
            Network or callable operation used for predicting
            weights associated to the experts.
        input_size: int
            The number of dimensions of the input.
        devices: Union[list, str]
            Device ("gpu" or "cpu") or list of devices in which
            the model is placed.
        binary_selection: bool
            The weights will be forced to be binary or not.
        hidden_size: Optional[int]
            If information about the experts hidden size is required, which occurs, 
            for instance, when they are ConvexDenseNetwork objects,
            it is necessary to define this argument.

        """

        # Determining the kind of device to be used for allocating the
        # subnetworks used in the DeepONet model
        self.device = self._set_device(devices=devices)

        self.experts_list = experts_list
        self.n_experts = len(experts_list)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.binary_selection = binary_selection
        self.is_gating_trainable = None

        if self.hidden_size == None:
            warnings.warn(
                "hidden_size is None. If you are using a convex model, as ConvexDenseNetwork,"
                + " it is better to provide a value for it."
            )

        # Gating (classifier) network/object
        # The default gating network is a single-layer fully-connected network
        if gating_network is None:
            gating_network = SLFNN(
                input_size=self.input_size,
                output_size=self.n_experts,
                activation="softmax",
            )

            self.gating_network = self.to_wrap(entity=gating_network,
                                               device=self.device)
        else:
            try:
                self.gating_network = self.to_wrap(entity=gating_network,
                                                   device=self.device)
            except:
                self.gating_network = gating_network
                print(
                    f"The object {self.gating_network} cannot be moved because is not a torch.nn.Module."
                )

        # Determining if the gating network is trainable or not
        if isinstance(self.gating_network, NetworkTemplate):
            self.is_gating_trainable = True
        else:
            self.is_gating_trainable = False
            # When the gating is not trainable, we consider that there
            # are a single expert choice for each sample which is already
            # chosen, in this case the selection is always binary
            self.binary_selection = True

        # Sending each sub-network to the correct device
        for ei, expert in enumerate(self.experts_list):
            self.experts_list[ei] = self.to_wrap(entity=expert, device=self.device)

        # Just the trainable objects need to be included as modules
        if self.is_gating_trainable is True:
            self.add_module("gating", self.gating_network)
        else:
            pass

        for ii, item in enumerate(self.experts_list):
            self.add_module(f"expert_{ii}", item)

        self.weights = sum([i.weights for i in self.experts_list], [])

        if self.is_gating_trainable is True:
            self.weights += self.gating_network.weights

        self.output_size = self.experts_list[-1].output_size

        # Selecting the method to be used for determining the
        # gating weights
        if self.is_gating_trainable is True:
            if self.binary_selection is True:
                self.get_weights = self._get_weights_binary
            else:
                self.get_weights = self._get_weights_bypass
        else:
            self.get_weights = self._get_weights_not_trainable

    def _get_weights_bypass(self, gating: torch.Tensor = None) -> torch.Tensor:
        """
        When the gating weights are trainable and no post-processing operation
        is applied over them.

        Parameters
        ----------

        gating: torch.Tensor
            The output of the gating operation.

        Returns
        -------

        torch.Tensor:
            The binary weights based on the clusters.

        """

        return gating

    def _get_weights_binary(self, gating: torch.Tensor = None) -> torch.Tensor:
        """
        Even when the gating weights are trainable, they can be forced to became
        binary.

        Parameters
        ----------

        gating: torch.Tensor
            The output of the gating operation.

        Returns
        -------

        torch.Tensor:
            The binary weights based on the clusters.

        """

        maxs = torch.max(gating, dim=1).values[:, None]

        return self.to_wrap(entity=torch.where(gating == maxs, 1, 0), device=self.device)

    def _get_weights_not_trainable(self, gating: torch.Tensor = None) -> torch.Tensor:
        """
        When the gating process is not trainable, it is considered some kind of
        clustering approach, which will return integers corresponding to the
        cluster for each sample in the batch

        Parameters
        ----------

        gating: torch.Tensor
            The output of the gating operation.

        Returns
        -------

        torch.Tensor:
            The binary weights based on the clusters.

        """

        batches_size = gating.shape[0]

        weights = torch.zeros(batches_size, self.n_experts)

        weights[
            np.arange(batches_size).astype(int).tolist(), gating.astype(int).tolist()
        ] = 1

        return self.to_wrap(entity=weights, device=self.device)

    def gate(self, input_data: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Gating (routing) the input, it means, attributing a weight for the output of
        each expert, which will be used for the allreduce operation executed on top
        of the MoE model.

        Parameters
        ----------

        input_data: Union[np.ndarray, torch.Tensor]
            The input data that will be gated and distributed among the experts.

        Returns
        -------

        torch.Tensor
            The penalties used for weighting the input distributed among the experts.
        """

        gating = self.gating_network.forward(input_data=input_data)
        gating_weights_ = self.get_weights(gating=gating)

        return gating_weights_

    # @guarantee_device
    def forward(
        self, input_data: Union[np.ndarray, torch.Tensor], **kwargs
    ) -> torch.Tensor:
        """
        Forward method

        Parameters
        ----------

        input_data: Union[np.ndarray, torch.Tensor]
            Data to be evaluated using the MoE object.
        kwargs: dict
            Used for bypassing arguments not defined in this model.

        Returns
        -------

        torch.Tensor
            The output of the MoE evaluation.
        """

        gating_weights_ = self.gate(input_data=input_data)

        gating_weights = torch.split(gating_weights_, 1, dim=1)

        def _forward(worker: NetworkTemplate = None) -> torch.Tensor:
            return worker.forward(input_data=input_data, **kwargs)

        output = list(map(_forward, self.experts_list))

        return sum([g * o for g, o in zip(gating_weights, output)])

    def summary(self) -> None:
        """
        It prints a general view of the architecture.
        """

        print(self)

# Splitting features among experts
class SplitPool(NetworkTemplate):

    def __init__(
        self,
        experts_list: List[NetworkTemplate],
        input_size: int = None,
        aggregation: Union[callable, NetworkTemplate] = None,
        last_activation: str = "relu",
        devices: Union[list, str] = None,
        hidden_size: Optional[int] = None,
    ) -> None:
        super(SplitPool, self).__init__()

        """
        Pool of experts to divide work

        Parameters
        ----------

        experts_list: List[NetworkTemplate]
            The list of neural networks used as experts. 
       input_size: int
            The number of dimensions of the input.
        devices: Union[list, str]
            Device ("gpu" or "cpu") or list of devices in which
            the model is placed.
       hidden_size: Optional[int]
            If information about the experts hidden size is required, which occurs, 
            for instance, when they are ConvexDenseNetwork objects,
            it is necessary to define this argument.

        """

        # Determining the kind of device to be used for allocating the
        # subnetworks used in the DeepONet model
        self.device = self._set_device(devices=devices)

        self.experts_list = experts_list
        self.n_experts = len(experts_list)
        self.input_size = input_size
        self.hidden_size = hidden_size

        if self.hidden_size == None:
            warnings.warn(
                "hidden_size is None. If you are using a convex model, as ConvexDenseNetwork,"
                + " it is better to provide a value for it."
            )

        # Sending each sub-network to the correct device
        for ei, expert in enumerate(self.experts_list):
            self.experts_list[ei] = self.to_wrap(entity=expert, device=self.device)

        for ii, item in enumerate(self.experts_list):
            self.add_module(f"expert_{ii}", item)

        self.weights = sum([i.weights for i in self.experts_list], [])

        self.output_size = self.experts_list[-1].output_size

        if not aggregation:
            self.aggregate = self._aggregate_default
        else:
            if isinstance(aggregation, NetworkTemplate):
                self.aggregate = self.to_wrap(entity=aggregation, device=self.device)
            else:
                self.aggregate = aggregation

        self.last_activation = self._get_operation(operation=last_activation)

    def _aggregate_default(self, output:List[torch.Tensor]) -> torch.Tensor:

        return torch.prod(output, dim=1, keepdim=True)

    def forward(
        self, input_data: Union[np.ndarray, torch.Tensor], **kwargs
    ) -> torch.Tensor:
        """
        Forward method

        Parameters
        ----------

        input_data: Union[np.ndarray, torch.Tensor]
            Data to be evaluated using the MoE object.
        kwargs: dict
            Used for bypassing arguments not defined in this model.

        Returns
        -------

        torch.Tensor
            The output of the SplitPool evaluation.
        """

        def _forward(worker: NetworkTemplate = None, index: int = None) -> torch.Tensor:
            return worker.forward(input_data=input_data[:, index][:, None], **kwargs)

        output = list(map(_forward, self.experts_list, list(np.arange(self.n_experts).astype(int))))

        output_ = torch.hstack(output)

        return self.last_activation(self.aggregate(output_))

    def summary(self) -> None:
        """
        It prints a general view of the architecture.
        """

        print(self)


