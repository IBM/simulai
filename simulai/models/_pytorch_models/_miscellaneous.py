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

        :param network: a convex dense network (it supports convex sum operations in the hidden spaces)
        :type: ConvexDenseNetwork
        :param encoder_u: first auxiliary encoder
        :type encoder_u: NetworkTemplate
        :param: encoder_v: second_auxiliary encoder
        :type encoder_u: NetworkTemplate
        :param devices: devices in which the model will be executed
        :type devices: str
        :return: nothing

        """

        super(ImprovedDenseNetwork, self).__init__()

        # Guaranteeing the compatibility between the encoders and the branch and trunk networks
        n_hs = network.hidden_size
        eu_os = encoder_u.output_size
        ev_os = encoder_v.output_size

        # Determining the kind of device to be used for allocating the
        # subnetworks used in the DeepONet model
        self.device = self._set_device(devices=devices)

        assert n_hs == eu_os == ev_os, (
            "The output of the encoders must have the same dimension"
            " of the network hidden size, but got"
            f" {eu_os}, {ev_os} and {n_hs}."
        )

        self.network = network.to(self.device)
        self.encoder_u = encoder_u.to(self.device)
        self.encoder_v = encoder_v.to(self.device)

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

        :param input_data: input dataset
        :type input_data: Union[np.ndarray, torch.Tensor]
        :return: operation evaluated over the input data
        :rtype: torch.Tensor

        """

        # Forward method execution
        v = self.encoder_v.forward(input_data=input_data)
        u = self.encoder_u.forward(input_data=input_data)

        output = self.network.forward(input_data=input_data, u=u, v=v).to(self.device)

        return output


# Mixture of Experts POC
class MoEPool(NetworkTemplate):
    def __init__(
        self,
        experts_list: List[NetworkTemplate],
        gating_network: [NetworkTemplate, callable] = None,
        input_size: int = None,
        devices: Union[list, str] = None,
        binary_selection: bool = False,
        hidden_size: Optional[int] = None,
    ) -> None:
        super(MoEPool, self).__init__()

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
                "hidden_size is None. If you are using a convex model, as ConvexDenseNetwork,"+
                " it is better to provide a value for it."
            )

        # Gating (classifier) network/object
        # The default gating network is a single-layer fully-connected network
        if gating_network is None:
            self.gating_network = SLFNN(
                input_size=self.input_size,
                output_size=self.n_experts,
                activation="softmax",
            ).to(self.device)

        else:
            try:
                self.gating_network = gating_network.to(self.device)
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
            self.experts_list[ei] = expert.to(self.device)

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
        return gating

    def _get_weights_binary(self, gating: torch.Tensor = None) -> torch.Tensor:
        maxs = torch.max(gating, dim=1).values[:, None]

        return torch.where(gating == maxs, 1, 0).to(self.device)

    # When the gating process is not trainable, it is considered some kind of
    # clustering approach, which will return integers corresponding to the
    # cluster for each sample in the batch
    def _get_weights_not_trainable(self, gating: torch.Tensor = None) -> torch.Tensor:
        batches_size = gating.shape[0]

        weights = torch.zeros(batches_size, self.n_experts)

        weights[
            np.arange(batches_size).astype(int).tolist(), gating.astype(int).tolist()
        ] = 1

        return weights.to(self.device)

    def gate(self, input_data: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        gating = self.gating_network.forward(input_data=input_data)
        gating_weights_ = self.get_weights(gating=gating)

        return gating_weights_

    # @guarantee_device
    def forward(
        self, input_data: Union[np.ndarray, torch.Tensor], **kwargs
    ) -> torch.Tensor:
        gating_weights_ = self.gate(input_data=input_data)

        gating_weights = torch.split(gating_weights_, 1, dim=1)

        def _forward(worker: NetworkTemplate = None) -> torch.Tensor:
            return worker.forward(input_data=input_data, **kwargs)

        output = list(map(_forward, self.experts_list))

        return sum([g * o for g, o in zip(gating_weights, output)])

    def summary(self) -> None:
        print(self)
