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

from typing import Union

import numpy as np
import torch

from simulai.regression import ConvexDenseNetwork
from simulai.templates import NetworkTemplate, guarantee_device

#####################
#### DeepONet family
#####################


class DeepONet(NetworkTemplate):
    name = "deeponet"
    engine = "torch"

    def __init__(
        self,
        trunk_network: NetworkTemplate = None,
        branch_network: NetworkTemplate = None,
        decoder_network: NetworkTemplate = None,  # The decoder network is optional and considered
        var_dim: int = 1,  # less effective than the output reshaping alternative
        devices: Union[str, list] = "cpu",
        product_type: str = None,
        model_id=None,
    ) -> None:
        """

        Classical Deep Operator Network (DeepONet), a deep learning version
        of the Universal Approximation Theorem

        :param trunk_network: subnetwork for processing the coordinates inputs
        :type trunk_network: NetworkTemplate
        :param branch_network: subnetwork for processing the forcing/conditioning inputs
        :type branch_network: NetworkTemplate
        :param decoder_network: subnetworks for converting the embedding to the output (optional)
        :type decoder_network: NetworkTemplate
        :param var_dim: number of output variables
        :type var_dim: int
        :param devices: devices in which the model will be executed
        :type devices: Union[str. list]
        :param product_type: type of product to execute in the embedding space
        :type product_type: str
        :param model_id: name for the model
        :type model_id: str
        :returns: nothing

        """

        super(DeepONet, self).__init__()

        # Determining the kind of device to be used for allocating the
        # subnetworks used in the DeepONet model
        self.device = self._set_device(devices=devices)

        self.trunk_network = trunk_network.to(self.device)
        self.branch_network = branch_network.to(self.device)

        self.add_module("trunk_network", self.trunk_network)
        self.add_module("branch_network", self.branch_network)

        if decoder_network is not None:
            self.decoder_network = decoder_network.to(self.device)
            self.add_module("decoder_network", self.decoder_network)
        else:
            self.decoder_network = decoder_network

        self.product_type = product_type
        self.model_id = model_id
        self.var_dim = var_dim

        # Checking up whether the output of each subnetwork are in correct shape
        assert self._latent_dimension_is_correct(self.trunk_network.output_size), (
            "The trunk network must have"
            " one-dimensional output , "
            "but received"
            f"{self.trunk_network.output_size}"
        )

        assert self._latent_dimension_is_correct(self.branch_network.output_size), (
            "The branch network must have"
            " one-dimensional output,"
            " but received"
            f"{self.branch_network.output_size}"
        )

        # Checking the compatibility of the subnetworks outputs for each kind of product being employed
        if self.product_type != "dense":
            output_branch = self.branch_network.output_size
            output_trunk = self.trunk_network.output_size

            # It checks if the inner product operation can be performed.
            assert output_branch == output_trunk, (
                f"The output dimensions for the sub-networks"
                f" trunk and branch must be equal but are"
                f" {output_branch}"
                f" and {output_trunk}"
            )
        else:
            output_branch = self.branch_network.output_size

            assert not output_branch % self.var_dim, (
                f"The number of branch latent outputs must"
                f" be divisible by the number of variables,"
                f" but received {output_branch}"
                f" and {self.var_dim}"
            )

        self.subnetworks = [
            net
            for net in [self.trunk_network, self.branch_network, self.decoder_network]
            if net is not None
        ]

        self.input_trunk = None
        self.input_branch = None

        self.output = None
        self.var_map = dict()

        # Checking up if the input of the decoder network has the correct dimension
        if self.decoder_network is not None:
            assert self.decoder_network.weights[0].shape[1] == 1, (
                "The decoder input is expected"
                " to have dimension (None, 1), but"
                f"received {self.decoder_network.weights[0].shape}"
            )
        else:
            pass

        # Selecting the correct forward approach to be used
        self._forward = self._forward_selector_()

        self.subnetworks_names = ["trunk", "branch"]

    def _latent_dimension_is_correct(self, dim: Union[int, tuple]) -> bool:
        """

        It checks if the latent dimension is consistent
        :param dim: latent space dimension
        :type dim: Union[int, tuple]
        :returns: the confirmation about the dimensionality correctness
        :rtype: bool

        """

        if type(dim) == int:
            return True
        elif type(dim) == tuple:
            if len(tuple) == 1:
                return True
            else:
                return False

    def _forward_decoder(
        self, output_trunk: torch.Tensor = None, output_branch: torch.Tensor = None
    ) -> torch.Tensor:
        """

        Forward method used when a decoder networks is present in the system

        :param output_trunk: the embedding generated by the trunk network
        :type output_trunk: torch.Tensor
        :param output_branch: the embedding generated by the branch network
        :type output_branch: torch.Tensor
        :returns: the product between the two embeddings
        :rtype: torch.Tensor

        """

        output_encoder = torch.sum(output_trunk * output_branch, dim=-1, keepdim=True)
        output = self.decoder_network.forward(output_encoder)

        return output

    def _forward_dense(
        self, output_trunk: torch.Tensor = None, output_branch: torch.Tensor = None
    ) -> torch.Tensor:
        """

        Forward method used when the embeddings are multiplied using a matrix-like product, it means, the trunk
        network outputs serve as "interpolation basis" for the branch outputs

        :param output_trunk: the embedding generated by the trunk network
        :type output_trunk: torch.Tensor
        :param output_branch: the embedding generated by the branch network
        :type output_branch: torch.Tensor
        :returns: the product between the two embeddings
        :rtype: torch.Tensor

        """

        latent_dim = int(output_branch.shape[-1] / self.var_dim)
        output_branch_reshaped = torch.reshape(
            output_branch, (-1, self.var_dim, latent_dim)
        )

        output = torch.matmul(output_branch_reshaped, output_trunk[..., None])
        output = torch.squeeze(output)

        return output

    def _forward_pointwise(
        self, output_trunk: torch.Tensor = None, output_branch: torch.Tensor = None
    ) -> torch.Tensor:
        """

        Forward method used when the embeddings are multiplied using a simple point-wise product, after that a
        reshaping is applied in order to produce multiple outputs

        :param output_trunk: the embedding generated by the trunk network
        :type output_trunk: torch.Tensor
        :param output_branch: the embedding generated by the branch network
        :type output_branch: torch.Tensor
        :returns: the product between the two embeddings
        :rtype: torch.Tensor

        """

        latent_dim = int(output_trunk.shape[-1] / self.var_dim)
        output_trunk_reshaped = torch.reshape(
            output_trunk, (-1, latent_dim, self.var_dim)
        )
        output_branch_reshaped = torch.reshape(
            output_branch, (-1, latent_dim, self.var_dim)
        )
        output = torch.sum(
            output_trunk_reshaped * output_branch_reshaped, dim=-2, keepdim=False
        )

        return output

    def _forward_vanilla(
        self, output_trunk: torch.Tensor = None, output_branch: torch.Tensor = None
    ) -> torch.Tensor:
        """

        Forward method used when the embeddings are multiplied using a simple point-wise product

        :param output_trunk: the embedding generated by the trunk network
        :type output_trunk: torch.Tensor
        :param output_branch: the embedding generated by the branch network
        :type output_branch: torch.Tensor
        :returns: the product between the two embeddings
        :rtype: torch.Tensor

        """

        output = torch.sum(output_trunk * output_branch, dim=-1, keepdim=True)

        return output

    def _forward_selector_(self) -> callable:
        """

        It selects the forward method to be used
        :returns: the callable corresponding to the required forward method
        :rtype: callable

        """

        if self.var_dim > 1:
            # The decoder network can be used for producing multidimensional outputs
            if self.decoder_network is not None:
                return self._forward_decoder

            # In contrast, a simple reshaping operation also can be used
            elif self.product_type == "dense":
                return self._forward_dense

            else:
                return self._forward_pointwise
        else:
            return self._forward_vanilla

    @property
    def _var_map(self) -> dict:
        # It checks all the data arrays in self.var_map have the same
        # batches dimension
        batches_dimensions = set([value.shape[0] for value in self.var_map.values()])

        assert (
            len(batches_dimensions) == 1
        ), "This dataset is not proper to apply shuffling"

        dim = list(batches_dimensions)[0]

        indices = np.arange(dim)

        np.random.shuffle(indices)

        var_map_shuffled = {key: value[indices] for key, value in self.var_map.items()}

        return var_map_shuffled

    @property
    def weights(self) -> list:
        return sum([net.weights for net in self.subnetworks], [])

    def forward(
        self,
        input_trunk: Union[np.ndarray, torch.Tensor] = None,
        input_branch: Union[np.ndarray, torch.Tensor] = None,
    ) -> torch.Tensor:
        """

        Wrapper forward method

        :param input_trunk: -
        :type input_trunk: Union[np.ndarray, torch.Tensor]
        :param input_branch: -
        :type input_branch: Union[np.ndarray, torch.Tensor]
        :returns: the result of all the hidden operations in the network
        :rtype: torch.Tensor

        """

        # Forward method execution
        output_trunk = self.trunk_network.forward(input_trunk).to(self.device)

        output_branch = self.branch_network.forward(input_branch).to(self.device)

        output = self._forward(output_trunk=output_trunk, output_branch=output_branch)

        return output

    @guarantee_device
    def eval(
        self,
        trunk_data: Union[np.ndarray, torch.Tensor] = None,
        branch_data: Union[np.ndarray, torch.Tensor] = None,
    ) -> np.ndarray:
        """

        It uses the network to make evaluations

        :param trunk_data: -
        :type trunk_data: Union[np.ndarray, torch.Tensor]
        :param branch_data: -
        :type branch_data: Union[np.ndarray, torch.Tensor]
        :returns: the result of all the hidden operations in the network
        :rtype: np.ndarray

        """

        output_tensor = self.forward(input_trunk=trunk_data, input_branch=branch_data)

        return output_tensor.cpu().detach().numpy()

    @guarantee_device
    def eval_subnetwork(
        self, name: str = None, input_data: Union[np.ndarray, torch.Tensor] = None
    ) -> np.ndarray:
        assert (
            name in self.subnetworks_names
        ), f"The name {name} is not a subnetwork of {self}."

        network_to_be_used = getattr(self, name + "_network")

        return network_to_be_used.forward(input_data).cpu().detach().numpy()

    def summary(self) -> None:
        print("Trunk Network:")
        self.trunk_network.summary()
        print("Branch Network:")
        self.branch_network.summary()


class ResDeepONet(DeepONet):
    name = "resdeeponet"
    engine = "torch"
    """The operation performed is: output = input_branch + D(param, input_branch)"""

    def __init__(
        self,
        trunk_network: NetworkTemplate = None,
        branch_network: NetworkTemplate = None,
        decoder_network: NetworkTemplate = None,  # The decoder network is optional and considered
        var_dim: int = 1,  # less effective than the output reshaping alternative
        devices: Union[str, list] = "cpu",
        product_type: str = None,
        residual: bool = True,
        multiply_by_trunk: bool = False,
        model_id=None,
    ) -> None:
        """

        Residual Deep Operator Network (DeepONet)

        :param trunk_network: subnetwork for processing the coordinates inputs
        :type trunk_network: NetworkTemplate
        :param branch_network: subnetwork for processing the forcing/conditioning inputs
        :type branch_network: NetworkTemplate
        :param decoder_network: subnetworks for converting the embedding to the output (optional)
        :type decoder_network: NetworkTemplate
        :param var_dim: number of output variables
        :type var_dim: int
        :param devices: devices in which the model will be executed
        :type devices: Union[str. list]
        :param residual: consider the DeepONet as a residual layer (sum the output to the branch input) or not.
        :type residual: bool
        :param multiply_by_trunk: multiply the output by the trunk input or not. NOTE: if the option 'residual'
                                  is activated it is performed after the multiplication:
                                   output*trunk_input + branch_input
        :param multiply_by_trunk: bool
        :param product_type: type of product to execute in the embedding space
        :type product_type: str
        :param model_id: name for the model
        :type model_id: str
        :returns: nothing

        """

        super(ResDeepONet, self).__init__(
            trunk_network=trunk_network,
            branch_network=branch_network,
            decoder_network=decoder_network,  # The decoder network is optional and considered
            var_dim=var_dim,  # less effective than the output reshaping alternative
            devices=devices,
            product_type=product_type,
            model_id=model_id,
        )

        input_dim = self.branch_network.input_size

        self.forward_ = super().forward

        if residual == True:
            assert input_dim == var_dim, (
                "For a residual network, it is necessary to have "
                "size of branch_network input equal to var_dim, but "
                f"received {input_dim} and {var_dim}."
            )
            self.forward = self._forward_default

        elif multiply_by_trunk == True:
            self.forward = self._forward_multiplied_by_trunk
        else:
            self.forward = self._forward_cut_residual

    def _forward_default(
        self,
        input_trunk: Union[np.ndarray, torch.Tensor] = None,
        input_branch: Union[np.ndarray, torch.Tensor] = None,
    ) -> torch.Tensor:
        """

        Wrapper forward method

        :param input_trunk: -
        :type input_trunk: Union[np.ndarray, torch.Tensor]
        :param input_branch: -
        :type input_branch: Union[np.ndarray, torch.Tensor]
        :returns: the result of all the hidden operations in the network
        :rtype: torch.Tensor

        """

        output_residual = self.forward_(
            input_trunk=input_trunk, input_branch=input_branch
        )

        return input_branch + output_residual

    def _forward_multiplied_by_trunk(
        self,
        input_trunk: Union[np.ndarray, torch.Tensor] = None,
        input_branch: Union[np.ndarray, torch.Tensor] = None,
    ) -> torch.Tensor:
        """

        Wrapper forward method

        :param input_trunk: -
        :type input_trunk: Union[np.ndarray, torch.Tensor]
        :param input_branch: -
        :type input_branch: Union[np.ndarray, torch.Tensor]
        :returns: the result of all the hidden operations in the network
        :rtype: torch.Tensor

        """

        output_residual = self.forward_(
            input_trunk=input_trunk, input_branch=input_branch
        )

        return input_branch + output_residual * input_trunk

    def _forward_cut_residual(
        self,
        input_trunk: Union[np.ndarray, torch.Tensor] = None,
        input_branch: Union[np.ndarray, torch.Tensor] = None,
    ) -> torch.Tensor:
        """

        Wrapper forward method

        :param input_trunk: -
        :type input_trunk: Union[np.ndarray, torch.Tensor]
        :param input_branch: -
        :type input_branch: Union[np.ndarray, torch.Tensor]
        :returns: the result of all the hidden operations in the network
        :rtype: torch.Tensor

        """

        output = self.forward_(input_trunk=input_trunk, input_branch=input_branch)

        return output


class ImprovedDeepONet(ResDeepONet):
    name = "improveddeeponet"
    engine = "torch"

    def __init__(
        self,
        trunk_network: ConvexDenseNetwork = None,
        branch_network: ConvexDenseNetwork = None,
        decoder_network: NetworkTemplate = None,
        encoder_trunk: NetworkTemplate = None,
        encoder_branch: NetworkTemplate = None,
        var_dim: int = 1,
        devices: Union[str, list] = "cpu",
        product_type: str = None,
        rescale_factors: np.ndarray = None,
        residual: bool = False,
        multiply_by_trunk: bool = False,
        model_id=None,
    ) -> None:
        """
        The so-called Improved DeepONet architecture aims at enhancing the communication
        between the trunk and branch pipelines during the training process, thus allowing
                    better generalization capabilities for the composite model.
        """

        # Guaranteeing the compatibility between the encoders and the branch and trunk networks
        t_hs = trunk_network.hidden_size
        et_os = encoder_trunk.output_size
        b_hs = branch_network.hidden_size
        eb_os = encoder_branch.output_size

        assert t_hs == et_os == b_hs == eb_os, (
            "The output of the trunk encoder must have the same dimension"
            " of the trunk network hidden size, but got"
            f" {encoder_trunk.output_size} and {trunk_network.hidden_size}"
        )

        super(ImprovedDeepONet, self).__init__(
            trunk_network=trunk_network,
            branch_network=branch_network,
            decoder_network=decoder_network,
            var_dim=var_dim,
            devices=devices,
            product_type=product_type,
            residual=residual,
            multiply_by_trunk=multiply_by_trunk,
            model_id=model_id,
        )

        # Rescaling factors for the output
        if rescale_factors is not None:
            assert (
                len(rescale_factors) == var_dim
            ), "The number of rescaling factors must be equal to var_dim."
            rescale_factors = torch.from_numpy(rescale_factors.astype("float32"))
        else:
            rescale_factors = torch.from_numpy(np.ones(self.var_dim).astype("float32"))

        self.rescale_factors = rescale_factors.to(self.device)

        self.encoder_trunk = encoder_trunk.to(self.device)
        self.encoder_branch = encoder_branch.to(self.device)

        self.add_module("encoder_trunk", self.encoder_trunk)
        self.add_module("encoder_branch", self.encoder_branch)

        self.forward_ = self._forward_improved

    def _forward_improved(
        self,
        input_trunk: Union[np.ndarray, torch.Tensor] = None,
        input_branch: Union[np.ndarray, torch.Tensor] = None,
    ) -> torch.Tensor:
        # Forward method execution
        v = self.encoder_trunk.forward(input_data=input_trunk)
        u = self.encoder_branch.forward(input_data=input_branch)

        output_trunk = self.trunk_network.forward(input_data=input_trunk, u=u, v=v).to(
            self.device
        )

        output_branch = self.branch_network.forward(
            input_data=input_branch, u=u, v=v
        ).to(self.device)

        output = self._forward(output_trunk=output_trunk, output_branch=output_branch)

        return output * self.rescale_factors

    @guarantee_device
    def eval_subnetwork(
        self,
        name: str = None,
        trunk_data: Union[np.ndarray, torch.Tensor] = None,
        branch_data: Union[np.ndarray, torch.Tensor] = None,
    ) -> np.ndarray:
        assert (
            name in self.subnetworks_names
        ), f"The name {name} is not a subnetwork of {self}."

        network_instance = getattr(self, name + "_network")
        input_data = locals()[name + "_data"]

        v = self.encoder_trunk.forward(input_data=trunk_data)
        u = self.encoder_branch.forward(input_data=branch_data)

        return (
            network_instance.forward(input_data=input_data, u=u, v=v)
            .cpu()
            .detach()
            .numpy()
        )

    def summary(self) -> None:
        print("Trunk Network:")
        self.trunk_network.summary()
        print("Encoder Trunk:")
        self.encoder_trunk.summary()
        print("Encoder Branch:")
        self.encoder_branch.summary()
        print("Branch Network:")
        self.branch_network.summary()


class FlexibleDeepONet(ResDeepONet):
    name = "flexibledeeponet"
    engine = "torch"

    def __init__(
        self,
        trunk_network: NetworkTemplate = None,
        branch_network: NetworkTemplate = None,
        decoder_network: NetworkTemplate = None,
        pre_network: NetworkTemplate = None,
        var_dim: int = 1,
        devices: Union[str, list] = "cpu",
        product_type: str = None,
        residual: bool = False,
        multiply_by_trunk: bool = False,
        model_id=None,
    ) -> None:
        """

        Flexible DeepONet uses a subnetwork called 'pre-network', which
                    plays the role of rescaling the trunk input according to the branch input.
        It is an attempt of reducing the training bias related to the different
                    orders of magnitude contained in the dataset.

        """

        # Guaranteeing the compatibility between the pre and the branch and trunk networks
        t_is = trunk_network.input_size
        p_is = pre_network.input_size
        p_os = pre_network.output_size
        b_is = branch_network.input_size

        assert (2 * t_is == p_os) and (b_is == p_is), (
            "The input of branch and pre networks must have the same dimension"
            " and the output of pre and the input of trunks, too, but got"
            f" {(b_is, p_is)} and {(t_is, p_os)}."
        )

        self.t_is = t_is

        super(FlexibleDeepONet, self).__init__(
            trunk_network=trunk_network,
            branch_network=branch_network,
            decoder_network=decoder_network,
            var_dim=var_dim,
            devices=devices,
            product_type=product_type,
            residual=residual,
            multiply_by_trunk=multiply_by_trunk,
            model_id=model_id,
        )

        self.pre_network = pre_network
        self.forward_ = self._forward_flexible
        self.subnetworks += [self.pre_network]
        self.subnetworks_names += ["pre"]

    def _rescaling_operation(
        self, input_data: torch.Tensor = None, rescaling_tensor: torch.Tensor = None
    ):
        angular = rescaling_tensor[:, : self.t_is]
        linear = rescaling_tensor[:, self.t_is :]

        return angular * input_data + linear

    def _forward_flexible(
        self,
        input_trunk: Union[np.ndarray, torch.Tensor] = None,
        input_branch: Union[np.ndarray, torch.Tensor] = None,
    ) -> torch.Tensor:
        # Forward method execution
        output_branch = self.branch_network.forward(input_data=input_branch).to(
            self.device
        )

        rescaling = self.pre_network.forward(input_data=input_branch).to(self.device)
        input_trunk_rescaled = self._rescaling_operation(
            input_data=input_trunk, rescaling_tensor=rescaling
        )

        output_trunk = self.trunk_network.forward(input_data=input_trunk_rescaled).to(
            self.device
        )

        output = self._forward(output_trunk=output_trunk, output_branch=output_branch)

        return output

    @guarantee_device
    def eval_subnetwork(
        self,
        name: str = None,
        trunk_data: Union[np.ndarray, torch.Tensor] = None,
        branch_data: Union[np.ndarray, torch.Tensor] = None,
    ) -> np.ndarray:
        assert (
            name in self.subnetworks_names
        ), f"The name {name} is not a subnetwork of {self}."

        # Pre and branch network has the same input
        pre_data = branch_data

        network_instance = getattr(self, name + "_network")
        input_data = locals()[name + "_data"]

        return network_instance.forward(input_data=input_data).cpu().detach().numpy()

    def summary(self) -> None:
        print("Trunk Network:")
        self.trunk_network.summary()
        print("Pre Network:")
        self.pre_network.summary()
        print("Branch Network:")
        self.branch_network.summary()
