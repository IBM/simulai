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


class NIF(NetworkTemplate):
    name = "nif"
    engine = "torch"

    def __init__(
        self,
        shape_network: NetworkTemplate = None,
        parameter_network: NetworkTemplate = None,
        decoder_network: NetworkTemplate = None,  # The decoder network is optional and considered
        var_dim: int = 1,  # less effective than the output reshaping alternative
        devices: Union[str, list] = "cpu",
        rescale_factors: np.ndarray = None,
        model_id: str = None,
    ) -> None:
        """Classical Deep Operator Network (DeepONet), a deep learning version
        of the Universal Approximation Theorem.

        Args:
            trunk_network (NetworkTemplate, optional): Subnetwork for processing the coordinates inputs. (Default value = None)
            branch_network (NetworkTemplate, optional): Subnetwork for processing the forcing/conditioning inputs. (Default value = None)
            decoder_network (NetworkTemplate, optional): Subnetworks for converting the embedding to the output (optional). (Default value = None)
            devices (Union[str, list], optional):  Devices in which the model will be executed. (Default value = "cpu")
            rescale_factors (np.ndarray, optional): Values used for rescaling the network outputs for a given order of magnitude. (Default value = None)
            model_id (str, optional): Name for the model (Default value = None)

        """

        super(DeepONet, self).__init__(devices=devices)

        # Determining the kind of device to be used for allocating the
        # subnetworks used in the DeepONet model
        self.device = self._set_device(devices=devices)
        self.use_bias = use_bias

        self.trunk_network = self.to_wrap(entity=trunk_network, device=self.device)
        self.branch_network = self.to_wrap(entity=branch_network, device=self.device)

        self.add_module("trunk_network", self.trunk_network)
        self.add_module("branch_network", self.branch_network)

        if decoder_network is not None:
            self.decoder_network = self.to_wrap(
                entity=decoder_network, device=self.device
            )
            self.add_module("decoder_network", self.decoder_network)
        else:
            self.decoder_network = decoder_network

        self.product_type = product_type
        self.model_id = model_id
        self.var_dim = var_dim

        # Rescaling factors for the output
        if rescale_factors is not None:
            assert (
                len(rescale_factors) == var_dim
            ), "The number of rescaling factors must be equal to var_dim."
            rescale_factors = torch.from_numpy(rescale_factors.astype("float32"))
            self.rescale_factors = self.to_wrap(
                entity=rescale_factors, device=self.device
            )
        else:
            self.rescale_factors = None

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

        # If bias is being used, check whether the network outputs are compatible.
        if self.use_bias:
            print("Bias is being used.")
            self._bias_compatibility_is_correct(
                dim_trunk=self.trunk_network.output_size,
                dim_branch=self.branch_network.output_size,
            )
            self.bias_wrapper = self._wrapper_bias_active
        else:
            self.bias_wrapper = self._wrapper_bias_inactive

        # Using a decoder on top of the model or not
        if self.decoder_network is not None:
            self.decoder_wrapper = self._wrapper_decoder_active
        else:
            self.decoder_wrapper = self._wrapper_decoder_inactive

        # Using rescaling factors or not
        if rescale_factors is not None:
            self.rescale_wrapper = self._wrapper_rescale_active
        else:
            self.rescale_wrapper = self._wrapper_rescale_inactive

        # Checking the compatibility of the subnetworks outputs for each kind of product being employed.
        if self.product_type != "dense":
            output_branch = self.branch_network.output_size
            output_trunk = self.trunk_network.output_size

            # It checks if the inner product operation can be performed.
            if not self.use_bias:
                assert output_branch == output_trunk, (
                    f"The output dimensions for the sub-networks"
                    f" trunk and branch must be equal but are"
                    f" {output_branch}"
                    f" and {output_trunk}"
                )
            else:
                print("Bias compatibility was already verified.")
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

        # TODO Checking up if the input of the decoder network has the correct dimension
        if self.decoder_network is not None:
            print("Decoder is being used.")
        else:
            pass

        # Selecting the correct forward approach to be used
        self._forward = self._forward_selector_()

        self.subnetworks_names = ["trunk", "branch"]

    def _latent_dimension_is_correct(self, dim: Union[int, tuple]) -> bool:
        """It checks if the latent dimension is consistent.

        Args:
            dim (Union[int, tuple]): Latent_space_dimension.

        Returns:
            bool: The confirmation about the dimensionality correctness.

        """

        if type(dim) == int:
            return True
        elif type(dim) == tuple:
            if len(tuple) == 1:
                return True
            else:
                return False

    def _bias_compatibility_is_correct(
        self, dim_trunk: Union[int, tuple], dim_branch: Union[int, tuple]
    ) -> bool:
        assert dim_branch == dim_trunk + self.var_dim, (
            "When using bias, the dimension"
            + "of the branch output should be"
            + "trunk output + var_dim."
        )

    def _forward_dense(
        self, output_trunk: torch.Tensor = None, output_branch: torch.Tensor = None
    ) -> torch.Tensor:
        """Forward method used when the embeddings are multiplied using a matrix-like product, it means, the trunk
        network outputs serve as "interpolation basis" for the branch outputs.

        Args:
            output_trunk (torch.Tensor, optional): The embedding generated by the trunk network. (Default value = None)
            output_branch (torch.Tensor, optional): The embedding generated by the branch network. (Default value = None)

        Returns:
            torch.Tensor: The product between the two embeddings.

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
        """Forward method used when the embeddings are multiplied using a simple point-wise product, after that a
        reshaping is applied in order to produce multiple outputs.

        Args:
            output_trunk (torch.Tensor, optional): The embedding generated by the trunk network. (Default value = None)
            output_branch (torch.Tensor, optional): The embedding generated by the branch network. (Default value = None)

        Returns:
            torch.Tensor: The product between the two embeddings.

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
        """Forward method used when the embeddings are multiplied using a simple point-wise product.

        Args:
            output_trunk (torch.Tensor, optional): The embedding generated by the trunk network. (Default value = None)
            output_branch (torch.Tensor, optional): The embedding generated by the branch network. (Default value = None)

        Returns:
            torch.Tensor: The product between the two embeddings.

        """

        output = torch.sum(output_trunk * output_branch, dim=-1, keepdim=True)

        return output

    def _forward_selector_(self) -> callable:
        """It selects the forward method to be used.


        Returns:
            callable : The callable corresponding to the required forward method.

        """

        if self.var_dim > 1:
            # It operates as a typical dense layer
            if self.product_type == "dense":
                return self._forward_dense
            # It executes an inner product by parts between the outputs
            # of the subnetworks branch and trunk
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

    # Now, a sequence of wrappers
    def _wrapper_bias_inactive(
        self,
        output_trunk: Union[np.ndarray, torch.Tensor] = None,
        output_branch: Union[np.ndarray, torch.Tensor] = None,
    ) -> torch.Tensor:
        output = self._forward(output_trunk=output_trunk, output_branch=output_branch)

        return output

    def _wrapper_bias_active(
        self,
        output_trunk: Union[np.ndarray, torch.Tensor] = None,
        output_branch: Union[np.ndarray, torch.Tensor] = None,
    ) -> torch.Tensor:
        output_branch_ = output_branch[:, : -self.var_dim]
        bias = output_branch[:, -self.var_dim :]

        output = (
            self._forward(output_trunk=output_trunk, output_branch=output_branch_)
            + bias
        )

        return output

    def _wrapper_decoder_active(
        self,
        input_data: Union[np.ndarray, torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.decoder_network.forward(input_data=input_data)

    def _wrapper_decoder_inactive(
        self,
        input_data: Union[np.ndarray, torch.Tensor] = None,
    ) -> torch.Tensor:
        return input_data

    def _wrapper_rescale_active(
        self,
        input_data: Union[np.ndarray, torch.Tensor] = None,
    ) -> torch.Tensor:
        return input_data * self.rescale_factors

    def _wrapper_rescale_inactive(
        self,
        input_data: Union[np.ndarray, torch.Tensor] = None,
    ) -> torch.Tensor:
        return input_data

    def forward(
        self,
        input_trunk: Union[np.ndarray, torch.Tensor] = None,
        input_branch: Union[np.ndarray, torch.Tensor] = None,
    ) -> torch.Tensor:
        """Wrapper forward method.

        Args:
            input_trunk (Union[np.ndarray, torch.Tensor], optional):  (Default value = None)
            input_branch (Union[np.ndarray, torch.Tensor], optional):  (Default value = None)

        Returns:
            torch.Tensor: The result of all the hidden operations in the network.

        """

        # Forward method execution
        output_trunk = self.to_wrap(
            entity=self.trunk_network.forward(input_trunk), device=self.device
        )

        output_branch = self.to_wrap(
            entity=self.branch_network.forward(input_branch), device=self.device
        )

        # Wrappers are applied to execute user-defined operations.
        # When those operations are not selected, these wrappers simply
        # bypass the inputs.
        output = self.bias_wrapper(
            output_trunk=output_trunk, output_branch=output_branch
        )

        return self.rescale_wrapper(input_data=self.decoder_wrapper(input_data=output))

    @guarantee_device
    def eval(
        self,
        trunk_data: Union[np.ndarray, torch.Tensor] = None,
        branch_data: Union[np.ndarray, torch.Tensor] = None,
    ) -> np.ndarray:
        """It uses the network to make evaluations.

        Args:
            trunk_data (Union[np.ndarray, torch.Tensor], optional):  (Default value = None)
            branch_data (Union[np.ndarray, torch.Tensor], optional):  (Default value = None)

        Returns:
            np.ndarray: The result of all the hidden operations in the network.

        """

        output_tensor = self.forward(input_trunk=trunk_data, input_branch=branch_data)

        return output_tensor.cpu().detach().numpy()

    @guarantee_device
    def eval_subnetwork(
        self, name: str = None, input_data: Union[np.ndarray, torch.Tensor] = None
    ) -> np.ndarray:
        """It evaluates the output of DeepONet subnetworks.

        Args:
            name (str, optional): Name of the subnetwork. (Default value = None)
            input_data (Union[np.ndarray, torch.Tensor], optional): The data used as input for the subnetwork. (Default value = None)

        Returns:
            np.ndarray: The evaluation performed by the subnetwork.

        """

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



