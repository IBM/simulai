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

from simulai import ARRAY_DTYPE
from simulai.regression import ConvexDenseNetwork, Linear
from simulai.templates import NetworkTemplate, guarantee_device

class WorkflowModule(torch.nn.Module):

    def __init__(self, network: NetworkTemplate=None) -> None:

        super(WorkflowModule, self).__init__()

        self.network = network

    def forward(self, parameters: torch.Tensor=None, 
                      input_tensor: torch.Tensor=None):
         self.network.set_parameters(parameters=parameters, requires_grad=False)

         return self.network(input_tensor)


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
        """Neural Implicit Flow.

        Args:
            shape_network (NetworkTemplate, optional): Subnetwork for processing the coordinates inputs. (Default value = None)
            parameter_network (NetworkTemplate, optional): Subnetwork for processing the forcing/conditioning inputs. (Default value = None)
            decoder_network (NetworkTemplate, optional): Subnetworks for converting the embedding to the output (optional). (Default value = None)
            devices (Union[str, list], optional):  Devices in which the model will be executed. (Default value = "cpu")
            rescale_factors (np.ndarray, optional): Values used for rescaling the network outputs for a given order of magnitude. (Default value = None)
            model_id (str, optional): Name for the model (Default value = None)

        """

        super(NIF, self).__init__(devices=devices)

        # Determining the kind of device to be used for allocating the
        # subnetworks used in the DeepONet model
        self.device = self._set_device(devices=devices)

        self.shape_network = self.to_wrap(entity=shape_network, device=self.device)
        self.parameter_network = self.to_wrap(entity=parameter_network, device=self.device)

        # The number of coefficients to be estimated 
        # by the parameter network
        self.n_shape_parameters = self.shape_network.n_parameters
        self.n_inputs_shape = self.shape_network.input_size 
        self.n_outputs_shape = self.shape_network.output_size
        self.n_inputs_parameter = self.parameter_network.input_size
        self.n_outputs_parameter = self.parameter_network.output_size

        # Latent projection. It is, as default choice, a linear
        # operation
        self.latent_projection = Linear(input_size=self.n_outputs_parameter,
                                        output_size=self.n_shape_parameters)

        # The shape network is not trainable, the coefficients are
        # estimated from the parameter network
        self.shape_network.detach_parameters()

        self.add_module("shape_network", self.shape_network)
        self.add_module("parameter_network", self.parameter_network)
        self.add_module("latent_projection", self.latent_projection)

        if decoder_network is not None:
            self.decoder_network = self.to_wrap(
                entity=decoder_network, device=self.device
            )
            self.add_module("decoder_network", self.decoder_network)
        else:
            self.decoder_network = decoder_network

        self.model_id = model_id

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

        self.subnetworks = [
            net
            for net in [self.shape_network, self.parameter_network, self.decoder_network]
            if net is not None
        ]

        self.input_shape = None
        self.input_parameter = None

        self.output = None

        # TODO Checking up if the input of the decoder network has the correct dimension
        if self.decoder_network is not None:
            print("Decoder is being used.")
        else:
            pass

        # Selecting the correct forward approach to be used

        self.subnetworks_names = ["shape", "parameter"]

        # Tracing the shape net workflow using TorchScript
        sample_parameters_tensor = torch.from_numpy(np.random.rand(self.n_shape_parameters).astype(ARRAY_DTYPE))
        sample_input_tensor = torch.from_numpy(np.random.rand(1_00, self.n_inputs_shape).astype(ARRAY_DTYPE))

        workflow_instance = WorkflowModule(network=self.shape_network)

        self.traced_shape_workflow = workflow_instance.forward
        self.vmapped_forward = torch.vmap(self._forward, in_dims=0)

    def _forward(
        self, input_shape: torch.Tensor = None, output_parameter: torch.Tensor = None
    ) -> torch.Tensor:
        """ 

        Args:
            input_shape (torch.Tensor, optional): The embedding generated by the trunk network. (Default value = None)
            output_parameter (torch.Tensor, optional): The embedding generated by the branch network. (Default value = None)

        Returns:
            torch.Tensor: The output estimated by the shape network.

        """
        # The latent space outputted by the parameter network is projected onto another
        # high-dimensional space (with dimensionality equivalent to the parameters space)
        estimated_parameters = self.latent_projection(output_parameter)
    
        workflow_instance = WorkflowModule(network=self.shape_network)

        # The shape network is set up using those estimated coefficients
        output = workflow_instance(estimated_parameters, input_shape)

        return output

    @property
    def weights(self) -> list:
        return sum([net.weights for net in self.subnetworks], [])

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
        input_shape: Union[np.ndarray, torch.Tensor] = None,
        input_parameter: Union[np.ndarray, torch.Tensor] = None,
    ) -> torch.Tensor:
        """Wrapper forward method.

        Args:
            input_shape (Union[np.ndarray, torch.Tensor], optional):  (Default value = None)
            input_parameter (Union[np.ndarray, torch.Tensor], optional):  (Default value = None)

        Returns:
            torch.Tensor: The result of all the hidden operations in the network.

        """

        # Forward method execution
        output_parameter = self.to_wrap(
            entity=self.parameter_network.forward(input_parameter), device=self.device
        )

        output = self.vmapped_forward(input_shape, output_parameter)

        # Wrappers are applied to execute user-defined operations.
        # When those operations are not selected, these wrappers simply
        # bypass the inputs.

        return self.rescale_wrapper(input_data=self.decoder_wrapper(input_data=output))

    @guarantee_device
    def eval(
        self,
        shape_data: Union[np.ndarray, torch.Tensor] = None,
        parameter_data: Union[np.ndarray, torch.Tensor] = None,
    ) -> np.ndarray:
        """It uses the network to make evaluations.

        Args:
            shape_data (Union[np.ndarray, torch.Tensor], optional):  (Default value = None)
            parameter_data (Union[np.ndarray, torch.Tensor], optional):  (Default value = None)

        Returns:
            np.ndarray: The result of all the hidden operations in the network.

        """

        output_tensor = self.forward(input_shape=shape_data, input_parameter=parameter_data)

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
        print(self)

