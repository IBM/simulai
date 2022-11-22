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

import torch
import numpy as np
from typing import Union, List, Optional

from simulai.templates import NetworkTemplate, guarantee_device, as_tensor
from simulai.regression import DenseNetwork, ConvexDenseNetwork
from simulai.regression import ConvolutionalNetwork, Linear, SLFNN

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

# Dense network with hidden encoders aimed at improving the convergence
class ImprovedDenseNetwork(NetworkTemplate):

    name = 'improveddense'
    engine = 'torch'

    def __init__(self, network:ConvexDenseNetwork,
                       encoder_u: NetworkTemplate=None,
                       encoder_v: NetworkTemplate=None):

        """

        Improved DenseNetwork

        It uses auxiliary encoder networks in order to enrich
        the hidden spaces between layers

        :param network: a convex dense networks (it supports convex sums operations in the hidden spaces)
        :type: ConvexDenseNetwork
        :param encoder_u: first auxiliary encoder
        :type encoder_u: NetworkTemplate
        :param: encoder_v: second_auxiliary encoder
        :type encoder_u: NetworkTemplate
        :return: nothing

        """

        super(ImprovedDenseNetwork, self).__init__()

        # Guaranteeing the compatibility between the encoders and the branch and trunk networks
        n_hs = network.hidden_size
        eu_os = encoder_u.output_size
        ev_os = encoder_v.output_size

        assert n_hs == eu_os == ev_os, "The output of the encoders must have the same dimension" \
                                       " of the network hidden size, but got" \
                                       f" {eu_os}, {ev_os} and {n_hs}."

        self.network = network
        self.encoder_u = encoder_u
        self.encoder_v = encoder_v

        self.add_module('network', self.network)
        self.add_module('encoder_u', self.encoder_u)
        self.add_module('encoder_v', self.encoder_v)

        self.weights += self.network.weights
        self.weights += self.encoder_u.weights
        self.weights += self.encoder_v.weights

    def forward(self, input_data: Union[np.ndarray, torch.Tensor] = None) -> torch.Tensor:

        """

        :param input_data: input dataset
        :type input_data: Union[np.ndarray, torch.Tensor]
        :return: operation evaluated over the input data
        :rtype: torch.Tensor

        """

        # Forward method execution
        v = self.encoder_v.forward(input_data=input_data)
        u = self.encoder_u.forward(input_data=input_data)

        output = self.trunk_network.forward(input_data=input_data, u=u, v=v).to(self.device)

        return output

#####################
#### DeepONet family
#####################

class DeepONet(NetworkTemplate):

    name = "deeponet"
    engine = "torch"

    def __init__(self, trunk_network: NetworkTemplate=None,
                       branch_network: NetworkTemplate=None,
                       decoder_network: NetworkTemplate=None, # The decoder network is optional and considered
                       var_dim: int=1,                        # less effective than the output reshaping alternative
                       devices:Union[str, list]='cpu',
                       product_type:str=None,
                       model_id=None) -> None:

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

        self.add_module('trunk_network', self.trunk_network)
        self.add_module('branch_network', self.branch_network)

        if decoder_network is not None:
            self.decoder_network = decoder_network.to(self.device)
            self.add_module('decoder_network', self.decoder_network)
        else:
            self.decoder_network = decoder_network

        self.product_type = product_type
        self.model_id = model_id
        self.var_dim = var_dim

        # Checking up whether the output of each subnetwork are in correct shape
        assert self._latent_dimension_is_correct(self.trunk_network.output_size), "The trunk network must have" \
                                                                                  " one-dimensional output , " \
                                                                                  "but received" \
                                                                                  f"{self.trunk_network.output_size}"

        assert self._latent_dimension_is_correct(self.branch_network.output_size), "The branch network must have" \
                                                                                  " one-dimensional output," \
                                                                                   " but received" \
                                                                                  f"{self.branch_network.output_size}"

        # Checking the compatibility of the subnetworks outputs for each kind of product being employed
        if self.product_type != "dense":

            output_branch = self.branch_network.output_size
            output_trunk = self.trunk_network.output_size

            # It checks if the inner product operation can be performed.
            assert output_branch == output_trunk, f"The output dimensions for the sub-networks" \
                                                  f" trunk and branch must be equal but are" \
                                                  f" {output_branch}" \
                                                  f" and {output_trunk}"
        else:

            output_branch = self.branch_network.output_size

            assert not output_branch % self.var_dim, f"The number of branch latent outputs must" \
                                                     f" be divisible by the number of variables," \
                                                     f" but received {output_branch}" \
                                                     f" and {self.var_dim}"

        self.subnetworks = [net for net in [self.trunk_network,
                                            self.branch_network,
                                            self.decoder_network] if net is not None]

        self.input_trunk = None
        self.input_branch = None

        self.output = None
        self.var_map = dict()

        # Checking up if the input of the decoder network has the correct dimension
        if self.decoder_network is not None:
            assert self.decoder_network.weights[0].shape[1] == 1, "The decoder input is expected" \
                                                                  " to have dimension (None, 1), but" \
                                                                  f"received {self.decoder_network.weights[0].shape}"
        else:
            pass

        # Selecting the correct forward approach to be used
        self._forward = self._forward_selector_()

        self.subnetworks_names = ['trunk', 'branch']

    def _latent_dimension_is_correct(self, dim:Union[int, tuple]) -> bool:

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

    def _forward_decoder(self, output_trunk:torch.Tensor=None, output_branch:torch.Tensor=None) -> torch.Tensor:

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

    def _forward_dense(self, output_trunk:torch.Tensor=None, output_branch:torch.Tensor=None) -> torch.Tensor:

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
        output_branch_reshaped = torch.reshape(output_branch, (-1, self.var_dim, latent_dim))

        output = torch.matmul(output_branch_reshaped, output_trunk[..., None])
        output = torch.squeeze(output)

        return output

    def _forward_pointwise(self, output_trunk:torch.Tensor=None, output_branch:torch.Tensor=None) -> torch.Tensor:

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
        output_trunk_reshaped = torch.reshape(output_trunk, (-1, latent_dim, self.var_dim))
        output_branch_reshaped = torch.reshape(output_branch, (-1, latent_dim, self.var_dim))
        output = torch.sum(output_trunk_reshaped * output_branch_reshaped, dim=-2, keepdim=False)

        return output

    def _forward_vanilla(self, output_trunk:torch.Tensor=None, output_branch:torch.Tensor=None) -> torch.Tensor:

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

        assert len(batches_dimensions) == 1, "This dataset is not proper to apply shuffling"

        dim = list(batches_dimensions)[0]

        indices = np.arange(dim)

        np.random.shuffle(indices)

        var_map_shuffled = {key: value[indices] for key, value in self.var_map.items()}

        return var_map_shuffled

    @property
    def weights(self) -> list:

        return sum([net.weights for net in self.subnetworks], [])

    def forward(self, input_trunk:Union[np.ndarray, torch.Tensor]=None,
                      input_branch:Union[np.ndarray, torch.Tensor]=None) -> torch.Tensor:

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
    def eval(self, trunk_data:Union[np.ndarray, torch.Tensor]=None,
                   branch_data:Union[np.ndarray, torch.Tensor]=None) -> np.ndarray:

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
    def eval_subnetwork(self, name: str = None, input_data: Union[np.ndarray, torch.Tensor] = None) -> np.ndarray:

        assert name in self.subnetworks_names, f"The name {name} is not a subnetwork of {self}."

        network_to_be_used = getattr(self, name + '_network')

        return network_to_be_used.forward(input_data).cpu().detach().numpy()

    def summary(self) -> None:

        print("Trunk Network:")
        self.trunk_network.summary()
        print("Branch Network:")
        self.branch_network.summary()

class ResDeepONet(DeepONet):

    name = "resdeeponet"
    engine = "torch"

    '''The operation performed is: output = input_branch + D(param, input_branch)'''

    def __init__(self, trunk_network: NetworkTemplate = None,
                       branch_network: NetworkTemplate = None,
                       decoder_network: NetworkTemplate = None,  # The decoder network is optional and considered
                       var_dim: int = 1,                         # less effective than the output reshaping alternative
                       devices: Union[str, list] = 'cpu',
                       product_type:str=None,
                       residual: bool = True,
                       multiply_by_trunk: bool = False,
                       model_id=None) -> None:

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

        super(ResDeepONet, self).__init__(trunk_network=trunk_network,
                                          branch_network=branch_network,
                                          decoder_network=decoder_network,  # The decoder network is optional and considered
                                          var_dim=var_dim,                  # less effective than the output reshaping alternative
                                          devices=devices,
                                          product_type=product_type,
                                          model_id=model_id)

        input_dim = self.branch_network.input_size
        assert  input_dim == var_dim, "For a residual network, it is necessary to have" \
                                      "size of branch_network input equal to var_dim, but " \
                                      f"received {input_dim} and {var_dim}."

        self.forward_ = super().forward

        if residual == True:
            self.forward = self._forward_default
        elif multiply_by_trunk == True:
            self.forward = self._forward_multiplied_by_trunk
        else:
            self.forward = self._forward_cut_residual

    def _forward_default(self, input_trunk: Union[np.ndarray, torch.Tensor] = None,
                               input_branch: Union[np.ndarray, torch.Tensor] = None) -> torch.Tensor:

        """

        Wrapper forward method

        :param input_trunk: -
        :type input_trunk: Union[np.ndarray, torch.Tensor]
        :param input_branch: -
        :type input_branch: Union[np.ndarray, torch.Tensor]
        :returns: the result of all the hidden operations in the network
        :rtype: torch.Tensor

        """

        output_residual = self.forward_(input_trunk=input_trunk, input_branch=input_branch)

        return input_branch + output_residual

    def _forward_multiplied_by_trunk(self, input_trunk: Union[np.ndarray, torch.Tensor] = None,
                                           input_branch: Union[np.ndarray, torch.Tensor] = None) -> torch.Tensor:

        """

        Wrapper forward method

        :param input_trunk: -
        :type input_trunk: Union[np.ndarray, torch.Tensor]
        :param input_branch: -
        :type input_branch: Union[np.ndarray, torch.Tensor]
        :returns: the result of all the hidden operations in the network
        :rtype: torch.Tensor

        """

        output_residual = self.forward_(input_trunk=input_trunk, input_branch=input_branch)

        return input_branch + output_residual*input_trunk

    def _forward_cut_residual(self, input_trunk: Union[np.ndarray, torch.Tensor] = None,
                                    input_branch: Union[np.ndarray, torch.Tensor] = None) -> torch.Tensor:

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

    name = "resdeeponet"
    engine = "torch"

    def __init__(self, trunk_network: ConvexDenseNetwork = None,
                       branch_network: ConvexDenseNetwork = None,
                       decoder_network: NetworkTemplate = None,
                       encoder_trunk: NetworkTemplate=None,
                       encoder_branch: NetworkTemplate=None,
                       var_dim: int = 1,
                       devices: Union[str, list] = 'cpu',
                       product_type: str = None,
                       rescale_factors:np.ndarray=None,
                       residual:bool=False,
                       multiply_by_trunk:bool=False,
                       model_id=None) -> None:

        # Guaranteeing the compatibility between the encoders and the branch and trunk networks
        t_hs = trunk_network.hidden_size
        et_os = encoder_trunk.output_size
        b_hs = branch_network.hidden_size
        eb_os = encoder_branch.output_size

        assert t_hs == et_os == b_hs == eb_os , "The output of the trunk encoder must have the same dimension" \
                                                " of the trunk network hidden size, but got"\
                                                f" {encoder_trunk.output_size} and {trunk_network.hidden_size}"

        super(ImprovedDeepONet, self).__init__(trunk_network=trunk_network,
                                               branch_network=branch_network,
                                               decoder_network=decoder_network,
                                               var_dim=var_dim,
                                               devices=devices,
                                               product_type=product_type,
                                               residual=residual,
                                               multiply_by_trunk=multiply_by_trunk,
                                               model_id=model_id)

        # Rescaling factors for the output
        if rescale_factors is not None:
            assert len(rescale_factors) == var_dim, "The number of rescaling factors must be equal to var_dim."
            rescale_factors = torch.from_numpy(rescale_factors.astype('float32'))
        else:
            rescale_factors = torch.from_numpy(np.ones(self.var_dim).astype('float32'))

        self.rescale_factors = rescale_factors.to(self.device)

        self.encoder_trunk = encoder_trunk.to(self.device)
        self.encoder_branch = encoder_branch.to(self.device)

        self.add_module('encoder_trunk', self.encoder_trunk)
        self.add_module('encoder_branch', self.encoder_branch)

        self.forward_ = self._forward_improved

    def _forward_improved(self, input_trunk: Union[np.ndarray, torch.Tensor] = None,
                                input_branch: Union[np.ndarray, torch.Tensor] = None) -> torch.Tensor:

        # Forward method execution
        v = self.encoder_trunk.forward(input_data=input_trunk)
        u = self.encoder_branch.forward(input_data=input_branch)

        output_trunk = self.trunk_network.forward(input_data=input_trunk, u=u, v=v).to(self.device)

        output_branch = self.branch_network.forward(input_data=input_branch, u=u, v=v).to(self.device)

        output = self._forward(output_trunk=output_trunk, output_branch=output_branch)

        return output*self.rescale_factors

    @guarantee_device
    def eval_subnetwork(self, name: str = None, trunk_data:Union[np.ndarray, torch.Tensor]=None,
                              branch_data:Union[np.ndarray, torch.Tensor]=None) -> np.ndarray:

        assert name in self.subnetworks_names, f"The name {name} is not a subnetwork of {self}."

        network_instance = getattr(self, name + '_network')
        input_data = locals()[name + '_data']

        v = self.encoder_trunk.forward(input_data=trunk_data)
        u = self.encoder_branch.forward(input_data=branch_data)

        return network_instance.forward(input_data=input_data, u=u, v=v).cpu().detach().numpy()

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

    name = "resdeeponet"
    engine = "torch"

    def __init__(self, trunk_network: NetworkTemplate = None,
                       branch_network: NetworkTemplate = None,
                       decoder_network: NetworkTemplate = None,
                       pre_network: NetworkTemplate = None,
                       var_dim: int = 1,
                       devices: Union[str, list] = 'cpu',
                       product_type: str = None,
                       residual: bool = False,
                       multiply_by_trunk: bool = False,
                       model_id=None) -> None:

        # Guaranteeing the compatibility between the pre and the branch and trunk networks
        t_is = trunk_network.input_size
        p_is = pre_network.input_size
        p_os = pre_network.output_size
        b_is = branch_network.input_size

        assert (2*t_is == p_os) and (b_is == p_is), "The input of branch and pre networks must have the same dimension" \
                                                        " and the output of pre and the input of trunks, too, but got" \
                                                        f" {(b_is, p_is)} and {(t_is, p_os)}."

        self.t_is = t_is

        super(FlexibleDeepONet, self).__init__(trunk_network=trunk_network,
                                               branch_network=branch_network,
                                               decoder_network=decoder_network,
                                               var_dim=var_dim,
                                               devices=devices,
                                               product_type=product_type,
                                               residual=residual,
                                               multiply_by_trunk=multiply_by_trunk,
                                               model_id=model_id)


        self.pre_network = pre_network
        self.forward_ = self._forward_flexible

    def _rescaling_operation(self, input_data:torch.Tensor=None, rescaling_tensor:torch.Tensor=None):

        angular = rescaling_tensor[:,:self.t_is]
        linear = rescaling_tensor[:, self.t_is:]

        return angular*input_data + linear

    def _forward_flexible(self, input_trunk: Union[np.ndarray, torch.Tensor] = None,
                                input_branch: Union[np.ndarray, torch.Tensor] = None) -> torch.Tensor:

        # Forward method execution
        output_branch = self.branch_network.forward(input_data=input_branch).to(self.device)

        rescaling = self.pre_network.forward(input_data=input_branch).to(self.device)
        input_trunk_rescaled = self._rescaling_operation(input_data=input_trunk, rescaling_tensor=rescaling)

        output_trunk = self.trunk_network.forward(input_data=input_trunk_rescaled).to(self.device)

        output = self._forward(output_trunk=output_trunk, output_branch=output_branch)

        return output

    @guarantee_device
    def eval_subnetwork(self, name: str = None, trunk_data: Union[np.ndarray, torch.Tensor] = None,
                              branch_data: Union[np.ndarray, torch.Tensor] = None) -> np.ndarray:

        assert name in self.subnetworks_names, f"The name {name} is not a subnetwork of {self}."

        network_instance = getattr(self, name + '_network')
        input_data = locals()[name + '_data']

        return network_instance.forward(input_data=input_data).cpu().detach().numpy()

    def summary(self) -> None:
        print("Trunk Network:")
        self.trunk_network.summary()
        print("Pre Network:")
        self.pre_network.summary()
        print("Branch Network:")
        self.branch_network.summary()

####

########################################
### Some usual AutoEncoder architectures
########################################

class AutoencoderMLP(NetworkTemplate):

    """

     This is an implementation of a cFully-connected AutoEncoder as
           Reduced Order Model;

               A MLP autoencoder architecture consists of three stages:
               --> Fully-connected encoder
               --> Fully connected decoder

           SCHEME:
                   |         |
                   |  |   |  |
           Z ->    |  | | |  |  -> Z_til
                   |  |   |  |
                   |         |

              ENCODER       DECODER


    """

    def __init__(self, encoder:DenseNetwork=None,
                       decoder: DenseNetwork=None,
                       devices:Union[str, list]='cpu') -> None:

        super(AutoencoderMLP, self).__init__()

        self.weights = list()

        # Determining the kind of device to be used for allocating the
        # subnetworks used in the DeepONet model
        self.device = self._set_device(devices=devices)

        self.encoder = encoder.to(self.device)
        self.decoder = decoder.to(self.device)

        self.add_module('encoder', self.encoder)
        self.add_module('decoder', self.decoder)

        self.weights += self.encoder.weights
        self.weights += self.decoder.weights

        self.last_encoder_channels = None

    def summary(self) -> None:

        """

        It prints the summary of the network architecture

        """

        self.encoder.summary()
        self.decoder.summary()

    def projection(self, input_data:Union[np.ndarray, torch.Tensor]=None) -> torch.Tensor:

        """

        Projecting the input dataset into the latent space

        :param input_data: the dataset to be projected
        :type input_data: Union[np.ndarray, torch.Tensor]
        :returns: the dataset projected over the latent space
        :rtype: torch.Tensor

        """

        latent = self.encoder.forward(input_data=input_data)

        return latent

    def reconstruction(self, input_data:Union[torch.Tensor, np.ndarray]=None) -> torch.Tensor:

        """

        Reconstructing the latent dataset to the original one

        :param input_data: the dataset to be reconstructed
        :type input_data: Union[np.ndarray, torch.Tensor]
        :returns: the dataset reconstructed
        :rtype: torch.Tensor

        """

        reconstructed = self.decoder.forward(input_data=input_data)

        return reconstructed

    def forward(self, input_data:Union[np.ndarray, torch.Tensor]=None) ->  torch.Tensor:

        """

        Executing the complete projection/reconstruction pipeline

        :param input_data: the input dataset
        :type input_data: Union[np.ndarray, torch.Tensor]
        :returns: the dataset reconstructed
        :rtype: torch.Tensor

        """

        latent = self.projection(input_data=input_data)
        reconstructed = self.reconstruction(input_data=latent)

        return reconstructed

    def eval_projection(self, input_data:Union[np.ndarray, torch.Tensor]=None) -> np.ndarray:

        """

        Projecting the input dataset into the latent space

        :param input_data: the dataset to be projected
        :type input_data: Union[np.ndarray, torch.Tensor]
        :returns: the dataset projected over the latent space
        :rtype: np.ndarray

        """

        return self.projection(input_data=input_data).detach().numpy()

# Convolutional AutoEncoder
class AutoencoderCNN(NetworkTemplate):

    """

    This is an implementation of a convolutional autoencoder as
           Reduced Order Model;

               An autoencoder architecture consists of three stages:
               --> The convolutional encoder
               --> The bottleneck stage, subdivided in:
                   --> Fully-connected encoder
                   --> Fully connected decoder
               --> The convolutional decoder

           SCHEME:
                                                  |         |
                                                  |  |   |  |
           Z -> [Conv] -> [Conv] -> ... [Conv] -> |  | | |  | -> [Conv.T] -> [Conv.T] -> ... [Conv.T] -> Z_til
                                                  |  |   |  |
                                                  |         |

                          ENCODER               DENSE BOTTLENECK           DECODER

    """

    def __init__(self, encoder:ConvolutionalNetwork=None,
                       bottleneck_encoder: Linear=None,
                       bottleneck_decoder: Linear=None,
                       decoder: ConvolutionalNetwork=None,
                       encoder_activation:str='relu',
                       devices:Union[str, list]='cpu') -> None:

        super(AutoencoderCNN, self).__init__()

        self.weights = list()

        # Determining the kind of device to be used for allocating the
        # subnetworks
        self.device = self._set_device(devices=devices)

        self.encoder = encoder.to(self.device)
        self.bottleneck_encoder = bottleneck_encoder.to(self.device)
        self.bottleneck_decoder = bottleneck_decoder.to(self.device)
        self.decoder = decoder.to(self.device)

        self.add_module('encoder', self.encoder)
        self.add_module('bottleneck_encoder', self.bottleneck_encoder)
        self.add_module('bottleneck_decoder', self.bottleneck_decoder)
        self.add_module('decoder', self.decoder)

        self.weights += self.encoder.weights
        self.weights += self.bottleneck_encoder.weights
        self.weights += self.bottleneck_decoder.weights
        self.weights += self.decoder.weights

        self.last_encoder_channels = None
        self.before_flatten_dimension = None

        self.encoder_activation = self._get_operation(operation=encoder_activation)

    def summary(self, input_data: Union[np.ndarray, torch.Tensor]=None, input_shape:list=None) -> torch.Tensor:

        """

        It prints the summary of the network architecture

        :param input_data: the input dataset
        :type input_data: Union[np.ndarray, torch.Tensor]
        :param input_shape: the shape of the input data
        :type input_shape: list
        :returns: the dataset projected over the latent space
        :rtype: torch.Tensor

        """

        self.encoder.summary(input_data=input_data, input_shape=input_shape, device=self.device)

        if isinstance(input_data, np.ndarray):
            btnk_input = self.encoder.forward(input_data=input_data)
        else:
            assert input_shape, "It is necessary to have input_shape when input_data is None."
            input_shape = self.encoder.input_size
            input_shape[0] = 1

            input_data = torch.ones(input_shape).to(self.device)

            btnk_input = self.encoder.forward(input_data=input_data)

        before_flatten_dimension = tuple(btnk_input.shape[1:])
        btnk_input = btnk_input.reshape((-1, np.prod(btnk_input.shape[1:])))

        latent = self.bottleneck_encoder.forward(input_data=btnk_input)

        self.bottleneck_encoder.summary()
        self.bottleneck_decoder.summary()

        bottleneck_output = self.encoder_activation(self.bottleneck_decoder.forward(input_data=latent))

        bottleneck_output = bottleneck_output.reshape((-1, *before_flatten_dimension))

        self.decoder.summary(input_data=bottleneck_output, device=self.device)

    @as_tensor
    def projection(self, input_data:Union[np.ndarray, torch.Tensor]) -> torch.Tensor:

        """

        Projecting the input dataset into the latent space

        :param input_data: the dataset to be projected
        :type input_data: Union[np.ndarray, torch.Tensor]
        :returns: the dataset projected over the latent space
        :rtype: torch.Tensor

        """

        btnk_input = self.encoder.forward(input_data=input_data)

        self.before_flatten_dimension = tuple(btnk_input.shape[1:])

        btnk_input = btnk_input.reshape((-1, np.prod(self.before_flatten_dimension)))

        latent = self.bottleneck_encoder.forward(input_data=btnk_input)

        return latent

    @as_tensor
    def reconstruction(self, input_data:Union[torch.Tensor, np.ndarray]) -> torch.Tensor:

        """

        Reconstructing the latent dataset to the original one

        :param input_data: the dataset to be reconstructed
        :type input_data: Union[np.ndarray, torch.Tensor]
        :returns: the dataset reconstructed
        :rtype: torch.Tensor

        """

        bottleneck_output = self.encoder_activation(self.bottleneck_decoder.forward(input_data=input_data))

        bottleneck_output = bottleneck_output.reshape((-1,) + self.before_flatten_dimension)

        reconstructed = self.decoder.forward(input_data=bottleneck_output)

        return reconstructed

    def forward(self, input_data:Union[np.ndarray, torch.Tensor]) ->  torch.Tensor:

        """

        Executing the complete projection/reconstruction pipeline
        :param input_data: the input dataset
        :type input_data: Union[np.ndarray, torch.Tensor]
        :returns: the dataset reconstructed
        :rtype: torch.Tensor

        """

        latent = self.projection(input_data=input_data)
        reconstructed = self.reconstruction(input_data=latent)

        return reconstructed

    def eval(self, input_data:Union[np.ndarray, torch.Tensor]=None) ->  np.ndarray:

        """

        Projecting the input dataset into the latent space

        :param input_data: the dataset to be projected
        :type input_data: Union[np.ndarray, torch.Tensor]
        :returns: the dataset projected over the latent space
        :rtype: np.ndarray

        """

        if isinstance(input_data, np.ndarray):
            input_data = torch.from_numpy(input_data.astype('float32'))

        input_data = input_data.to(self.device)

        return super().eval(input_data=input_data)

    def project(self, input_data:Union[np.ndarray, torch.Tensor]=None) ->  np.ndarray:

        """

        Projecting the input dataset into the latent space
        :param input_data: the dataset to be projected
        :type input_data: Union[np.ndarray, torch.Tensor]
        :returns: the dataset projected over the latent space
        :rtype: np.ndarray

        """

        projected_data = self.projection(input_data=input_data)

        return projected_data.cpu().detach().numpy()

    def reconstruct(self, input_data: Union[np.ndarray, torch.Tensor] = None) -> np.ndarray:

        """

        Reconstructing the latent dataset to the original one

        :param input_data: the dataset to be reconstructed
        :type input_data: Union[np.ndarray, torch.Tensor]
        :returns: the dataset reconstructed
        :rtype: np.ndarray

        """

        reconstructed_data = self.reconstruction(input_data=input_data)

        return reconstructed_data.cpu().detach().numpy()

class AutoencoderKoopman(NetworkTemplate):

    """

           This is an implementation of a Koopman autoencoder as
           Reduced Order Model;

               A Koopman autoencoder architecture consists of five stages:
               --> The convolutional encoder [Optional]
               --> Fully-connected encoder
               --> Koopman operator
               --> Fully connected decoder
               --> The convolutional decoder [Optional]

           SCHEME:
                                                         K (KOOPMAN OPERATOR)
                                                         ^
                                                  |      |      |
                                                  |  |   |   |  |
           Z -> [Conv] -> [Conv] -> ... [Conv] -> |  | | - | |  | -> [Conv.T] -> [Conv.T] -> ... [Conv.T] -> Z_til
                                                  |  |       |  |
                                                  |             |

                          ENCODER               DENSE BOTTLENECK           DECODER

    """

    def __init__(self, encoder:Union[ConvolutionalNetwork, DenseNetwork]=None,
                       bottleneck_encoder: Optional[Union[Linear, DenseNetwork]]=None,
                       bottleneck_decoder: Optional[Union[Linear, DenseNetwork]]=None,
                       decoder: Union[ConvolutionalNetwork, DenseNetwork]=None,
                       devices:Union[str, list]='cpu') -> None:

        super(AutoencoderKoopman, self).__init__()

        self.weights = list()

        # Determining the kind of device to be used for allocating the
        # subnetworks
        self.device = self._set_device(devices=devices)

        self.encoder = encoder.to(self.device)
        self.decoder = decoder.to(self.device)

        self.add_module('encoder', self.encoder)
        self.add_module('decoder', self.decoder)

        self.weights += self.encoder.weights
        self.weights += self.decoder.weights

        # These subnetworks are optional
        if bottleneck_encoder is not None and bottleneck_decoder is not None:

            self.bottleneck_encoder = bottleneck_encoder.to(self.device)
            self.bottleneck_decoder = bottleneck_decoder.to(self.device)

            self.add_module('bottleneck_encoder', self.bottleneck_encoder)
            self.add_module('bottleneck_decoder', self.bottleneck_decoder)

            self.weights += self.bottleneck_encoder.weights
            self.weights += self.bottleneck_decoder.weights

        if bottleneck_encoder is not None and bottleneck_decoder is not None:

            self.projection = self._projection_with_bottleneck
            self.reconstruction = self._reconstruction_with_bottleneck
        else:
            self.projection = self._projection
            self.reconstruction = self._reconstruction

        self.last_encoder_channels = None
        self.before_flatten_dimension = None

        self.latent_dimension = None

        if bottleneck_encoder is not None:
            self.latent_dimension =  bottleneck_encoder.output_size
        else:
            self.latent_dimension = self.encoder.output_size

        self.K_op = torch.nn.Linear(self.latent_dimension,
                                    self.latent_dimension, bias=False).weight.to(self.device)

    def summary(self, input_data: Union[np.ndarray, torch.Tensor]=None, input_shape:list=None) -> torch.Tensor:

        self.encoder.summary(input_data=input_data, input_shape=input_shape, device=self.device)

        self.bottleneck_encoder.summary()
        self.bottleneck_decoder.summary()

    @as_tensor
    def _projection_with_bottleneck(self, input_data:Union[np.ndarray, torch.Tensor]=None) -> torch.Tensor:

        btnk_input = self.encoder.forward(input_data=input_data)

        self.before_flatten_dimension = tuple(btnk_input.shape[1:])

        btnk_input = btnk_input.reshape((-1, np.prod(self.before_flatten_dimension)))

        latent = self.bottleneck_encoder.forward(input_data=btnk_input)

        return latent

    @as_tensor
    def _projection(self, input_data: Union[np.ndarray, torch.Tensor]=None) -> torch.Tensor:

        latent = self.encoder.forward(input_data=input_data)

        return latent

    @as_tensor
    def _reconstruction_with_bottleneck(self, input_data:Union[torch.Tensor, np.ndarray]=None) -> torch.Tensor:

        bottleneck_output = torch.nn.ReLU()(self.bottleneck_decoder.forward(input_data=input_data))

        bottleneck_output = bottleneck_output.reshape((-1,) + self.before_flatten_dimension)

        reconstructed = self.decoder.forward(input_data=bottleneck_output)

        return reconstructed

    @as_tensor
    def _reconstruction(self, input_data: Union[torch.Tensor, np.ndarray]=None) -> torch.Tensor:

        reconstructed = self.decoder.forward(input_data=input_data)

        return reconstructed

    # Evaluating the operation u^{u+m} = K^m u^{i}
    def latent_forward_m(self, input_data:Union[np.ndarray, torch.Tensor]=None, m:int=1) ->  torch.Tensor:

        return torch.matmul(input_data, torch.pow(self.K_op.T, m))

    # Evaluating the operation u^{u+1} = K u^{i}
    def latent_forward(self, input_data: Union[np.ndarray, torch.Tensor]=None) -> torch.Tensor:

        return torch.matmul(input_data, self.K_op.T)

    # Evaluating the operation Ũ = D(E(U))
    def reconstruction_forward(self, input_data:Union[np.ndarray, torch.Tensor]=None) ->  torch.Tensor:

        latent = self.projection(input_data=input_data)
        reconstructed = self.reconstruction(input_data=latent)

        return reconstructed

    # Evaluating the operation Ũ_m = D(K^m E(U))
    def reconstruction_forward_m(self, input_data:Union[np.ndarray, torch.Tensor]=None, m:int=1) ->  torch.Tensor:

        latent = self.projection(input_data=input_data)
        latent_m = self.latent_forward_m(input_data=latent, m=m)
        reconstructed_m = self.reconstruction(input_data=latent_m)

        return reconstructed_m

    def predict(self, input_data:Union[np.ndarray, torch.Tensor]=None, n_steps:int=1) ->  np.ndarray:

        if isinstance(input_data, np.ndarray):
            input_data = torch.from_numpy(input_data.astype('float32'))

        predictions = list()
        latent = self.projection(input_data=input_data)
        init_latent = latent

        # Extrapolating in the latent space over n_steps steps
        for s in range(n_steps):

            latent_s = self.latent_forward(input_data=init_latent)
            init_latent = latent_s
            predictions.append(latent_s)

        predictions = torch.vstack(predictions)

        reconstructed_predictions = self.reconstruction(input_data=predictions)

        return reconstructed_predictions.detach().numpy()

    def project(self, input_data:Union[np.ndarray, torch.Tensor]=None) ->  np.ndarray:

        projected_data = self.projection(input_data=input_data)

        return projected_data.cpu().detach().numpy()

    def reconstruct(self, input_data: Union[np.ndarray, torch.Tensor] = None) -> np.ndarray:

        reconstructed_data = self.reconstruction(input_data=input_data)

        return reconstructed_data.cpu().detach().numpy()

class AutoencoderVariational(NetworkTemplate):

    """

           This is an implementation of a Koopman autoencoder as
           Reduced Order Model;

               A Koopman autoencoder architecture consists of five stages:
               --> The convolutional encoder [Optional]
               --> Fully-connected encoder
               --> Gaussian noise
               --> Fully connected decoder
               --> The convolutional decoder [Optional]

           SCHEME:
                                                         Gaussian noise
                                                         ^
                                                  |      |      |
                                                  |  |   |   |  |
           Z -> [Conv] -> [Conv] -> ... [Conv] -> |  | | - | |  | -> [Conv.T] -> [Conv.T] -> ... [Conv.T] -> Z_til
                                                  |  |       |  |
                                                  |             |

                          ENCODER               DENSE BOTTLENECK           DECODER

    """

    def __init__(self, encoder:Union[ConvolutionalNetwork, DenseNetwork]=None,
                       bottleneck_encoder: Optional[Union[Linear, DenseNetwork]]=None,
                       bottleneck_decoder: Optional[Union[Linear, DenseNetwork]]=None,
                       decoder: Union[ConvolutionalNetwork, DenseNetwork]=None,
                       encoder_activation: str = 'relu',
                       scale:float=1e-3,
                       devices:Union[str, list]='cpu') -> None:

        super(AutoencoderVariational, self).__init__()

        self.weights = list()

        # Determining the kind of device to be used for allocating the
        # subnetworks
        self.device = self._set_device(devices=devices)

        self.encoder = encoder.to(self.device)
        self.decoder = decoder.to(self.device)

        self.add_module('encoder', self.encoder)
        self.add_module('decoder', self.decoder)

        self.weights += self.encoder.weights
        self.weights += self.decoder.weights

        # These subnetworks are optional
        if bottleneck_encoder is not None and bottleneck_decoder is not None:

            self.bottleneck_encoder = bottleneck_encoder.to(self.device)
            self.bottleneck_decoder = bottleneck_decoder.to(self.device)

            self.add_module('bottleneck_encoder', self.bottleneck_encoder)
            self.add_module('bottleneck_decoder', self.bottleneck_decoder)

            self.weights += self.bottleneck_encoder.weights
            self.weights += self.bottleneck_decoder.weights

        if bottleneck_encoder is not None and bottleneck_decoder is not None:

            self.projection = self._projection_with_bottleneck
            self.reconstruction = self._reconstruction_with_bottleneck
        else:
            self.projection = self._projection
            self.reconstruction = self._reconstruction

        self.last_encoder_channels = None
        self.before_flatten_dimension = None

        self.latent_dimension = None

        if bottleneck_encoder is not None:
            self.latent_dimension =  bottleneck_encoder.output_size
        else:
            self.latent_dimension = self.encoder.output_size

        self.z_mean = torch.nn.Linear(self.latent_dimension, self.latent_dimension).to(self.device)
        self.z_log_var = torch.nn.Linear(self.latent_dimension, self.latent_dimension).to(self.device)

        self.add_module('z_mean', self.z_mean)
        self.add_module('z_log_var', self.z_log_var)

        self.weights += [self.z_mean.weight]
        self.weights += [self.z_log_var.weight]

        self.mu = None
        self.log_v = None
        self.scale = scale

        self.encoder_activation = self._get_operation(operation=encoder_activation)

    def summary(self, input_data: Union[np.ndarray, torch.Tensor]=None, input_shape:list=None) -> torch.Tensor:

        self.encoder.summary(input_data=input_data, input_shape=input_shape, device=self.device)

        self.before_flatten_dimension = tuple(self.encoder.output_size[1:])

        self.bottleneck_encoder.summary()
        self.bottleneck_decoder.summary()

    @as_tensor
    def _projection_with_bottleneck(self, input_data:Union[np.ndarray, torch.Tensor]=None) -> torch.Tensor:

        btnk_input = self.encoder.forward(input_data=input_data)

        self.before_flatten_dimension = tuple(self.encoder.output_size[1:])

        btnk_input = btnk_input.reshape((-1, np.prod(self.before_flatten_dimension)))

        latent = self.bottleneck_encoder.forward(input_data=btnk_input)

        return latent

    @as_tensor
    def _projection(self, input_data: Union[np.ndarray, torch.Tensor]=None) -> torch.Tensor:

        latent = self.encoder.forward(input_data=input_data)

        return latent

    @as_tensor
    def _reconstruction_with_bottleneck(self, input_data:Union[torch.Tensor, np.ndarray]=None) -> torch.Tensor:

        bottleneck_output = self.encoder_activation((self.bottleneck_decoder.forward(input_data=input_data)))

        bottleneck_output = bottleneck_output.reshape((-1,) + self.before_flatten_dimension)

        reconstructed = self.decoder.forward(input_data=bottleneck_output)

        return reconstructed

    @as_tensor
    def _reconstruction(self, input_data: Union[torch.Tensor, np.ndarray]=None) -> torch.Tensor:

        reconstructed = self.decoder.forward(input_data=input_data)

        return reconstructed

    def Mu(self, input_data: Union[np.ndarray, torch.Tensor]=None,
                 to_numpy:bool=False) -> Union[np.ndarray, torch.Tensor]:

        latent = self.projection(input_data=input_data)

        if to_numpy == True:
            return self.z_mean(latent).detach().numpy()
        else:
            return self.z_mean(latent)

    def Sigma(self, input_data: Union[np.ndarray, torch.Tensor] = None,
                    to_numpy:bool=False) -> Union[np.ndarray, torch.Tensor]:

        latent = self.projection(input_data=input_data)

        if to_numpy == True:
            return torch.exp(self.z_log_var(latent)/2).detach().numpy()
        else:
            return torch.exp(self.z_log_var(latent)/2)

    def CoVariance(self, input_data: Union[np.ndarray, torch.Tensor] = None, inv:bool=False,
                         to_numpy:bool=False) -> Union[np.ndarray, torch.Tensor]:

        if inv == False:

            Sigma_inv = 1/self.Sigma(input_data=input_data)
            covariance = torch.diag_embed(Sigma_inv)

        else:

            Sigma = self.Sigma(input_data=input_data)
            covariance = torch.diag_embed(Sigma)

        if to_numpy == True:
            return covariance.detach().numpy()
        else:
            return covariance

    def latent_gaussian_noisy(self, input_data: Union[np.ndarray, torch.Tensor]=None) -> torch.Tensor:

        self.mu = self.z_mean(input_data)
        self.log_v = self.z_log_var(input_data)
        eps = self.scale * torch.autograd.Variable(torch.randn(*self.log_v.size())).type_as(self.log_v)

        return self.mu + torch.exp(self.log_v / 2.) * eps

    # Evaluating the operation Ũ = D(E(U))
    def reconstruction_forward(self, input_data:Union[np.ndarray, torch.Tensor]=None) ->  torch.Tensor:

        latent = self.projection(input_data=input_data)
        latent_noisy = self.latent_gaussian_noisy(input_data=latent)
        reconstructed = self.reconstruction(input_data=latent_noisy)

        return reconstructed

    # Evaluating the operation Ũ = D(E(U))
    def reconstruction_eval(self, input_data: Union[np.ndarray, torch.Tensor] = None) -> torch.Tensor:

        encoder_output = self.projection(input_data=input_data)
        latent = self.z_mean(encoder_output)
        reconstructed = self.reconstruction(input_data=latent)

        return reconstructed

    def project(self, input_data:Union[np.ndarray, torch.Tensor]=None) ->  np.ndarray:

        if isinstance(input_data, np.ndarray):
            input_data = torch.from_numpy(input_data.astype('float32'))

        projected_data_latent = self.Mu(input_data=input_data)

        return projected_data_latent

    def reconstruct(self, input_data: Union[np.ndarray, torch.Tensor] = None) -> np.ndarray:

        if isinstance(input_data, np.ndarray):
            input_data = torch.from_numpy(input_data.astype('float32'))

        reconstructed_data = self.reconstruction(input_data=input_data)

        return reconstructed_data.cpu().detach().numpy()

    def eval(self, input_data: Union[np.ndarray, torch.Tensor] = None) -> np.ndarray:

        if isinstance(input_data, np.ndarray):
            input_data = torch.from_numpy(input_data.astype('float32'))

        input_data = input_data.to(self.device)

        return self.reconstruction_eval(input_data=input_data).cpu().detach().numpy()

# Mixture of Experts POC
class MoEPool(NetworkTemplate):

    def __init__(self, experts_list:List[NetworkTemplate], gating_network:NetworkTemplate=None,
                       input_size:int=None, devices:Union[list, str]=None,
                       binary_selection:bool=False) -> None:

        super(MoEPool, self).__init__()

        # Determining the kind of device to be used for allocating the
        # subnetworks used in the DeepONet model
        self.device = self._set_device(devices=devices)


        self.experts_list = experts_list
        self.n_experts = len(experts_list)
        self.input_size = input_size
        self.binary_selection = binary_selection

        # Gating (classifier) network

        # The default gating network is a single-layer fully-connected network
        if gating_network is None:
            self.gating_network = SLFNN(input_size=self.input_size,
                                        output_size=self.n_experts,
                                        activation='softmax').to(self.device)

        else:
            self.gating_network = gating_network.to(self.device)

        # Sending each sub-network to the correct device
        for ei, expert in enumerate(self.experts_list):
            self.experts_list[ei] = expert.to(self.device)

        self.add_module('gating', self.gating_network)

        for ii, item in enumerate(self.experts_list):
            self.add_module(f'expert_{ii}', item)

        self.weights = sum([i.weights for i in self.experts_list], [])
        self.weights += self.gating_network.weights

        self.output_size = self.experts_list[-1].output_size

        if self.binary_selection is True:
            self.get_weights = self._get_weights_binary
        else:
            self.get_weights = self._get_weights_bypass

    def _get_weights_bypass(self, gating:torch.Tensor=None) -> torch.Tensor:

        return  gating

    def _get_weights_binary(self, gating:torch.Tensor=None) -> torch.Tensor:

        maxs = torch.max(gating, dim=1).values[:, None]

        return torch.where(gating == maxs, 1, 0).to(self.device)

    def gate(self, input_data: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:

        gating = self.gating_network.forward(input_data=input_data)
        gating_weights_ = self.get_weights(gating=gating)

        return gating_weights_

    def forward(self, input_data: Union[np.ndarray, torch.Tensor], **kwargs) -> torch.Tensor:

        gating_weights_ = self.gate(input_data=input_data)

        gating_weights = torch.split(gating_weights_, 1, dim=1)

        def _forward(worker: NetworkTemplate = None) -> torch.Tensor:
            return worker.forward(input_data=input_data, **kwargs)

        output = list(map(_forward, self.experts_list))

        return sum([g*o for g,o in zip(gating_weights, output)])
