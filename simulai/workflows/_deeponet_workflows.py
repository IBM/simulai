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

from typing import Optional, Tuple, List, Union

from simulai.models import DeepONet
from simulai.regression import ConvolutionalNetwork, DenseNetwork
from simulai.templates import NetworkInstanceGen

class ConvDeepONet:

    def __init__(
        self,
        trunk_config:dict=None,
        n_outputs:int=None,
        dim:int=None,
        n_latent:int=None,
        branch_input_dim:Union[Tuple, List]=None,
        shallow: bool = True,
        use_batch_norm: bool = True,
        branch_activation: str = "tanh",
        model_id:str="cnn_deeponet",
        product_type:str=None,
        cnn_flatten:bool=True,
    ) -> None:

        self.trunk_config = trunk_config
        self.n_outputs = n_outputs
        self.dim = dim
        self.n_latent = n_latent
        self.use_batch_norm = use_batch_norm
        self.shallow = shallow
        self.branch_activation = branch_activation
        self.branch_input_dim = branch_input_dim
        self.product_type = product_type
        self.cnn_flatten = cnn_flatten
        self.model_id = model_id

    def __call__(self):

        n_inputs = self.trunk_config["input_size"]
        n_latent = self.trunk_config["output_size"]

        netgen = NetworkInstanceGen(architecture="cnn",
                                    dim=self.dim,
                                    shallow=self.shallow, 
                                    use_batch_norm=self.use_batch_norm)

        branch_net = netgen(input_dim=self.branch_input_dim,
                            output_dim=self.n_latent*self.n_outputs,
                            activation=self.branch_activation, flatten=self.cnn_flatten)

        # Instantiating and training the surrogate model
        trunk_net = DenseNetwork(**self.trunk_config)
        # It prints a summary of the network features
        trunk_net.summary()
        branch_net.summary(input_shape=list(self.branch_input_dim))

        net = DeepONet(
            trunk_network=trunk_net,
            branch_network=branch_net,
            var_dim=self.n_outputs,
            product_type=self.product_type,
            model_id=self.model_id,
        )

        return net

