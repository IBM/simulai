import copy
import numpy as np
import torch
from typing import Union, Tuple

from simulai.templates import NetworkTemplate, ConvNetworkTemplate, as_tensor
from simulai.regression import DenseNetwork, SLFNN

# A CNN UNet encoder or decodeder is no more than a curved CNN
# in which intermediary outputs and inputs are also stored.
class CNNUnetEncoder(ConvNetworkTemplate):
    name = "convunetencoder"
    engine = "torch"

    def __init__(
        self,
        layers: list = None,
        activations: list = None,
        pre_layer: Optional[torch.nn.Module] = None,
        case: str = "2d",
        last_activation: str = "identity",
        transpose: bool = False,
        flatten: bool = False,
        intermediary_outputs_indices: List[int] = None,
        name: str = None,
    ) -> None:

        super(CNNUnetEncoder, self).__init__(layers=layers,
                                            activations=activations,
                                            pre_layer=pre_layer,
                                            case=case,
                                            last_activation=last_activation,
                                            transpose=transpose,
                                            flatten=flatten,
                                            name=name,
                                    )

        self.intermediary_outputs_indices = intermediary_outputs_indices

    @as_tensor
    @channels_dim
    def forward(
        self, input_data: Union[torch.Tensor, np.ndarray] = None
    ) -> torch.Tensor:

        intermediary_outputs = list()

        for j in self.intermediary_outputs_indices:
            intermediary_outputs[j] = self.pipeline[:j](input_data)

        main_output = self.pipeline(input_data)

        return main_output, intermediary_outputs
    
class CNNUnetDecoder(ConvNetworkTemplate):
    name = "convunetencoder"
    engine = "torch"

    def __init__(
        self,
        layers: list = None,
        activations: list = None,
        pre_layer: Optional[torch.nn.Module] = None,
        case: str = "2d",
        last_activation: str = "identity",
        transpose: bool = False,
        flatten: bool = False,
        intermediary_outputs_indices: List[int] = None,
        name: str = None,
    ) -> None:

        super(CNNUnetEncoder, self).__init__(layers=layers,
                                            activations=activations,
                                            pre_layer=pre_layer,
                                            case=case,
                                            last_activation=last_activation,
                                            transpose=transpose,
                                            flatten=flatten,
                                            name=name,
                                    )

        self.intermediary_inputs_indices = intermediary_inputs_indices

    @as_tensor
    @channels_dim
    def forward(
        self, input_data: Union[torch.Tensor, np.ndarray] = None,
        intermediary_encoder_outputs:List[torch.Tensor] = None,
    ) -> torch.Tensor:

        current_input = input_data

        for j, layer_j in enumerate(self.list_of_layers):

            if j in self.intermediary_inputs_indices:
                input_j = torch.stack([current_input, self.intermediary_encoder_outputs[j]], axis=-1)
            else:
                input_j = current_input

            output_j = layer_j(input_j)
            current_input = output_j

        return current_input

class UNet(NetworkTemplate):

    def __init__(self, n_levels:int=None, layers_config:dict=None) -> None:

        self.n_levels = n_levels
        self.layers_config_encoder = self.layers_config["encoder"] 
        self.layers_config_decoder = self.layers_config["decoder"] 
        self.encoder_horizontal_outputs = dict()
       
        # Configuring the encoder
        encoder_type = self.layers_config_encoder.get("type")
        layers_config_encoder = self.layers_config_encoder.get("architecture")
        
        if encoder_type == "cnn":
            self.encoder = CNNUnetEncoder(**layers_config_encoder)
        else:
            raise Exception(f"Option {encoder_type} is not available.")

         # Configuring the decoder
        decoder_type = self.layers_config_decoder.get("type")
        layers_config_encoder = self.layers_config_encoder.get("architecture")
        
        if encoder_type == "cnn":
            self.decoder = CNNUnetDecoder(**layers_config_decoder)
        else:
            raise Exception(f"Option {encoder_type} is not available.")
        
    @as_tensor
    def forward(self, input_data: Union[torch.Tensor, np.ndarray] = None
    ) -> torch.Tensor:

        encoder_main_output, encoder_intermediary_outputs = self.encoder(input_data=input_data)
        output = self.decoder(input_path = encoder_main_output,
                              intermediary_encoder_outputs=encoder_intermediary_outputs)

        return output
        

