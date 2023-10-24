import copy
import numpy as np
import torch
from typing import Union, List, Tuple, Optional

from simulai.templates import NetworkTemplate, as_tensor, channels_dim
from simulai.regression import DenseNetwork, SLFNN, ConvolutionalNetwork

# A CNN UNet encoder or decodeder is no more than a curved CNN
# in which intermediary outputs and inputs are also stored.
class CNNUnetEncoder(ConvolutionalNetwork):
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
        
        self.pipeline = torch.nn.Sequential(*[layer_j for layer_j in self.list_of_layers 
                                              if not isinstance(layer_j, torch.nn.Identity)])
    @as_tensor
    @channels_dim
    def forward(
        self, input_data: Union[torch.Tensor, np.ndarray] = None
    ) -> torch.Tensor:

        intermediary_outputs = list()

        for j in self.intermediary_outputs_indices:
            intermediary_outputs.append(self.pipeline[:j](input_data))

        main_output = self.pipeline(input_data)
        
        return main_output, intermediary_outputs
    
class CNNUnetDecoder(ConvolutionalNetwork):
    name = "convunetdecoder"
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
        intermediary_inputs_indices: List[int] = None,
        name: str = None,
        channels_last=False, 
    ) -> None:

        super(CNNUnetDecoder, self).__init__(layers=layers,
                                            activations=activations,
                                            pre_layer=pre_layer,
                                            case=case,
                                            last_activation=last_activation,
                                            transpose=transpose,
                                            flatten=flatten,
                                            name=name,
                                    )

        self.intermediary_inputs_indices = intermediary_inputs_indices
        
        if channels_last:
            self.concat_axis = -1
        else:
            self.concat_axis = 1

        self.list_of_layers = [layer_j for layer_j in self.list_of_layers 
                                              if not isinstance(layer_j, torch.nn.Identity)]
        self.pipeline = torch.nn.Sequential(*self.list_of_layers)

    #@as_tensor
    #@channels_dim
    def forward(
        self, input_data: Union[torch.Tensor, np.ndarray] = None,
        intermediary_encoder_outputs:List[torch.Tensor] = None,
    ) -> torch.Tensor:

        current_input = input_data
        intermediary_encoder_outputs = intermediary_encoder_outputs[::-1]
     
        for j, layer_j in enumerate(self.list_of_layers):

            if j in self.intermediary_inputs_indices:
                i = self.intermediary_inputs_indices.index(j)

                input_j = torch.cat([current_input, intermediary_encoder_outputs[i]], dim=self.concat_axis)
            else:
                input_j = current_input

            output_j = layer_j(input_j)
            current_input = output_j

        return current_input

class UNet(NetworkTemplate):


    def __init__(self, layers_config:dict=None,
                 intermediary_outputs_indices:List=None,
                 intermediary_inputs_indices:List=None,
                 encoder_extra_args:dict=dict(),
                 decoder_extra_args:dict=dict()) -> None:

        super(UNet, self).__init__()

        self.layers_config = layers_config
        self.intermediary_outputs_indices = intermediary_outputs_indices
        self.intermediary_inputs_indices = intermediary_inputs_indices

        self.layers_config_encoder = self.layers_config["encoder"] 
        self.layers_config_decoder = self.layers_config["decoder"] 

        self.encoder_activations = self.layers_config["encoder_activations"]
        self.decoder_activations = self.layers_config["decoder_activations"]

        self.encoder_horizontal_outputs = dict()
       
        # Configuring the encoder
        encoder_type = self.layers_config_encoder.get("type")
        layers_config_encoder = self.layers_config_encoder.get("architecture")
       
        if encoder_type == "cnn":
            self.encoder = CNNUnetEncoder(layers=self.layers_config_encoder["architecture"],
                                          activations=self.encoder_activations,
                                          intermediary_outputs_indices=self.intermediary_outputs_indices,
                                          case="2d", name="encoder",
                                          **encoder_extra_args)
        else:
            raise Exception(f"Option {encoder_type} is not available.")

         # Configuring the decoder
        decoder_type = self.layers_config_decoder.get("type")
        layers_config_encoder = self.layers_config_encoder.get("architecture")
        
        if encoder_type == "cnn":
            self.decoder = CNNUnetDecoder(layers=self.layers_config_decoder["architecture"],
                                          activations=self.decoder_activations,
                                          intermediary_inputs_indices=self.intermediary_inputs_indices,
                                          case="2d", name="decoder",
                                          **decoder_extra_args)
        else:
            raise Exception(f"Option {encoder_type} is not available.")

        self.add_module("encoder", self.encoder)
        self.add_module("decoder", self.decoder)

    @as_tensor
    def forward(self, input_data: Union[torch.Tensor, np.ndarray] = None
    ) -> torch.Tensor:

        encoder_main_output, encoder_intermediary_outputs = self.encoder(input_data=input_data)
        output = self.decoder(input_data = encoder_main_output,
                              intermediary_encoder_outputs=encoder_intermediary_outputs)

        return output
        
    def summary(self):

        print(self)
