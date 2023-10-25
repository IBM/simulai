import copy
import numpy as np
import torch
from typing import Union, List, Tuple, Optional, Dict

from simulai.templates import NetworkTemplate, as_tensor, channels_dim
from simulai.regression import DenseNetwork, SLFNN, ConvolutionalNetwork

# A CNN UNet encoder or decodeder is no more than a curved CNN
# in which intermediary outputs and inputs are also stored.
class CNNUnetEncoder(ConvolutionalNetwork):
    name = "convunetencoder"
    engine = "torch"

    def __init__(
        self,
        layers: List[Dict] = None,
        activations: Union[str, List[str]] = None,
        pre_layer: Optional[torch.nn.Module] = None,
        case: str = "2d",
        last_activation: str = "identity",
        transpose: bool = False,
        flatten: bool = False,
        intermediary_outputs_indices: List[int] = None,
        name: str = None,
    ) -> None:
        """
        A CNN encoder for U-Nets.

        Parameters
        ----------
        layers : List[Dict] 
            A list of configurations dictionaries for instantiating the layers.
        activations :
            A string or a list of strings defining the kind of activation to be used.
        pre_layer : Optional[torch.nn.Module]
            A layer for pre-processing the input.
        case : str
            The kind of CNN to be used, in ["1d", "2d", "3d"].
        last_activation : str
            The kind of activation to be used after the last layer.
        transpose : bool
            Using transposed convolution or not.
        flatten : bool
            Flattening the output or not.
        intermediary_outputs_indices : List[int],
            A list of indices for indicating what are the encoder outputs, which will be
            subsequently inputted in the decoder stage.
        name : str
            A name for the model. 

        """

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
    ) -> [torch.Tensor, List[torch.Tensor]]:
        """
        The CNN U-Net encoder forward method. 

        Parameters
        ----------
        input_data : Union[torch.Tensor, np.ndarray],
            A dataset to be inputted in the CNN U-Net encoder.

        Returns
        -------
        [torch.Tensor, List[torch.Tensor]]
            A list containing the main encoder output (latent space) and 
            another list of outputs, corresponding to the intermediary encoder
            outputs. 
        """

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
        """
        A CNN decoder for U-Nets.

        Parameters
        ----------
        layers : List[Dict] 
            A list of configurations dictionaries for instantiating the layers.
        activations :
            A string or a list of strings defining the kind of activation to be used.
        pre_layer : Optional[torch.nn.Module]
            A layer for pre-processing the input.
        case : str
            The kind of CNN to be used, in ["1d", "2d", "3d"].
        last_activation : str
            The kind of activation to be used after the last layer.
        transpose : bool
            Using transposed convolution or not.
        flatten : bool
            Flattening the output or not.
        intermediary_inputs_indices : List[int],
            A list of indices for indicating what are the decoder outputs.
        name : str
            A name for the model. 

        """


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

    def forward(
        self, input_data: Union[torch.Tensor, np.ndarray] = None,
        intermediary_encoder_outputs:List[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        The CNN U-Net decoder forward method. 

        Parameters
        ----------
        input_data : Union[torch.Tensor, np.ndarray],
            A dataset to be inputted in the CNN U-Net decoder.

        intermediary_encoder_outputs : List[torch.Tensor]
            A list of tensors, corresponding to the intermediary encoder outputs.

        Returns
        -------
        torch.Tensor
            The decoder (and U-Net) output.        
        """

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


    def __init__(self, layers_config:Dict=None,
                 intermediary_outputs_indices:List[int]=None,
                 intermediary_inputs_indices:List[int]=None,
                 encoder_extra_args:Dict=dict(),
                 decoder_extra_args:Dict=dict()) -> None:
        """
        U-Net. 

        Parameters
        ----------
        layers_config : Dict
            A dictionary containing the complete configuration for the 
            U-Net encoder and decoder.
        intermediary_outputs_indices : List[int]
            A list of indices for indicating the encoder outputs.
        intermediary_inputs_indices : List[int]
            A list of indices for indicating the decoder inputs.
        encoder_extra_args : Dict
            A dictionary containing extra arguments for the encoder.
        decoder_extra_args : Dict
            A dictionary containing extra arguments for the decoder. 

        """

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
        """
        The U-Net forward method. 

        Parameters
        ----------
        input_data : Union[torch.Tensor, np.ndarray],
            A dataset to be inputted in the CNN U-Net encoder.

        Returns
        -------
        torch.Tensor
            The U-Net output. 
        """

        encoder_main_output, encoder_intermediary_outputs = self.encoder(input_data=input_data)
        output = self.decoder(input_data = encoder_main_output,
                              intermediary_encoder_outputs=encoder_intermediary_outputs)

        return output
        
    def summary(self):
        """
        It shows a general view of the architecture.
        """

        print(self)
