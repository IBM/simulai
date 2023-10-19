import copy
import numpy as np
import torch
from typing import Union, Tuple

from simulai.templates import NetworkTemplate, as_tensor
from simulai.regression import SLFNN

class Base(NetworkTemplate):

    def __init__(self):
        pass

    def _activation_getter(self, activation: Union[str, torch.nn.Module]) -> torch.nn.Module:

        if isinstance(activation, torch.nn.Module):
            return encoder_activation
        elif isinstance(activation, str):
            return self._get_operation(operation=activation, is_activation=True)
        else:
            raise Exception(f"The activation {activation} is not supported.")

class BasicEncoder(Base):

    def __init__(self, num_heads=1,
                 activation:Union[str, torch.nn.Module]='relu',
                 mlp_layer:torch.Module=None,
                 input_dim:Union[int, Tuple]=None, 
                 ):

        super(BasicEncoder, self).__init__()

        self.num_heads = num_heads

        self.input_dim = input_dim

        self.activation = self._activation_getter(activation=activation)

        self.mlp_layer = mlp_layer

        self.self_attention = torch.nn.MultiheadAttention(embed_dim=self.input_dim,
                                                          num_heads=self.num_heads,
        self.add_module("mlp_layer", self.mlp_layer)                                                          batch_first=True)
        self.add_module("self_attention", self.self_attention)

    def forward(self, input_data: Union[torch.Tensor, np.ndarray] = None
    ) -> torch.Tensor:

        h = input_data
        h1 = self.activation(h)
        h = h + self.self_attention(h1, h1, h1)[0]
        h2 = self.activation(h)
        h = h + self.mlp_layer(h2)

        return h

class BasicDecoder(Base):

    def __init__(self, num_heads:int=1,
                 activation:Union[str, torch.nn.Module]='relu',
                 mlp_layer:torch.Module=None,
                 input_dim:Union[int, Tuple]=None):

        super(BasicDecoder, self).__init__()

        self.num_heads = num_heads

        self.input_dim = input_dim

        self.activation = self._activation_getter(activation=activation)

        self.mlp_layer = mlp_layer

        self.self_attention = torch.nn.MultiheadAttention(embed_dim=self.input_dim,
                                                          num_heads=self.num_heads,
                                                          batch_first=True)
        self.add_module("mlp_layer", self.mlp_layer)                                                          batch_first=True)
        self.add_module("self_attention", self.self_attention)

    def forward(self, input_data: Union[torch.Tensor, np.ndarray] = None,
                encoder_outputs:torch.Tensor=None,
    ) -> torch.Tensor:

        h = input_data
        h1 = self.activation(h)
        h = h + self.self_attention(h1, encoder_outputs, encoder_outputs)[0]
        h2 = self.activation(h)
        h = h + self.mlp_layer(h2)

        return h

class Transformer(NetworkTemplate):

    def __init__(self, num_heads_encoder:int=1,
                       num_heads_decoder:int=1,
                       input_dim:int=Union[int, Tuple],
                       encoder_activation: Union[str, torch.nn.Module]='relu',
                       decoder_activation: Union[str, torch.nn.Module]='relu', 
                       encoder_mlp_layer:torch.nn.Module=None,
                       decoder_mlp_layer:torch.nn.Module=None,
                       number_of_encoders:int=1,
                       number_of_decoders:int=1) -> None:

        super(Transformer, self).__init__()

        self.num_heads_encoder = num_heads_encoder
        self.num_heads_decoder = num_heads_decoder

        self.input_dim_encoder = input_dim_encoder
        self.input_dim_decoder = input_dim_decoder

        self.encoder_mlp_layer = encoder_mlp_layer
        self.decoder_mlp_layer = decoder_mlp_layer

        self.number_of_encoders = number_of_encoders
        self.number_of_decoders = number_of_encoders

        self.encoder_mlp_layers_list = list()
        self.decoder_mlp_layers_list = list()

        # Creating independent copies for the MLP layers which will be used 
        # by the multiple encoders/decoders.
        for e in range(self.number_of_encoders):
            self.encoder_mlp_layers_list.append(copy.deepcopy(self.encoder_mlp_layer))

        for d in range(self.number_of_decoders):
            self.encoder_mlp_layers_list.append(copy.deepcopy(self.encoder_mlp_layer))

        # Defining the encoder architecture
        self.EncoderStage = torch.nn.Sequential(
                                [BasicEncoder(num_heads=self.num_heads_encoder, 
                                              activation=self.encoder_activation,
                                              mlp_layer=self.encoder_mlp_layers_list[e],
                                              input_dim=self.input_dim_encoder) for e in range(self.number_of_encoders)]
                            )

        # Defining the decoder architecture
        self.DecoderStage =  torch.nn.Sequential(
                                [BasicDecoder(num_heads=self.num_heads_decoder,
                                              activation=self.decoder_activation,
                                              mlp_layer=self.encoder_mlp_layers_list[d],
                                              input_dim=self.input_dim_decoder) for d in range(self.number_of_decoders)]
                            )

        self.add_module("encoder", self.EncoderStage)
        self.add_module("decoder", self.DecoderStage)

        for encoder_e in self.EncoderStage:
            self.weights += encoder_e.weights

        for decoder_d in self.DecoderStage:
            self.weights += decoder_d.weights

    @as_tensor
    def forward(self, input_data: Union[torch.Tensor, np.ndarray] = None) -> torch.Tensor:

       encoder_outputs = self.EncoderStage(input_data)
       output = self.DecoderStage(input_data, encoder_output)

       return output

