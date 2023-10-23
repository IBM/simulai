import copy
import numpy as np
import torch
from typing import Union, Tuple

from simulai.templates import NetworkTemplate, as_tensor
from simulai.regression import DenseNetwork, SLFNN

class BaseTemplate(NetworkTemplate):

    def __init__(self):

        super(BaseTemplate, self).__init__()

    def _activation_getter(self, activation: Union[str, torch.nn.Module]) -> torch.nn.Module:

        if isinstance(activation, torch.nn.Module):
            return encoder_activation
        elif isinstance(activation, str):
            return self._get_operation(operation=activation, is_activation=True)
        else:
            raise Exception(f"The activation {activation} is not supported.")

class BasicEncoder(BaseTemplate):

    def __init__(self, num_heads=1,
                 activation:Union[str, torch.nn.Module]='relu',
                 mlp_layer:torch.nn.Module=None,
                 embed_dim:Union[int, Tuple]=None,
                 ):

        super(BasicEncoder, self).__init__()

        self.num_heads = num_heads

        self.embed_dim = embed_dim

        self.activation_1 = self._activation_getter(activation=activation)
        self.activation_2 = self._activation_getter(activation=activation)

        self.mlp_layer = mlp_layer

        self.self_attention = torch.nn.MultiheadAttention(embed_dim=self.embed_dim,
                                                          num_heads=self.num_heads,
                                                          batch_first=True)

        self.add_module("mlp_layer", self.mlp_layer)
        self.add_module("self_attention", self.self_attention)

        # Attention heads are not being included in the weights regularization
        self.weights = list()
        self.weights += self.mlp_layer.weights

    def forward(self, input_data: Union[torch.Tensor, np.ndarray] = None
    ) -> torch.Tensor:

        h = input_data
        h1 = self.activation_1(h)
        h = h + self.self_attention(h1, h1, h1)[0]
        h2 = self.activation_2(h)
        h = h + self.mlp_layer(h2)

        return h

class BasicDecoder(BaseTemplate):

    def __init__(self, num_heads:int=1,
                 activation:Union[str, torch.nn.Module]='relu',
                 mlp_layer:torch.nn.Module=None,
                 embed_dim:Union[int, Tuple]=None):

        super(BasicDecoder, self).__init__()

        self.num_heads = num_heads

        self.embed_dim = embed_dim

        self.activation_1 = self._activation_getter(activation=activation)
        self.activation_2 = self._activation_getter(activation=activation)

        self.mlp_layer = mlp_layer

        self.self_attention = torch.nn.MultiheadAttention(embed_dim=self.embed_dim,
                                                          num_heads=self.num_heads,
                                                          batch_first=True)
        self.add_module("mlp_layer", self.mlp_layer)
        self.add_module("self_attention", self.self_attention)

        # Attention heads are not being included in the weights regularization
        self.weights = list()
        self.weights += self.mlp_layer.weights


    def forward(self, input_data: Union[torch.Tensor, np.ndarray] = None,
                encoder_output:torch.Tensor=None,
    ) -> torch.Tensor:

        h = input_data
        h1 = self.activation_1(h)
        h = h + self.self_attention(h1, encoder_output, encoder_output)[0]
        h2 = self.activation_2(h)
        h = h + self.mlp_layer(h2)

        return h

class Transformer(NetworkTemplate):

    def __init__(self, num_heads_encoder:int=1,
                       num_heads_decoder:int=1,
                       embed_dim_encoder:int=Union[int, Tuple],
                       embed_dim_decoder:int=Union[int, Tuple],
                       encoder_activation: Union[str, torch.nn.Module]='relu',
                       decoder_activation: Union[str, torch.nn.Module]='relu',
                       encoder_mlp_layer_config:dict=None,
                       decoder_mlp_layer_config:dict=None,
                       number_of_encoders:int=1,
                       number_of_decoders:int=1) -> None:

        super(Transformer, self).__init__()

        self.num_heads_encoder = num_heads_encoder
        self.num_heads_decoder = num_heads_decoder

        self.embed_dim_encoder = embed_dim_encoder
        self.embed_dim_decoder = embed_dim_decoder

        self.encoder_mlp_layer_dict = encoder_mlp_layer_config
        self.decoder_mlp_layer_dict = decoder_mlp_layer_config

        self.number_of_encoders = number_of_encoders
        self.number_of_decoders = number_of_encoders

        self.encoder_activation = encoder_activation
        self.decoder_activation = decoder_activation

        self.encoder_mlp_layers_list = list()
        self.decoder_mlp_layers_list = list()

        # Creating independent copies for the MLP layers which will be used 
        # by the multiple encoders/decoders.
        for e in range(self.number_of_encoders):
            self.encoder_mlp_layers_list.append(
                                                    DenseNetwork(**self.encoder_mlp_layer_dict)
                                               )

        for d in range(self.number_of_decoders):
            self.decoder_mlp_layers_list.append(

                                                    DenseNetwork(**self.decoder_mlp_layer_dict)
                                               )

        # Defining the encoder architecture
        self.EncoderStage = torch.nn.Sequential(
                                *[BasicEncoder(num_heads=self.num_heads_encoder,
                                              activation=self.encoder_activation,
                                              mlp_layer=self.encoder_mlp_layers_list[e],
                                              embed_dim=self.embed_dim_encoder) for e in range(self.number_of_encoders)]
                            )

        # Defining the decoder architecture
        self.DecoderStage =  [BasicDecoder(num_heads=self.num_heads_decoder,
                                           activation=self.decoder_activation,
                                           mlp_layer=self.decoder_mlp_layers_list[d],
                                           embed_dim=self.embed_dim_decoder) for d in range(self.number_of_decoders)
                              ]



        self.weights = list()

        for e, encoder_e in enumerate(self.EncoderStage):
            self.weights += encoder_e.weights
            self.add_module(f"encoder_{e}", encoder_e)

        for d, decoder_d in enumerate(self.DecoderStage):
            self.weights += decoder_d.weights
            self.add_module(f"decoder_{d}", decoder_d)

    @as_tensor
    def forward(self, input_data: Union[torch.Tensor, np.ndarray] = None) -> torch.Tensor:

       encoder_output = self.EncoderStage(input_data)

       current_input = input_data
       for decoder in self.DecoderStage:
           output = decoder(input_data=current_input, encoder_output=encoder_output)
           current_input = output

       return output

    def summary(self):

       print(self)
