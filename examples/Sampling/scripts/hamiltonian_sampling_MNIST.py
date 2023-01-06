import os
import numpy as np

from simulai.optimization import Optimizer
from simulai.metrics import L2Norm
from simulai.file import SPFile

def model():

    from simulai.regression import ConvolutionalNetwork, SLFNN
    from simulai.models import AutoencoderVariational

    transpose = False

    n_inputs = 1
    n_outputs = 1

    ### Layers Configurations ####
    ### BEGIN
    encoder_layers = [

                          {'in_channels': n_inputs, 'out_channels': 16, 'kernel_size': 3, 'stride': 1, 'padding': 1,
                           'after_conv': {'type':'maxpool2d', 'kernel_size': 2, 'stride': 2}},

                          {'in_channels': 16, 'out_channels': 32, 'kernel_size': 3, 'stride': 1, 'padding': 1,
                           'after_conv': {'type':'maxpool2d', 'kernel_size': 2, 'stride': 2}},

                          {'in_channels': 32, 'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1,
                          'after_conv': {'type': 'maxpool2d', 'kernel_size': 2, 'stride': 2}},

                      ]

    bottleneck_encoder_layers = {
                                 'input_size': 576,
                                 'output_size': 16,
                                 'activation': 'identity',
                                 'name': 'bottleneck_encoder'
                                }

    bottleneck_decoder_layers = {
                                 'input_size': 16,
                                 'output_size': 576,
                                 'activation': 'identity',
                                 'name': 'bottleneck_decoder'
                                }

    decoder_layers = [

                        {'in_channels': 64, 'out_channels': 32, 'kernel_size': 3, 'stride': 1, 'padding': 1,
                         'before_conv': {'type': 'upsample', 'scale_factor': 2, 'mode': 'bicubic'}},

                        {'in_channels': 32, 'out_channels': 16, 'kernel_size': 3, 'stride': 1, 'padding': 1,
                         'before_conv': {'type': 'upsample', 'scale_factor': 2, 'mode': 'bicubic'}},

                        {'in_channels': 16, 'out_channels': n_outputs, 'kernel_size': 3, 'stride': 1, 'padding': 3,
                         'before_conv': {'type': 'upsample', 'scale_factor': 2, 'mode': 'bicubic'}}

    ]

    ### END
    ### Layers Configurations ####

    # Instantiating network
    encoder = ConvolutionalNetwork(layers=encoder_layers, activations='tanh', case='2d', name="encoder")
    bottleneck_encoder = SLFNN(**bottleneck_encoder_layers)
    bottleneck_decoder = SLFNN(**bottleneck_decoder_layers)
    decoder = ConvolutionalNetwork(layers=decoder_layers, activations='tanh', case='2d', transpose=transpose,
                                   name="decoder")

    autoencoder = AutoencoderVariational(encoder=encoder, bottleneck_encoder=bottleneck_encoder,
                                         bottleneck_decoder=bottleneck_decoder,
                                         decoder=decoder, encoder_activation='tanh')

    print(f"Network has {autoencoder.n_parameters} parameters.")

    return autoencoder

def train_autoencoder_mnist(train_data:np.ndarray=None, test_data:np.ndarray=None,
                            model_name:str=None):

    lr = 1e-3
    n_epochs = 10_00
    batch_size = 1_000

    autoencoder = model()

    autoencoder.summary(input_shape=list(train_data.shape))

    optimizer_config = {'lr': lr}
    params = {'lambda_1': 0., 'lambda_2': 0., 'use_mean':False, 'relative':True}

    optimizer = Optimizer('adam', params=optimizer_config)

    optimizer.fit(op=autoencoder, input_data=train_data, target_data=train_data,
                  n_epochs=n_epochs, loss="vaermse", params=params, batch_size=batch_size, device='gpu')

    saver = SPFile(compact=False)
    saver.write(save_dir="/tmp", name=model_name, model=autoencoder, template=model)

    estimated_test_data = autoencoder.eval(input_data=test_data)

    l2_norm = L2Norm()

    error = 100*l2_norm(data=estimated_test_data, reference_data=test_data, relative_norm=True)

    print(f'Projection error: {error} %')

def eval_autoencoder(model_name:str=None, test_data:np.ndarray=None):

    saver = SPFile(compact=False)
    autoencoder = saver.read(model_path=os.path.join('/tmp', model_name))

    print('')

if __name__ == "__main__":

    data = np.load('/tmp/mnist.npz')
    model_name = 'autoencoder_mnist'
    train_data = data['x_train'][:, None, ...]
    test_data = data['x_test'][:, None, ...]

    train_autoencoder_mnist(train_data=train_data, test_data=test_data)
    eval_autoencoder(model_name=model_name, test_data=test_data)