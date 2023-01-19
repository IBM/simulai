import os
import numpy as np
from unittest import TestCase

from simulai.optimization import Optimizer
from simulai.file import SPFile

def model():

    from simulai.regression import DenseNetwork
    from simulai.models import AutoencoderMLP

    encoder_config = {
                        'layers_units': [256, 128, 64, 16],  # Hidden layers
                        'activations': 'tanh',
                        'input_size': 256,
                        'output_size': 8,
                        'name': 'trunk_net'
                     }

    decoder_config  = {
                        'layers_units': [16, 64, 128, 256],  # Hidden layers
                        'activations': 'tanh',
                        'input_size': 8,
                        'output_size': 256,
                        'name': 'trunk_net'
                        }

    encoder = DenseNetwork(**encoder_config)
    decoder = DenseNetwork(**decoder_config)

    autoencoder = AutoencoderMLP(encoder=encoder, decoder=decoder)

    print(f"Network has {autoencoder.n_parameters} parameters.")

    return autoencoder

class TestAutoencoder(TestCase):

    def setUp(self) -> None:
        pass

    def test_autoencoder_eval(self):

        data = np.random.rand(1_000, 256)

        autoencoder = model()

        estimated_output = autoencoder.eval(input_data=data)

        assert estimated_output.shape == data.shape

    def test_autoencoder_save_restore(self):

        data = np.random.rand(1_000, 256)

        autoencoder = model()

        saver = SPFile(compact=False)
        saver.write(save_dir="/tmp", name=f'autoencoder_{id(autoencoder)}', model=autoencoder, template=model)

        autoencoder_reload = saver.read(model_path=os.path.join('/tmp', f'autoencoder_{id(autoencoder)}'))

        estimated_output = autoencoder_reload.eval(input_data=data)

        assert estimated_output.shape == data.shape

    def test_autoencoder_train(self):

        params =  {'lambda_1': 0., 'lambda_2': 0., 'axis': 1,
                    'weights': 256*[1/256], 'use_mean':False, 'relative':True}

        data = np.random.rand(1_000, 256)

        lr = 1e-3
        n_epochs = 10

        autoencoder = model()

        autoencoder.summary()

        optimizer_config = {'lr': lr}

        optimizer = Optimizer('adam', params=optimizer_config)

        optimizer.fit(op=autoencoder, input_data=data, target_data=data,
                      n_epochs=n_epochs, loss='wrmse', params=params)

        saver = SPFile(compact=False)
        saver.write(save_dir="/tmp", name='autoencoder_rb_just_test', model=autoencoder, template=model)
