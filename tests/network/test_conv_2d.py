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

import os
import numpy as np
from unittest import TestCase
import torch

from simulai.file import SPFile
from simulai.optimization import Optimizer

from utils import configure_device

DEVICE = configure_device()

def generate_data(n_samples:int=None, image_size:tuple=None,
                  n_inputs:int=None, n_outputs:int=None) -> (torch.Tensor, torch.Tensor):

    input_data = np.random.rand(n_samples, n_inputs, *image_size)
    output_data = np.random.rand(n_samples, n_outputs)

    return torch.from_numpy(input_data.astype(np.float32)), torch.from_numpy(output_data.astype(np.float32))

# Model template
def model_2d():

    from simulai.regression import ConvolutionalNetwork

    # Configuring model
    n_inputs = 1

    layers = [

        {'in_channels': n_inputs, 'out_channels': 2, 'kernel_size': 3, 'stride': 1, 'padding': 1,
         'after_conv': {'type': 'maxpool2d', 'kernel_size': 2, 'stride': 2}},

        {'in_channels': 2, 'out_channels': 4, 'kernel_size': 3, 'stride': 1, 'padding': 1,
         'after_conv': {'type': 'maxpool2d', 'kernel_size': 2, 'stride': 2}},

        {'in_channels': 4, 'out_channels': 8, 'kernel_size': 3, 'stride': 1, 'padding': 1,
         'after_conv': {'type': 'maxpool2d', 'kernel_size': 2, 'stride': 2}},

        {'in_channels': 8, 'out_channels': 16, 'kernel_size': 3, 'stride': 1, 'padding': 1,
         'after_conv': {'type': 'maxpool2d', 'kernel_size': 2, 'stride': 2}}
    ]

    # Instantiating network
    convnet = ConvolutionalNetwork(layers=layers, activations='sigmoid', case='2d', name='net', flatten=True)

    return convnet

class TestConvNet2D(TestCase):

        def setUp(self) -> None:
            pass

        def test_convnet_2d_n_parameters(self):

            convnet = model_2d()

            assert type(convnet.n_parameters) == int

        def test_convnet_2d_eval(self):

            input_data, output_data = generate_data(n_samples=100, image_size=(16,16), n_inputs=1, n_outputs=16)

            convnet = model_2d()

            estimated_output_data = convnet.eval(input_data=input_data)

            assert estimated_output_data.shape == output_data.shape, "The output of eval is not correct." \
                                                                     f" Expected {output_data.shape}," \
                                                                     f" but received {estimated_output_data.shape}."

        def test_convnet_2d_save_restore(self):

            convnet = model_2d()

            input_data, output_data = generate_data(n_samples=100, image_size=(16, 16), n_inputs=1, n_outputs=16)

            model_name = f'convnet_{str(id(convnet))}'
            
            # Saving model
            print("Saving model.")

            saver = SPFile(compact=False)
            saver.write(save_dir="/tmp", name=model_name, model=convnet, template=model_2d)

            print("Restoring model.")

            convnet_restored = saver.read(model_path=os.path.join("/tmp", model_name))

            estimated_output_data = convnet_restored.eval(input_data=input_data)

            assert estimated_output_data.shape == output_data.shape, "The output of eval is not correct." \
                                                                     f" Expected {output_data.shape}," \
                                                                     f" but received {estimated_output_data.shape}."

        def test_convnet_2d_forward(self):

            n_epochs = 100

            lr = 1e-3  # Initial learning rate for the ADAM algorithm
            optimizer_config = {'lr': lr}

            input_data, output_data = generate_data(n_samples=100, image_size=(16, 16), n_inputs=1, n_outputs=16)

            convnet = model_2d()

            # Instnatiating optimizer
            params = {'lambda_1': 0., 'lambda_2': 0.}
            optimizer = Optimizer('adam', params=optimizer_config)

            ### Training
            optimizer.fit(op=convnet, input_data=input_data, target_data=output_data,
                          n_epochs=n_epochs, loss="rmse", params=params, batch_size=10, device=DEVICE)

            ### Evaluating
            estimated_output_data = convnet.eval(input_data=input_data)

            assert estimated_output_data.shape == output_data.shape, "The output of eval is not correct." \
                                                                     f" Expected {output_data.shape}," \
                                                                     f" but received {estimated_output_data.shape}."