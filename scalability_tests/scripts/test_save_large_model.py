import numpy as np
import os

from simulai.models import AutoencoderVariational
from simulai.file import SPFile

input_data = np.random.rand(100, 1, 64, 128)

def model():

    from simulai.models import AutoencoderVariational

    autoencoder = AutoencoderVariational(
        input_dim=(None, 1, 64, 128),
        latent_dim=8,
        activation="tanh",
        architecture="cnn",
        case="2d",
        shallow=True,
        name='test_model',
    )
    return autoencoder

autoencoder = model()

estimated_data = autoencoder.eval(input_data=input_data)

autoencoder.summary()

model_name = "autoencoder_test"

print("Saving model.")

saver = SPFile(compact=False)
saver.write(save_dir="/tmp", name=model_name, model=autoencoder, template=model)

print("Restoring model.")

saver = SPFile(compact=False)

autoencoder_restored = saver.read(model_path=os.path.join("/tmp", model_name), device='cpu')

estimated_data_restored = autoencoder_restored.eval(input_data=input_data)

assert np.array_equal(estimated_data, estimated_data_restored) , (
    "The output of eval is not correct."
    f" Expected {output_data.shape},"
    f" but received {estimated_output_data.shape}."
)
