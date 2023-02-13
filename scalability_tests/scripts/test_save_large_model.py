import os

import numpy as np

from simulai.file import SPFile
from simulai.optimization import Optimizer

input_data = np.random.rand(100, 1, 80, 80)


def model():
    from simulai.models import AutoencoderVariational

    autoencoder = AutoencoderVariational(
        input_dim=(None, 1, 80, 80),
        latent_dim=8,
        activation="tanh",
        architecture="cnn",
        case="2d",
        shallow=True,
        name="test_model",
        devices="gpu",
    )
    return autoencoder


if __name__ == "__main__":
    DEVICE = "gpu"

    autoencoder = model()

    autoencoder.summary()

    optimizer_config = {"lr": 1e-3}

    maximum_values = (1 / np.linalg.norm(input_data, 2, axis=0)).tolist()
    params = {"use_mean": True, "relative": True, "beta": 1}

    optimizer = Optimizer("adam", params=optimizer_config)

    optimizer.fit(
        op=autoencoder,
        input_data=input_data,
        target_data=input_data,
        n_epochs=100,
        loss="vaermse",
        params=params,
        device=DEVICE,
    )

    estimated_data = autoencoder.eval(input_data=input_data)

    model_name = "autoencoder_test"

    print("Saving model.")

    saver = SPFile(compact=False)
    saver.write(save_dir="/tmp", name=model_name, model=autoencoder, template=model)

    print("Restoring model.")

    saver = SPFile(compact=False)

    autoencoder_restored = saver.read(
        model_path=os.path.join("/tmp", model_name), device="cpu"
    )

    estimated_data_restored = autoencoder_restored.eval(input_data=input_data)

    assert np.array_equal(
        estimated_data, estimated_data_restored
    ), "The output of eval is not correct."
