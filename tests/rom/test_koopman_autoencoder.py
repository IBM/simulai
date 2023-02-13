import os
from unittest import TestCase

import numpy as np
from test_cnn_autoencoder import model

from simulai.file import SPFile
from simulai.optimization import Optimizer


class TestAutoencoder(TestCase):
    def setUp(self) -> None:
        pass

    def test_autoencoder_eval(self):
        data = np.random.rand(1_000, 3, 16, 16)

        autoencoder = model()

        estimated_output = autoencoder.eval(input_data=data)

        assert estimated_output.shape == data.shape

    def test_autoencoder_save_restore(self):
        data = np.random.rand(1_000, 3, 16, 16)

        autoencoder = model(architecture="AutoencoderKoopman")

        saver = SPFile(compact=False)
        saver.write(
            save_dir="/tmp",
            name=f"autoencoder_{id(autoencoder)}",
            model=autoencoder,
            template=model,
        )

        autoencoder_reload = saver.read(
            model_path=os.path.join("/tmp", f"autoencoder_{id(autoencoder)}")
        )

        estimated_output = autoencoder_reload.eval(input_data=data)

        assert estimated_output.shape == data.shape

    def test_autoencoder_train(self):
        loss_function = "kaermse"

        params = {"lambda_1": 0.0, "lambda_2": 0.0, "use_mean": False, "relative": True}

        data = np.random.rand(1_000, 3, 16, 16)

        lr = 1e-3
        n_epochs = 10

        autoencoder = model(architecture="AutoencoderKoopman")

        autoencoder.summary(input_shape=[None, 3, 16, 16])

        optimizer_config = {"lr": lr}

        optimizer = Optimizer("adam", params=optimizer_config)

        optimizer.fit(
            op=autoencoder,
            input_data=data,
            target_data=data,
            n_epochs=n_epochs,
            loss=loss_function,
            params=params,
        )

        saver = SPFile(compact=False)
        saver.write(
            save_dir="/tmp",
            name="autoencoder_rb_just_test",
            model=autoencoder,
            template=model,
        )
