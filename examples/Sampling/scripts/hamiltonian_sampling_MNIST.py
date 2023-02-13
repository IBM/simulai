import os
from argparse import ArgumentParser

import numpy as np

from simulai.file import SPFile
from simulai.metrics import L2Norm
from simulai.optimization import Optimizer
from simulai.sampling import HMC, G_metric, HamiltonianEquations, LeapFrogIntegrator


def model():
    from simulai.models import AutoencoderVariational
    from simulai.regression import SLFNN, ConvolutionalNetwork

    transpose = False

    n_inputs = 1
    n_outputs = 1

    ### Layers Configurations ####
    ### BEGIN
    encoder_layers = [
        {
            "in_channels": n_inputs,
            "out_channels": 16,
            "kernel_size": 3,
            "stride": 1,
            "padding": 1,
            "after_conv": {"type": "maxpool2d", "kernel_size": 2, "stride": 2},
        },
        {
            "in_channels": 16,
            "out_channels": 32,
            "kernel_size": 3,
            "stride": 1,
            "padding": 1,
            "after_conv": {"type": "maxpool2d", "kernel_size": 2, "stride": 2},
        },
        {
            "in_channels": 32,
            "out_channels": 64,
            "kernel_size": 3,
            "stride": 1,
            "padding": 1,
            "after_conv": {"type": "maxpool2d", "kernel_size": 2, "stride": 2},
        },
    ]

    bottleneck_encoder_layers = {
        "input_size": 576,
        "output_size": 16,
        "activation": "identity",
        "name": "bottleneck_encoder",
    }

    bottleneck_decoder_layers = {
        "input_size": 16,
        "output_size": 576,
        "activation": "identity",
        "name": "bottleneck_decoder",
    }

    decoder_layers = [
        {
            "in_channels": 64,
            "out_channels": 32,
            "kernel_size": 3,
            "stride": 1,
            "padding": 1,
            "before_conv": {"type": "upsample", "scale_factor": 2, "mode": "bicubic"},
        },
        {
            "in_channels": 32,
            "out_channels": 16,
            "kernel_size": 3,
            "stride": 1,
            "padding": 1,
            "before_conv": {"type": "upsample", "scale_factor": 2, "mode": "bicubic"},
        },
        {
            "in_channels": 16,
            "out_channels": n_outputs,
            "kernel_size": 3,
            "stride": 1,
            "padding": 3,
            "before_conv": {"type": "upsample", "scale_factor": 2, "mode": "bicubic"},
        },
    ]

    ### END
    ### Layers Configurations ####

    # Instantiating network
    encoder = ConvolutionalNetwork(
        layers=encoder_layers, activations="tanh", case="2d", name="encoder"
    )
    bottleneck_encoder = SLFNN(**bottleneck_encoder_layers)
    bottleneck_decoder = SLFNN(**bottleneck_decoder_layers)
    decoder = ConvolutionalNetwork(
        layers=decoder_layers,
        activations="tanh",
        case="2d",
        transpose=transpose,
        name="decoder",
    )

    autoencoder = AutoencoderVariational(
        encoder=encoder,
        bottleneck_encoder=bottleneck_encoder,
        bottleneck_decoder=bottleneck_decoder,
        decoder=decoder,
        encoder_activation="tanh",
    )

    print(f"Network has {autoencoder.n_parameters} parameters.")

    return autoencoder


def train_autoencoder_mnist(
    train_data: np.ndarray = None,
    test_data: np.ndarray = None,
    path: str = None,
    model_name: str = None,
    n_epochs: int = None,
    batch_size: int = None,
):
    lr = 1e-3

    autoencoder = model()

    autoencoder.summary(input_shape=list(train_data.shape))

    optimizer_config = {"lr": lr}
    params = {"lambda_1": 0.0, "lambda_2": 0.0, "use_mean": False, "relative": True}

    optimizer = Optimizer("adam", params=optimizer_config)

    optimizer.fit(
        op=autoencoder,
        input_data=train_data,
        target_data=train_data,
        n_epochs=n_epochs,
        loss="vaermse",
        params=params,
        batch_size=batch_size,
        device="gpu",
    )

    saver = SPFile(compact=False)
    saver.write(save_dir=path, name=model_name, model=autoencoder, template=model)

    estimated_test_data = autoencoder.eval(input_data=test_data)

    l2_norm = L2Norm()

    error = 100 * l2_norm(
        data=estimated_test_data, reference_data=test_data, relative_norm=True
    )

    print(f"Projection error: {error} %")


def eval_autoencoder(
    model_name: str = None, test_data: np.ndarray = None, path: str = None
):
    saver = SPFile(compact=False)
    autoencoder = saver.read(model_path=os.path.join(path, model_name))
    autoencoder.summary(input_shape=list(test_data.shape))

    Mu = autoencoder.Mu(input_data=test_data)

    # Evaluating latent variables
    G_z = G_metric(k=50, tau=0.001, lambd=1, model=autoencoder, input_data=test_data)

    hamiltonian = HamiltonianEquations(metric=G_z)

    n_steps = 10
    N = 10
    e_lf = 0.01

    integrator = LeapFrogIntegrator(system=hamiltonian, n_steps=n_steps, e_lf=e_lf)
    sampler = HMC(integrator=integrator, N=N)

    z_sampled = sampler.solve(z_0=Mu[100])

    reconstructed = autoencoder.reconstruct(input_data=z_sampled)
    reconstructed_ = autoencoder.reconstruct(input_data=Mu[100])

    assert reconstructed_ - reconstructed


if __name__ == "__main__":
    # Reading command line arguments.
    parser = ArgumentParser(description="Reading input parameters")

    parser.add_argument(
        "--data_path", type=str, help="Path to the dataset", default="/tmp/mnist.npz"
    )
    args = parser.parse_args()
    data_path = args.data_path

    path = os.path.basename(data_path)

    data = np.load(data_path)
    model_name = "autoencoder_mnist"
    train_data = data["x_train"][:, None, ...]
    test_data = data["x_test"][:, None, ...]

    n_epochs = 10_000
    batch_size = 1_000

    train_autoencoder_mnist(
        train_data=train_data,
        test_data=test_data,
        model_name=model_name,
        path=path,
        n_epochs=n_epochs,
        batch_size=batch_size,
    )

    # eval_autoencoder(model_name=model_name, test_data=test_data)
