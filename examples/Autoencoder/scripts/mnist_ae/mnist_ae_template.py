def model():

    from simulai.models import AutoencoderVariational

    autoencoder = AutoencoderVariational(
        input_dim=(None, 1, 28, 28),
        latent_dim=2,
        activation="relu",
        architecture="cnn",
        case="2d",
        shallow=False,
    )

    return autoencoder
