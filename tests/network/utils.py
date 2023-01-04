import os

def configure_device():

    simulai_network_gpu = os.getenv('SIMULAI_NETWORK_GPU')

    if not simulai_network_gpu:
        device = 'cpu'
    else:
        import torch

        if not torch.cuda.is_available():
            raise Exception("There is no gpu available to execute the tests.")

        else:
            device = 'gpu'

    print(f"Using device: {device}")

    return device