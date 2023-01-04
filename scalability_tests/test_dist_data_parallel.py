import torch
from torch.nn.parallel import DistributedDataParallel as DDP

if not torch.cuda.is_available():
    raise Exception("There is no gpu available to execute the tests.")

# DeepONet with a FNN as trunk and a CNN as branch
def model():

    from simulai.regression import DenseNetwork, ConvolutionalNetwork
    from simulai.models import DeepONet

    n_inputs = 1
    n_outputs = 2

    n_latent = 50

    # Configuration for the fully-connected trunk network
    trunk_config = {
        'layers_units': 7*[200],  # Hidden layers
        'activations': 'elu',
        'input_size': 1,
        'output_size': n_latent * n_outputs,
        'name': 'trunk_net'
    }

    # CNN layers
    layers = [

        {'in_channels': n_inputs, 'out_channels': 2, 'kernel_size': 3, 'stride': 1, 'padding': 1,
         'after_conv': {'type': 'maxpool2d', 'kernel_size': 2, 'stride': 2}},

        {'in_channels': 2, 'out_channels': 4, 'kernel_size': 3, 'stride': 1, 'padding': 1,
         'after_conv': {'type': 'maxpool2d', 'kernel_size': 2, 'stride': 2}},

        {'in_channels': 4, 'out_channels': 8, 'kernel_size': 3, 'stride': 1, 'padding': 1,
         'after_conv': {'type': 'maxpool2d', 'kernel_size': 2, 'stride': 2}},

        {'in_channels': 8, 'out_channels': n_latent * n_outputs, 'kernel_size': 3, 'stride': 1, 'padding': 1,
         'after_conv': {'type': 'maxpool2d', 'kernel_size': 2, 'stride': 2}}
    ]

    # Instantiating and training the surrogate model
    trunk_net = DenseNetwork(**trunk_config)
    branch_net = ConvolutionalNetwork(layers=layers, activations='sigmoid', case='2d', name='net', flatten=True)

    # It prints a summary of the network features
    trunk_net.summary()
    branch_net.summary(input_shape=[None, 1, 16, 16])

    net = DeepONet(trunk_network=trunk_net,
                   branch_network=branch_net,
                   var_dim=n_outputs,
                   model_id='deeponet')

    print(f"This network has {net.n_parameters} parameters.")

    return net

n_ranks_list = [2, 4, 8]

for n in n_ranks_list:

    net = model()

    torch.cuda.set_device(1)

    torch.distributed.init_process_group(backend='nccl', world_size=n)

