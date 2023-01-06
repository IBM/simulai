import os
import time
import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp

from simulai.optimization import Optimizer

if not torch.cuda.is_available():
    raise Exception("There is no GPU available to execute the tests.")

def generate_data(n_samples:int=None, image_size:tuple=None,
                  n_inputs:int=None, n_outputs:int=None) -> (torch.Tensor, torch.Tensor):

    input_data = np.random.rand(n_samples, n_inputs, *image_size)
    output_data = np.random.rand(n_samples, n_outputs)

    return torch.from_numpy(input_data.astype(np.float32)), torch.from_numpy(output_data.astype(np.float32))

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
        'activations': 'tanh',
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

backend = 'nccl'

def execute_demo(rank, world_size):

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    dist = torch.distributed.init_process_group(backend=backend, rank=rank, world_size=world_size)

    #rank = torch.distributed.get_rank()
    print(f"Executing DDP job in {rank}.")

    device_id = rank % torch.cuda.device_count()

    net = model().to(device_id)

    ddp_net = DDP(net, device_ids=[device_id])

    n_epochs = 50_000

    n_samples = 100_000
    batch_size = 1_000
    lr = 1e-3  # Initial learning rate for the ADAM algorithm

    # Instantiating optimizer
    params = {'lambda_1': 0., 'lambda_2': 1e-10}
    optimizer_config = {'lr': lr}
    optimizer = Optimizer('adam', params=optimizer_config)

    # Generating datasets
    input_data, output_data = generate_data(n_samples=n_samples, image_size=(16, 16), n_inputs=1, n_outputs=16)

    ### Training
    current_time = time.time()
    optimizer.fit(op=ddp_net, input_data=input_data, target_data=output_data,
                  n_epochs=n_epochs, loss="rmse", params=params, batch_size=batch_size)
    elapsed_time = time.time() - current_time

    print(f"Elapsed time for {rank} ranks: {elapsed_time} s.")

def exec(kind, n_ranks_list):

    if kind == 'multiprocess':
        n_ranks = 1

        for n in n_ranks_list:
            mp.spawn(execute_demo, args=(n_ranks,), nprocs=n, join=True)

    elif kind == 'distributed':

        for n in n_ranks_list:
            mp.spawn(execute_demo, args=(n,), nprocs=n, join=True)

if __name__ == "__main__":

    n_ranks_list = [2, 4, 8]

    exec('multiprocess', n_ranks_list)

    exec('distributed', n_ranks_list)
