import time
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
backend = 'nccl'

def execute_demo():
    for n in n_ranks_list:

        dist = torch.distributed.init_process_group(backend=backend, world_size=n)

        rank = dist.get_rank()

        device_id = rank % torch.cuda.device_count()

        net = model().to(device_id)

        ddp_net = DDP(net, device_ids=[device_id])

        # Instantiating optimizer
        params = {'lambda_1': 0., 'lambda_2': 0.}
        optimizer = Optimizer('adam', params=optimizer_config)

        n_epochs = 100

        n_samples = 100_000
        batch_size=1_000
        lr = 1e-3  # Initial learning rate for the ADAM algorithm
        optimizer_config = {'lr': lr}

        # Generating datasets
        input_data, output_data = generate_data(n_samples=n_samples image_size=(16, 16), n_inputs=1, n_outputs=16)


        ### Training
        current_time = time.time()
        optimizer.fit(op=ddp_net, input_data=input_data, target_data=output_data,
                      n_epochs=n_epochs, loss="rmse", params=params, batch_size=batch_siz\e)
        elapsed_time = time.time() - elapsed_time

        print(f"Elapsed time for {n} ranks: {elapsed_time} s.")
        
if __name__ = "__main__":

    execute_demo()
