import os
import time
from typing import Tuple

import torch
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from simulai.optimization import Optimizer

if not torch.cuda.is_available():
    raise Exception("There is no GPU available to execute the tests.")


def generate_data(
    n_samples: int = None,
    image_size: Tuple[int, int] = None,
    n_inputs: int = None,
    n_outputs: int = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate random input and output data.

    Parameters
    ----------
    n_samples : int, optional
        The number of samples to generate.
    image_size : tuple of ints, optional
        The size of the images.
    n_inputs : int, optional
        The number of input channels.
    n_outputs : int, optional
        The number of output channels.

    Returns
    -------
    input_data : torch.Tensor
        The generated input data.
    output_data : torch.Tensor
        The generated output data.

    Examples
    --------
    >>> input_data, output_data = generate_data(n_samples=10, image_size=(3, 32, 32),
    ...                                        n_inputs=3, n_outputs=10)
    """
    input_data = torch.rand(n_samples, n_inputs, *image_size)
    output_data = torch.rand(n_samples, n_outputs)

    return input_data, output_data


# DeepONet with a FNN as trunk and a CNN as branch
def model():
    """
    Create and return a DeepONet network.

    Returns
    -------
    net : torch.nn.Module
        The created DeepONet network.

    Examples
    --------
    >>> net = model()
    """
    from simulai.models import DeepONet
    from simulai.regression import ConvolutionalNetwork, DenseNetwork

    N_İNPUTS = 1
    N_OUTPUTS = 2

    N_LATENT = 50

    # Configuration for the fully-connected trunk network
    trunk_config = {
        "layers_units": 7 * [200],  # Hidden layers
        "activations": "tanh",
        "input_size": N_İNPUTS,
        "output_size": N_LATENT * N_OUTPUTS,
        "name": "trunk_net",
    }

    # CNN layers
    layers = [
        {
            "in_channels": N_İNPUTS,
            "out_channels": 2,
            "kernel_size": 3,
            "stride": 1,
            "padding": 1,
            "after_conv": {"type": "maxpool2d", "kernel_size": 2, "stride": 2},
        },
        {
            "in_channels": 2,
            "out_channels": 4,
            "kernel_size": 3,
            "stride": 1,
            "padding": 1,
            "after_conv": {"type": "maxpool2d", "kernel_size": 2, "stride": 2},
        },
        {
            "in_channels": 4,
            "out_channels": 8,
            "kernel_size": 3,
            "stride": 1,
            "padding": 1,
            "after_conv": {"type": "maxpool2d", "kernel_size": 2, "stride": 2},
        },
        {
            "in_channels": 8,
            "out_channels": N_LATENT * N_OUTPUTS,
            "kernel_size": 3,
            "stride": 1,
            "padding": 1,
            "after_conv": {"type": "maxpool2d", "kernel_size": 2, "stride": 2},
        },
    ]

    # Instantiating the surrogate model
    trunk_net = DenseNetwork(**trunk_config)
    branch_net = ConvolutionalNetwork(
        layers=layers, activations="sigmoid", case="2d", name="net", flatten=True
    )

    # It prints a summary of the network features
    trunk_net.summary()
    branch_net.summary(input_shape=[None, N_İNPUTS, 16, 16])

    net = DeepONet(
        trunk_network=trunk_net,
        branch_network=branch_net,
        var_dim=N_OUTPUTS,
        model_id="deeponet",
    )

    print(f"This network has {net.n_parameters} parameters.")

    return net


def evaluate_model(
    model: torch.nn.Module, test_data: Tuple[torch.Tensor, torch.Tensor]
) -> float:
    """
    Evaluate the given model on the given test data and return the mean squared error.

    Parameters
    ----------
    model : torch.nn.Module
        The model to evaluate.
    test_data : tuple of torch.Tensor
        The test data, as a tuple of input and output tensors.

    Returns
    -------
    mse : float
        The mean squared error of the model on the test data.

    Examples
    --------
    >>> model = model()
    >>> test_data = generate_data(n_samples=10, image_size=(3, 32, 32), n_inputs=3, n_outputs=10)
    >>> mse = evaluate_model(model, test_data)
    """
    model.eval()
    with torch.no_grad():
        inputs, targets = test_data
        outputs = model(inputs)
        mse = ((outputs - targets) ** 2).mean()
    return mse.item()


def execute_demo(rank, world_size):
    """Execute a distributed training demo with Data Parallelism.

    Parameters
    ----------
    rank : int
        The rank of the current process.
    world_size : int
        The number of processes participating in the distributed training.

    Returns
    -------
    elapsed_time : float
        The elapsed time for the training in seconds.
    """

    # Set environment variables for distributed training
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # Initialize the distributed training group
    dist = torch.distributed.init_process_group(
        backend="nccl", rank=rank, world_size=world_size
    )

    # Print the rank of the current process
    print(f"Executing DDP job in {rank}.")

    # rank = torch.distributed.get_rank()
    # print(f"Executing DDP job in {rank}.")

    # Choose the device to run the model on (based on the rank)
    device_id = rank % torch.cuda.device_count()

    # Create the model and move it to the chosen device
    net = model().to(device_id)

    # Create a parallel version of the model
    ddp_net = DDP(net, device_ids=[device_id])

    # Training parameters
    n_epochs = 50_000
    n_samples = 100_000
    batch_size = 1_000
    lr = 1e-3  # Initial learning rate for the ADAM algorithm

    # Instantiating optimizer
    params = {"lambda_1": 0.0, "lambda_2": 1e-10}
    optimizer_config = {"lr": lr}
    optimizer = Optimizer("adam", params=optimizer_config)

    # Generating datasets
    input_data, output_data = generate_data(
        n_samples=n_samples, image_size=(16, 16), n_inputs=1, n_outputs=16
    )

    # Train the model
    current_time = time.time()
    optimizer.fit(
        op=ddp_net,
        input_data=input_data,
        target_data=output_data,
        n_epochs=n_epochs,
        loss="rmse",
        params=params,
        batch_size=batch_size,
    )
    elapsed_time = time.time() - current_time

    print(f"Elapsed time for {rank} ranks: {elapsed_time} s.")


def exec(kind, n_ranks_list):
    """
    Execute the demo with the given kind of execution and a list of number of ranks.

    Parameters
    ----------
    kind : str
        The kind of execution. Can be either 'multiprocess' or 'distributed'.
    n_ranks_list : list of ints
        A list of number of ranks to use.

    Returns
    -------
    None

    Examples
    --------
    >>> exec('multiprocess', [2, 4, 8])
    >>> exec('distributed', [2, 4, 8])
    """

    if kind == "multiprocess":
        n_ranks = 1

        for n in n_ranks_list:
            mp.spawn(execute_demo, args=(n_ranks,), nprocs=n, join=True)

    elif kind == "distributed":
        for n in n_ranks_list:
            mp.spawn(execute_demo, args=(n,), nprocs=n, join=True)


if __name__ == "__main__":
    n_ranks_list = [2, 4, 8]

    exec("multiprocess", n_ranks_list)
    exec("distributed", n_ranks_list)
