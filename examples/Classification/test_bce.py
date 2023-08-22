import numpy as np

from simulai.models import SplitPool
from simulai.optimization import Optimizer
from simulai.regression import DenseNetwork

lr = 1e-3
optimizer_config = {"lr": lr}

optimizer = Optimizer(
    "adam",
    params=optimizer_config,
    lr_decay_scheduler_params={
        "name": "ExponentialLR",
        "gamma": 0.9,
        "decay_frequency": 5_000,
    },
)

n_inputs_b = 10
n_outputs = 1

config = {
    "layers_units": 7 * [100],  # Hidden layers
    "activations": "tanh",
    "input_size": 1,
    "output_size": n_outputs,
    "name": "branch_net",
}

agg_config = {
    "layers_units": 7 * [100],  # Hidden layers
    "activations": "tanh",
    "last_activation": "softmax",
    "input_size": n_inputs_b,
    "output_size": n_outputs,
    "name": "branch_net",
}

aggregation = DenseNetwork(**agg_config)

experts_list = list()
n_experts = 10
n_epochs = 1000

for ex in range(n_experts):
    experts_list.append(DenseNetwork(**config))

net = SplitPool(experts_list=experts_list, input_size=n_inputs_b, aggregation=aggregation,
                devices="gpu", last_activation="softmax")

input_data = np.random.rand(1_000, n_inputs_b)
target_data = np.random.rand(1_000, n_outputs)

params = {"lambda_1": 0.0, "lambda_2": 1e-6}

optimizer.fit(
    op=net,
    input_data=input_data,
    target_data=target_data,
    params=params,
    n_epochs=n_epochs,
    loss="bce",
    device="gpu",
)

