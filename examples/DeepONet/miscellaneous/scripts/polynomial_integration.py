import numpy as np
import torch 

from simulai.residuals import SymbolicOperator
from simulai.tokens import Dot, Gp
from simulai.optimization import Optimizer

def model():

    import numpy as np

    from simulai.models import DeepONet, ImprovedDenseNetwork
    from simulai.regression import SLFNN, ConvexDenseNetwork

    n_latent = 100
    n_inputs_b = 5
    n_inputs_t = 1
    n_outputs = 1

    # Configuration for the fully-connected trunk network
    trunk_config = {
        "layers_units": 7 * [100],  # Hidden layers
        "activations": "tanh",
        "input_size": n_inputs_t,
        "output_size": n_latent * n_outputs,
        "name": "trunk_net",
    }

    # Configuration for the fully-connected branch network
    branch_config = {
        "layers_units": 7 * [100],  # Hidden layers
        "activations": "tanh",
        "input_size": n_inputs_b,
        "output_size": n_latent * n_outputs,
        "name": "branch_net",
    }

    encoder_u_trunk = SLFNN(input_size=n_inputs_t, output_size=100, activation="tanh")
    encoder_v_trunk = SLFNN(input_size=n_inputs_t, output_size=100, activation="tanh")
    encoder_u_branch = SLFNN(input_size=n_inputs_b, output_size=100, activation="tanh")
    encoder_v_branch = SLFNN(input_size=n_inputs_b, output_size=100, activation="tanh")

    # Instantiating and training the surrogate model
    trunk_net_dense = ConvexDenseNetwork(**trunk_config)
    branch_net_dense = ConvexDenseNetwork(**branch_config)

    trunk_net = ImprovedDenseNetwork(
        network=trunk_net_dense, encoder_u=encoder_u_trunk, encoder_v=encoder_v_trunk
    )

    branch_net = ImprovedDenseNetwork(
        network=branch_net_dense, encoder_u=encoder_u_branch, encoder_v=encoder_v_branch
    )

    # It prints a summary of the network features
    trunk_net.summary()
    branch_net.summary()

    deep_o_net = DeepONet(
        trunk_network=trunk_net,
        branch_network=branch_net,
        var_dim=n_outputs,
        rescale_factors=np.array([1]),
        devices="gpu",
        model_id="flame_net",
    )

    return deep_o_net

f = f"D(u, t) - Dot(a, Gp(t, t, 4))"

input_labels = ["t", "a"]
output_labels = ["u"]
save_path = "./"
model_name = "polynomial_integrator"

T = 1
t_interval = [0, T]
N = 1_00

net = model()

residual = SymbolicOperator(
    expressions=[f],
    input_vars=input_labels,
    external_functions={"Dot": Dot, "Gp": Gp},
    output_vars=output_labels,
    inputs_key="input_trunk|input_branch[0,4]",
    function=net,
    engine="torch",
)

t = np.linspace(*t_interval, N)[:, None]
a = np.random.rand(1_000, 5)
input_trunk = np.tile(t, (1_000, 1))
input_branch = np.tile(a, (N, 1))

input_data = {"input_trunk": input_trunk, "input_branch": input_branch}

optimizer_config = {"lr": 1e-3}
lambda_1 = 0.0
lambda_2 = 0.0
n_epochs = 1_000
batch_size = 100

optimizer = Optimizer(
    "adam",
    params=optimizer_config,
    lr_decay_scheduler_params={
        "name": "ExponentialLR",
        "gamma": 0.9,
        "decay_frequency": 5_000,
    },
    checkpoint_params={
        "save_dir": save_path,
        "name": model_name,
        "template": model,
        "checkpoint_frequency": 10_000,
        "overwrite": False,
    },
    summary_writer=True,
)

params = {
    "lambda_1": lambda_1,
    "lambda_2": lambda_2,
    "residual": residual,
    "weights_residual": [1, 1, 1],
    "initial_input": {"input_trunk": np.zeros((1_000, 1)), "input_branch": a},
    "initial_state": np.zeros((1_000, 1)),
}

optimizer.fit(
    op=net,
    input_data=input_data,
    n_epochs=n_epochs,
    loss="opirmse",
    params=params,
    device="gpu",
    batch_size=1_00,
    use_jit=True,
)


