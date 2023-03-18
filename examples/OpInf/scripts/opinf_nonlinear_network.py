# (C) Copyright IBM Corp. 2019, 2020, 2021, 2022.

#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at

#           http://www.apache.org/licenses/LICENSE-2.0

#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

# It is necessary to share the correct engine before starting the
# job
os.environ["engine"] = "pytorch"

from examples.utils.lorenz_solver import lorenz_solver
from simulai.math.integration import LSODA, ClassWrapper
from simulai.optimization import Optimizer, ScipyInterface
from simulai.regression import (
    AutoEncoderKoopman,
    DenseNetwork,
    KoopmanNetwork,
    OpInfNetwork,
)


class LorenzJacobian:
    def __init__(self, sigma=None, rho=None, beta=None):
        self.sigma = sigma
        self.rho = rho
        self.beta = beta

    def __call__(self, data):
        x = data[0]
        y = data[1]
        z = data[2]

        return np.array(
            [[self.sigma, self.sigma, 0], [-z + self.rho, -1, -x], [y, x, -self.beta]]
        )


def loss(approximated_data, target_data, fun, lambd_1=0, lambd_2=0) -> np.ndarray:
    loss = (
        np.mean(np.square(approximated_data - target_data))
        + lambd_1 * fun.A_op.weights_l2
        + lambd_2 * fun.H_op.weights_l2
    )

    sys.stdout.write(("\rloss: {}").format(loss))
    sys.stdout.flush()

    return loss


def test_opinf_nonlinear_int_lsoda():
    dt = 0.005
    T_max = 100
    rho = 28
    beta = 8 / 3
    beta_str = "8/3"
    sigma = 10
    n_field = 3
    train_fraction = 0.8
    lr = 1e-3
    n_epochs = 20_000
    second_optimizer = True

    initial_state = np.array([1, 0, 0])[None, :]

    lorenz_data, derivative_lorenz_data, time = lorenz_solver(
        rho=rho,
        dt=dt,
        T=T_max,
        sigma=sigma,
        initial_state=initial_state,
        beta=beta,
        beta_str=beta_str,
        data_path="/tmp",
        solver="RK45",
    )

    t = time
    n_steps = time.shape[0]
    nt = int(train_fraction * n_steps)
    t_test = t[nt:]

    train_field = lorenz_data[:nt]  # manufactured nonlinear oscillator data
    train_field_derivative = derivative_lorenz_data[:nt]

    test_field = lorenz_data[nt:]  # manufactured nonlinear oscillator data
    test_field_derivatives = derivative_lorenz_data[nt:]

    # Maximum derivative magnitudes to be used as loss weights
    maximum_values = (1 / np.linalg.norm(train_field_derivative, 2, axis=0)).tolist()

    params = {"lambda_1": 0.0, "lambda_2": (1e-3, 1e-3), "weights": maximum_values}

    lorenz_op = OpInfNetwork(n_inputs=n_field, name="lorenz")
    optimizer_config = {"lr": lr}
    optimizer = Optimizer("adam", params=optimizer_config)

    input_data = {"input_field": train_field}

    optimizer.fit(
        op=lorenz_op,
        input_data=input_data,
        target_data=train_field_derivative,
        n_epochs=n_epochs,
        loss="wrmse",
        params=params,
        batch_size=5000,
    )

    print(f"A_hat: {np.array_str(lorenz_op.A_hat, precision=2, suppress_small=True)}")
    print(f"H_hat: {np.array_str(lorenz_op.H_hat, precision=2, suppress_small=True)}")
    print(f"c_hat: {np.array_str(lorenz_op.c_hat, precision=2, suppress_small=True)}")

    init_state = train_field[-1:]

    lorenz_op.construct_K_op()
    lorenz_op_np = lorenz_op.to_numpy()
    estimated_field_derivatives = lorenz_op_np.eval(input_field=test_field)

    tags = ["x", "y", "z"]

    for var in range(n_field):
        plt.title(f"Time-derivative for variable {tags[var]}")
        plt.plot(t_test, test_field_derivatives[:, var], label="Exact")
        plt.plot(t_test, estimated_field_derivatives[:, var], label="Approximated")
        plt.xlabel("time (s)")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join("/tmp", f"derivatives_var_{var}.png"))
        plt.show()

    # Using the derivatives surrogate for time-integrating
    right_operator = ClassWrapper(lorenz_op)

    solver = LSODA(right_operator)

    initial_state = init_state[0]

    estimated_field = solver.run(initial_state, t_test)

    for var in range(n_field):
        plt.title(f"Variable {tags[var]}")
        plt.plot(t_test, test_field[:, var], label="Exact")
        plt.plot(t_test, estimated_field[:, var], label="Approximated")
        plt.xlabel("time (s)")
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join("/tmp", f"integrated_var_{var}.png"))
        plt.show()

    if second_optimizer is True:
        lorenz_op_np.construct_K_op()

        optimizer_scipy = ScipyInterface(
            fun=lorenz_op_np,
            optimizer="L-BFGS-B",
            loss=loss,
            loss_config={"lambd_1": 1e-3, "lambd_2": 1e-3},
            optimizer_config={"jac": "3-point"},
        )

        optimizer_scipy.fit(input_data=input_data, target_data=train_field_derivative)

        print(
            f"A_hat: {np.array_str(lorenz_op_np.A_hat, precision=2, suppress_small=True)}"
        )
        print(
            f"H_hat: {np.array_str(lorenz_op_np.H_hat, precision=2, suppress_small=True)}"
        )
        print(
            f"c_hat: {np.array_str(lorenz_op_np.c_hat, precision=2, suppress_small=True)}"
        )

        estimated_field_derivatives = lorenz_op_np.eval(input_field=test_field)

        tags = ["x", "y", "z"]

        for var in range(n_field):
            plt.title(f"Time-derivative for variable {tags[var]}")
            plt.plot(t_test, test_field_derivatives[:, var], label="Exact")
            plt.plot(t_test, estimated_field_derivatives[:, var], label="Approximated")
            plt.xlabel("time (s)")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join("/tmp", f"derivatives_var_{var}.png"))
            plt.show()

        # Using the derivatives surrogate for time-integrating
        right_operator = ClassWrapper(lorenz_op_np)

        solver = LSODA(right_operator)

        initial_state = init_state[0]

        estimated_field = solver.run(initial_state, t_test)

        for var in range(n_field):
            plt.title(f"Variable {tags[var]}")
            plt.plot(t_test, test_field[:, var], label="Exact")
            plt.plot(t_test, estimated_field[:, var], label="Approximated")
            plt.xlabel("time (s)")
            plt.grid(True)
            plt.legend()
            plt.savefig(os.path.join("/tmp", f"integrated_var_{var}.png"))
            plt.show()

        lorenz_op_np.save(save_path="/tmp", model_name="lorenz_op_numpy")


def test_koopman_nonlinear_int_lsoda():
    dt = 0.005
    T_max = 100
    rho = 28
    beta = 8 / 3
    beta_str = "8/3"
    sigma = 10
    n_field = 3
    train_fraction = 0.8
    lr = 1e-3
    n_epochs = 2000
    second_optimizer = False  # True

    initial_state = np.array([1, 0, 0])[None, :]

    lorenz_data, derivative_lorenz_data, time = lorenz_solver(
        rho=rho,
        dt=dt,
        T=T_max,
        sigma=sigma,
        initial_state=initial_state,
        beta=beta,
        beta_str=beta_str,
        data_path="/tmp",
        solver="RK45",
    )

    t = time
    n_steps = time.shape[0]
    nt = int(train_fraction * n_steps)
    t_test = t[nt:]

    train_field = lorenz_data[:nt]  # manufactured nonlinear oscillator data
    train_field_derivative = derivative_lorenz_data[:nt]

    test_field = lorenz_data[nt:]  # manufactured nonlinear oscillator data
    test_field_derivatives = derivative_lorenz_data[nt:]

    # Maximum derivative magnitudes to be used as loss weights
    maximum_values = (1 / np.linalg.norm(train_field_derivative, 2, axis=0)).tolist()

    params = {"lambda_1": 0.0, "lambda_2": (1e-3, 1e-3), "weights": maximum_values}

    operator_config = {"name": "lorenz", "setup_architecture": False}

    lorenz_op = KoopmanNetwork(
        n_inputs=n_field,
        observables=["x", "x**2", "tanh(x)"],
        operator_config=operator_config,
    )
    optimizer_config = {"lr": lr}
    optimizer = Optimizer("adam", params=optimizer_config)

    input_data = {"input_field": train_field}

    optimizer.fit(
        op=lorenz_op,
        input_data=input_data,
        target_data=train_field_derivative,
        n_epochs=n_epochs,
        loss="wrmse",
        params=params,
        batch_size=5000,
    )

    print(f"A_hat: {np.array_str(lorenz_op.A_hat, precision=2, suppress_small=True)}")
    print(f"H_hat: {np.array_str(lorenz_op.H_hat, precision=2, suppress_small=True)}")
    print(f"c_hat: {np.array_str(lorenz_op.c_hat, precision=2, suppress_small=True)}")

    init_state = train_field[-1:]

    lorenz_op.construct_K_op()

    estimated_field_derivatives = lorenz_op.eval(input_data=test_field)

    tags = ["x", "y", "z"]

    for var in range(n_field):
        plt.title(f"Time-derivative for variable {tags[var]}")
        plt.plot(t_test, test_field_derivatives[:, var], label="Exact")
        plt.plot(t_test, estimated_field_derivatives[:, var], label="Approximated")
        plt.xlabel("time (s)")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join("/tmp", f"derivatives_var_{var}.png"))
        plt.show()

    # Using the derivatives surrogate for time-integrating
    right_operator = ClassWrapper(lorenz_op)

    solver = LSODA(right_operator)

    initial_state = init_state[0]

    estimated_field = solver.run(initial_state, t_test)

    for var in range(n_field):
        plt.title(f"Variable {tags[var]}")
        plt.plot(t_test, test_field[:, var], label="Exact")
        plt.plot(t_test, estimated_field[:, var], label="Approximated")
        plt.xlabel("time (s)")
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join("/tmp", f"integrated_var_{var}.png"))
        plt.show()

    lorenz_op_np = lorenz_op.to_numpy()
    if second_optimizer is True:
        lorenz_op_np.construct_K_op()

        optimizer_scipy = ScipyInterface(
            fun=lorenz_op_np,
            optimizer="L-BFGS-B",
            loss=loss,
            loss_config={"lambd_1": 1e-3, "lambd_2": 1e-3},
            optimizer_config={"jac": "3-point"},
        )

        optimizer_scipy.fit(input_data=input_data, target_data=train_field_derivative)

        print(
            f"A_hat: {np.array_str(lorenz_op_np.A_hat, precision=2, suppress_small=True)}"
        )
        print(
            f"H_hat: {np.array_str(lorenz_op_np.H_hat, precision=2, suppress_small=True)}"
        )
        print(
            f"c_hat: {np.array_str(lorenz_op_np.c_hat, precision=2, suppress_small=True)}"
        )

        estimated_field_derivatives = lorenz_op_np.eval(input_field=test_field)

        tags = ["x", "y", "z"]

        for var in range(n_field):
            plt.title(f"Time-derivative for variable {tags[var]}")
            plt.plot(t_test, test_field_derivatives[:, var], label="Exact")
            plt.plot(t_test, estimated_field_derivatives[:, var], label="Approximated")
            plt.xlabel("time (s)")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join("/tmp", f"derivatives_var_{var}.png"))
            plt.show()

        # Using the derivatives surrogate for time-integrating
        right_operator = ClassWrapper(lorenz_op_np)

        solver = LSODA(right_operator)

        initial_state = init_state[0]

        estimated_field = solver.run(initial_state, t_test)

        for var in range(n_field):
            plt.title(f"Variable {tags[var]}")
            plt.plot(t_test, test_field[:, var], label="Exact")
            plt.plot(t_test, estimated_field[:, var], label="Approximated")
            plt.xlabel("time (s)")
            plt.grid(True)
            plt.legend()
            plt.savefig(os.path.join("/tmp", f"integrated_var_{var}.png"))
            plt.show()

        lorenz_op_np.save(save_path="/tmp", model_name="lorenz_op_numpy")


def test_koopman_autoencoder_nonlinear_int_lsoda():
    dt = 0.005
    T_max = 100
    rho = 28
    beta = 8 / 3
    beta_str = "8/3"
    sigma = 10
    n_field = 3
    train_fraction = 0.8
    second_optimizer = False

    initial_state = np.array([1, 0, 0])[None, :]

    lorenz_data, derivative_lorenz_data, time = lorenz_solver(
        rho=rho,
        dt=dt,
        T=T_max,
        sigma=sigma,
        initial_state=initial_state,
        beta=beta,
        beta_str=beta_str,
        data_path="/tmp",
        solver="RK45",
    )

    t = time
    n_steps = time.shape[0]
    nt = int(train_fraction * n_steps)
    t_test = t[nt:]

    train_field = lorenz_data[:nt]  # manufactured nonlinear oscillator data
    train_field_derivative = derivative_lorenz_data[:nt]

    test_field = lorenz_data[nt:]  # manufactured nonlinear oscillator data
    test_field_derivatives = derivative_lorenz_data[nt:]

    ##############################################################################
    # Configuring sub-networks
    ##############################################################################

    n_epochs = int(2e3)  # Maximum number of iterations for ADAM
    lr = 1e-3  # Initial learning rate for the ADAM algorithm
    n_modes = 3
    latent_dim = 20

    # Configuration for the encoder fully-connected network
    encoder_config = {
        "layers_units": [6, 10, 20],  # Hidden layers
        "activations": "elu",
        "input_size": n_modes,
        "output_size": latent_dim,
        "name": "encoder_net",
    }

    # Configuration for the decoder fully-connected network
    decoder_config = {
        "layers_units": [20, 10, 6],  # Hidden layers
        "activations": "elu",
        "input_size": latent_dim,
        "output_size": n_modes,
        "name": "encoder_net",
    }

    # Maximum derivative magnitudes to be used as loss weights
    maximum_values = (1 / np.linalg.norm(train_field_derivative, 2, axis=0)).tolist()
    params = {
        "lambda_1": 0.0,
        "lambda_2": (1e-5, 1e-3, 1e-3, 1e-5),
        "weights": maximum_values,
    }

    encoder_net = DenseNetwork(**encoder_config)
    decoder_net = DenseNetwork(**decoder_config)

    encoder_net.summary()
    decoder_net.summary()

    opinf_net = OpInfNetwork(n_inputs=latent_dim, name="ks_opinf_net")

    lorenz_net = AutoEncoderKoopman(
        encoder=encoder_net, decoder=decoder_net, opinf_net=opinf_net
    )

    optimizer_config = {"lr": lr}
    optimizer = Optimizer("adam", params=optimizer_config)

    optimizer.fit(
        op=lorenz_net,
        input_data=train_field,
        target_data=train_field_derivative,
        n_epochs=n_epochs,
        loss="wrmse",
        params=params,
    )

    print(
        f"A_hat: {np.array_str(lorenz_net.opinf_net.A_hat, precision=2, suppress_small=True)}"
    )
    print(
        f"H_hat: {np.array_str(lorenz_net.opinf_net.H_hat, precision=2, suppress_small=True)}"
    )
    print(
        f"c_hat: {np.array_str(lorenz_net.opinf_net.c_hat, precision=2, suppress_small=True)}"
    )

    init_state = train_field[-1:]

    estimated_field_derivatives = lorenz_net.eval(input_data=test_field)

    tags = ["x", "y", "z"]

    for var in range(n_field):
        plt.title(f"Time-derivative for variable {tags[var]}")
        plt.plot(t_test, test_field_derivatives[:, var], label="Exact")
        plt.plot(t_test, estimated_field_derivatives[:, var], label="Approximated")
        plt.xlabel("time (s)")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join("/tmp", f"derivatives_var_{var}.png"))
        plt.show()

    # Using the derivatives surrogate for time-integrating
    right_operator = ClassWrapper(lorenz_net)

    solver = LSODA(right_operator)

    initial_state = init_state[0]

    estimated_field = solver.run(initial_state, t_test)

    for var in range(n_field):
        plt.title(f"Variable {tags[var]}")
        plt.plot(t_test, test_field[:, var], label="Exact")
        plt.plot(t_test, estimated_field[:, var], label="Approximated")
        plt.xlabel("time (s)")
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join("/tmp", f"integrated_var_{var}.png"))
        plt.show()

    if second_optimizer is True:
        lorenz_net.construct_K_op()

        optimizer_scipy = ScipyInterface(
            fun=lorenz_net,
            optimizer="L-BFGS-B",
            loss=loss,
            loss_config={"lambd_1": 1e-3, "lambd_2": 1e-3},
            optimizer_config={"jac": "3-point"},
        )

        optimizer_scipy.fit(input_data=train_field, target_data=train_field_derivative)

        print(
            f"A_hat: {np.array_str(lorenz_net.A_hat, precision=2, suppress_small=True)}"
        )
        print(
            f"H_hat: {np.array_str(lorenz_net.H_hat, precision=2, suppress_small=True)}"
        )
        print(
            f"c_hat: {np.array_str(lorenz_net.c_hat, precision=2, suppress_small=True)}"
        )

        estimated_field_derivatives = lorenz_net.eval(input_data=test_field)

        tags = ["x", "y", "z"]

        for var in range(n_field):
            plt.title(f"Time-derivative for variable {tags[var]}")
            plt.plot(t_test, test_field_derivatives[:, var], label="Exact")
            plt.plot(t_test, estimated_field_derivatives[:, var], label="Approximated")
            plt.xlabel("time (s)")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join("/tmp", f"derivatives_var_{var}.png"))
            plt.show()

        # Using the derivatives surrogate for time-integrating
        right_operator = ClassWrapper(lorenz_net)

        solver = LSODA(right_operator)

        initial_state = init_state[0]

        estimated_field = solver.run(initial_state, t_test)

        for var in range(n_field):
            plt.title(f"Variable {tags[var]}")
            plt.plot(t_test, test_field[:, var], label="Exact")
            plt.plot(t_test, estimated_field[:, var], label="Approximated")
            plt.xlabel("time (s)")
            plt.grid(True)
            plt.legend()
            plt.savefig(os.path.join("/tmp", f"integrated_var_{var}.png"))
            plt.show()

        lorenz_net.save(save_path="/tmp", model_name="lorenz_op_numpy")
