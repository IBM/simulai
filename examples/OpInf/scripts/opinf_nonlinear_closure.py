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

import matplotlib.pyplot as plt
import numpy as np

os.environ["engine"] = "pytorch"

from examples.utils.lorenz_solver import lorenz_solver
from simulai.math.integration import LSODA, ClassWrapper
from simulai.metrics import L2Norm
from simulai.optimization import Optimizer
from simulai.regression import DenseNetwork, OpInf


class TestModelPoolESN:
    def __init__(self):
        pass

    def test_opinf_closure_dense(self):
        dt = 0.005
        T_max = 100
        rho = 28
        beta = 8 / 3
        beta_str = "8/3"
        sigma = 10
        n_field = 3
        lambda_linear = 1e-3
        lambda_quadratic = 1e-3
        train_fraction = 0.8

        initial_state = np.array([1, 0, 0])[None, :]

        lorenz_data, derivative_lorenz_data, time = lorenz_solver(
            rho=rho,
            dt=dt,
            T=T_max,
            sigma=sigma,
            initial_state=initial_state,
            beta=beta,
            beta_str=beta_str,
            data_path="on_memory",
            solver="RK45",
        )

        t = time
        n_steps = time.shape[0]
        nt = int(train_fraction * n_steps)
        nt_test = n_steps - nt
        t_test = t[nt:]

        train_field = lorenz_data[:nt]  # manufactured nonlinear oscillator data
        train_field_derivative = derivative_lorenz_data[:nt]

        test_field = lorenz_data[nt:]  # manufactured nonlinear oscillator data
        test_field_derivatives = derivative_lorenz_data[nt:]

        lorenz_op = OpInf(bias_rescale=1e-15, solver="lstsq")
        lorenz_op.set(lambda_linear=lambda_linear, lambda_quadratic=lambda_quadratic)
        lorenz_op.fit(input_data=train_field, target_data=train_field_derivative)

        print(
            f"A_hat: {np.array_str(lorenz_op.A_hat, precision=3, suppress_small=True)}"
        )
        print(
            f"H_hat: {np.array_str(lorenz_op.H_hat, precision=3, suppress_small=True)}"
        )
        print(
            f"c_hat: {np.array_str(lorenz_op.c_hat, precision=3, suppress_small=True)}"
        )

        init_state = train_field[-1:]

        estimated_field_derivatives = lorenz_op.eval(input_data=test_field)

        tags = ["x", "y", "z"]

        for var in range(n_field):
            plt.title(f"Time-derivative for variable {tags[var]}")
            plt.plot(t_test, test_field_derivatives[:, var], label="Exact")
            plt.plot(t_test, estimated_field_derivatives[:, var], label="Approximated")
            plt.xlabel("time (s)")
            plt.legend()
            plt.grid(True)
            plt.savefig(f"derivatives_var_{var}.png")
            plt.close()

        lorenz_op.construct_K_op()

        # Using the derivatives surrogate for time-integrating
        right_operator = ClassWrapper(lorenz_op)

        solver = LSODA(right_operator)

        initial_state = init_state[0]

        estimated_field = solver.run(initial_state, t_test)

        l2_norm = L2Norm()
        error = 100 * l2_norm(
            data=estimated_field, reference_data=test_field, relative_norm=True
        )
        print(f"Evaluation error: {error} %")

        for var in range(n_field):
            plt.title(f"Variable {tags[var]}")
            plt.plot(t_test, test_field[:, var], label="Exact")
            plt.plot(t_test, estimated_field[:, var], label="Approximated")
            plt.xlabel("time (s)")
            plt.grid(True)
            plt.legend()
            plt.savefig(f"integrated_var_{var}.png")
            plt.close()

        for var in range(n_field):
            plt.title(f"Pointwise error {tags[var]}")
            plt.plot(t_test, np.abs(test_field[:, var] - estimated_field[:, var]))
            plt.xlabel("time (s)")
            plt.grid(True)
            plt.legend()
            plt.savefig(f"pointwise_error_integrated_var_{var}.png")
            plt.close()

        residual_derivatives = derivative_lorenz_data - lorenz_op.eval(
            input_data=lorenz_data
        )

        train_field = lorenz_data[:nt]  # manufactured nonlinear oscillator data
        train_field_residual_derivative = residual_derivatives[:nt]

        test_field = lorenz_data[nt:]  # manufactured nonlinear oscillator data
        test_field_residual_derivative = residual_derivatives[nt:]

        # Training closure
        lambda_1 = 0.0  # Penalty for the L¹ regularization (Lasso)
        lambda_2 = 1e-4  # Penalty factor for the L² regularization
        n_epochs = int(1e4)  # Maximum number of iterations for ADAM
        lr = 1e-3  # Initial learning rate for the ADAM algorithm

        # Configuration for the fully-connected network
        config = {
            "layers_units": [100, 100],  # Hidden layers
            "activations": "tanh",
            "input_size": n_field,
            "output_size": n_field,
            "name": "lorenz_net",
        }

        optimizer_config = {"lr": lr}

        # Instantiating and training the surrogate model
        lorenz_net = DenseNetwork(**config)

        # It prints a summary of the network features
        lorenz_net.summary()

        maximum_values = (
            1 / np.linalg.norm(train_field_residual_derivative, 2, axis=0)
        ).tolist()
        params = {
            "lambda_1": lambda_1,
            "lambda_2": lambda_2,
        }  # , 'weights': maximum_values}

        optimizer = Optimizer("adam", params=optimizer_config)

        optimizer.fit(
            op=lorenz_net,
            input_data=train_field,
            target_data=train_field_residual_derivative,
            n_epochs=n_epochs,
            loss="rmse",
            params=params,
        )

        approximated_residual_data = lorenz_net.eval(input_data=test_field)

        for var in range(n_field):
            plt.title(
                f"Comparison between aapproximated and ground truth residuals {tags[var]}"
            )
            plt.plot(t_test, test_field_residual_derivative[:, var])
            plt.plot(t_test, approximated_residual_data[:, var])
            plt.xlabel("time (s)")
            plt.grid(True)
            plt.legend()
            plt.savefig(f"residual_derivative_approximation{var}.png")
            plt.show()

    def test_opinf_closure_opinf(self):
        dt = 0.005
        T_max = 100
        rho = 28
        beta = 8 / 3
        beta_str = "8/3"
        sigma = 10
        n_field = 3
        lambda_linear = 1e-3
        lambda_quadratic = 1e-3
        train_fraction = 0.8

        initial_state = np.array([1, 0, 0])[None, :]

        lorenz_data, derivative_lorenz_data, time = lorenz_solver(
            rho=rho,
            dt=dt,
            T=T_max,
            sigma=sigma,
            initial_state=initial_state,
            beta=beta,
            beta_str=beta_str,
            data_path="on_memory",
            solver="RK45",
        )

        t = time
        n_steps = time.shape[0]
        nt = int(train_fraction * n_steps)
        nt_test = n_steps - nt
        t_test = t[nt:]

        train_field = lorenz_data[:nt]  # manufactured nonlinear oscillator data
        train_field_derivative = derivative_lorenz_data[:nt]

        test_field = lorenz_data[nt:]  # manufactured nonlinear oscillator data
        test_field_derivatives = derivative_lorenz_data[nt:]

        lorenz_op = OpInf(bias_rescale=1e-15, solver="lstsq")
        lorenz_op.set(lambda_linear=lambda_linear, lambda_quadratic=lambda_quadratic)
        lorenz_op.fit(input_data=train_field, target_data=train_field_derivative)

        print(
            f"A_hat: {np.array_str(lorenz_op.A_hat, precision=3, suppress_small=True)}"
        )
        print(
            f"H_hat: {np.array_str(lorenz_op.H_hat, precision=3, suppress_small=True)}"
        )
        print(
            f"c_hat: {np.array_str(lorenz_op.c_hat, precision=3, suppress_small=True)}"
        )

        init_state = train_field[-1:]

        estimated_field_derivatives = lorenz_op.eval(input_data=test_field)

        tags = ["x", "y", "z"]

        for var in range(n_field):
            plt.title(f"Time-derivative for variable {tags[var]}")
            plt.plot(t_test, test_field_derivatives[:, var], label="Exact")
            plt.plot(t_test, estimated_field_derivatives[:, var], label="Approximated")
            plt.xlabel("time (s)")
            plt.legend()
            plt.grid(True)
            plt.savefig(f"derivatives_var_{var}.png")
            plt.close()

        lorenz_op.construct_K_op()

        # Using the derivatives surrogate for time-integrating
        right_operator = ClassWrapper(lorenz_op)

        solver = LSODA(right_operator)

        initial_state = init_state[0]

        estimated_field = solver.run(initial_state, t_test)

        l2_norm = L2Norm()
        error = 100 * l2_norm(
            data=estimated_field, reference_data=test_field, relative_norm=True
        )
        print(f"Evaluation error: {error} %")

        for var in range(n_field):
            plt.title(f"Variable {tags[var]}")
            plt.plot(t_test, test_field[:, var], label="Exact")
            plt.plot(t_test, estimated_field[:, var], label="Approximated")
            plt.xlabel("time (s)")
            plt.grid(True)
            plt.legend()
            plt.savefig(f"integrated_var_{var}.png")
            plt.close()

        for var in range(n_field):
            plt.title(f"Pointwise error {tags[var]}")
            plt.plot(t_test, np.abs(test_field[:, var] - estimated_field[:, var]))
            plt.xlabel("time (s)")
            plt.grid(True)
            plt.legend()
            plt.savefig(f"pointwise_error_integrated_var_{var}.png")
            plt.close()

        residual_derivatives = derivative_lorenz_data - lorenz_op.eval(
            input_data=lorenz_data
        )

        train_field = lorenz_data[:nt]  # manufactured nonlinear oscillator data
        train_field_residual_derivative = residual_derivatives[:nt]

        test_field = lorenz_data[nt:]  # manufactured nonlinear oscillator data
        test_field_residual_derivative = residual_derivatives[nt:]

        # Training closure
        lorenz_residual_op = OpInf(bias_rescale=1)
        lorenz_residual_op.set(
            lambda_linear=1e-3 * lambda_linear, lambda_quadratic=1e-3 * lambda_quadratic
        )
        lorenz_residual_op.fit(
            input_data=train_field, target_data=train_field_residual_derivative
        )

        print(
            f"A_hat: {np.array_str(lorenz_residual_op.A_hat, precision=3, suppress_small=True)}"
        )
        print(
            f"H_hat: {np.array_str(lorenz_residual_op.H_hat, precision=3, suppress_small=True)}"
        )
        print(
            f"c_hat: {np.array_str(lorenz_residual_op.c_hat, precision=3, suppress_small=True)}"
        )

        approximated_residual_data = lorenz_residual_op.eval(input_data=test_field)

        for var in range(n_field):
            plt.title(
                f"Comparison between aapproximated and ground truth residuals {tags[var]}"
            )
            plt.plot(t_test, test_field_residual_derivative[:, var])
            plt.plot(t_test, approximated_residual_data[:, var])
            plt.xlabel("time (s)")
            plt.grid(True)
            plt.legend()
            plt.savefig(f"residual_derivative_approximation{var}.png")
            plt.show()
