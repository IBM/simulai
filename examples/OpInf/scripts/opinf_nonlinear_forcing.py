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

os.environ["engine"] = "pytorch"

import logging

from examples.utils.lorenz_solver import lorenz_solver, lorenz_solver_forcing
from examples.utils.oscillator_solver import oscillator_solver_forcing
from simulai.math.integration import LSODA, RK4, ClassWrapper, FunctionWrapper
from simulai.metrics import L2Norm, LyapunovUnits
from simulai.regression import ExtendedOpInf, OpInf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestOpInfNonlinear:
    def __init__(self):
        pass

    def test_opinf_nonlinear(self):
        dt = 0.001
        T_max = 20
        rho = 28
        beta = 8 / 3
        beta_str = "8/3"
        sigma = 10
        n_field = 3
        t = np.arange(0.8 * T_max, T_max, dt)

        initial_state = np.array([1, 0, 0])[None, :]
        lorenz_data, derivative_lorenz_data = lorenz_solver(
            rho=rho,
            dt=dt,
            T=T_max,
            sigma=sigma,
            initial_state=initial_state,
            beta=beta,
            beta_str=beta_str,
            data_path="on_memory",
        )

        lambda_linear = 1e-3
        lambda_quadratic = 1e-3
        n_steps = int(T_max / dt)
        nt = int(0.8 * n_steps)
        nt_test = n_steps - nt

        train_field = lorenz_data[:nt]  # manufactured nonlinear oscillator data
        train_field_derivative = derivative_lorenz_data[:nt]

        test_field = lorenz_data[nt:]  # manufactured nonlinear oscillator data
        test_field_derivatives = derivative_lorenz_data[nt:]

        lorenz_op = OpInf(bias_rescale=1e-15)
        lorenz_op.set(lambda_linear=lambda_linear, lambda_quadratic=lambda_quadratic)
        lorenz_op.fit(input_data=train_field, target_data=train_field_derivative)

        logging.info(
            f"A_hat: {np.array_str(lorenz_op.A_hat, precision=2, suppress_small=True)}"
        )
        logging.info(
            f"H_hat: {np.array_str(lorenz_op.H_hat, precision=2, suppress_small=True)}"
        )
        logging.info(
            f"c_hat: {np.array_str(lorenz_op.c_hat, precision=2, suppress_small=True)}"
        )

        init_state = train_field[-1:]

        estimated_field_derivatives = lorenz_op.eval(input_data=test_field)

        tags = ["x", "y", "z"]

        for var in range(n_field):
            plt.title(f"Time-derivative for variable {tags[var]}")
            plt.plot(t, test_field_derivatives[:, var], label="Exact")
            plt.plot(t, estimated_field_derivatives[:, var], label="Approximated")
            plt.xlabel("time (s)")
            plt.legend()
            plt.grid(True)
            plt.savefig(f"derivatives_var_{var}.png")
            plt.show()

        # Using the derivatives surrogate for time-integrating
        right_operator = FunctionWrapper(lorenz_op.eval, extra_dim=False)
        lorenz_op.construct_K_op()
        jacobian_estimative = lorenz_op.jacobian(np.array([1, 1, 1])[None, :])

        solver = RK4(right_operator)

        time = 0.80 * T_max
        estimated_variables = list()
        initial_state = init_state
        N_steps = int(T_max / dt)
        n_steps = test_field.shape[0]
        interval = int(N_steps / n_steps)

        ii = 0
        # Approach based on Lui & Wolf (https://arxiv.org/abs/1903.05206)
        while time <= T_max:
            state, derivative_state = solver.step(initial_state, dt)
            estimated_variables.append(state)
            initial_state = state
            sys.stdout.write("\rIteration {}".format(ii))
            sys.stdout.flush()
            time += dt
            ii += 1

        estimated_field = np.vstack(estimated_variables)

        lyapunov_estimator = LyapunovUnits(lyapunov_unit=0.96, tol=0.10, time_scale=dt)
        n_units = lyapunov_estimator(
            data=estimated_field[:], reference_data=test_field, relative_norm=True
        )
        print(f"Number of Lyapunov units extrapolated: {n_units}")

        for var in range(n_field):
            plt.title(f"Variable {tags[var]}")
            plt.plot(t, test_field[:, var], label="Exact")
            plt.plot(t, estimated_field[:, var], label="Approximated")
            plt.xlabel("time (s)")
            plt.grid(True)
            plt.legend()
            plt.savefig(f"integrated_var_{var}.png")
            plt.show()

    def test_opinf_nonlinear_int_lsoda(self):
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

        lorenz_op = OpInf(bias_rescale=1, solver="pinv_close")
        lorenz_op.set(lambda_linear=lambda_linear, lambda_quadratic=lambda_quadratic)
        lorenz_op.fit(
            input_data=train_field,
            target_data=train_field_derivative,
            batch_size=1000,
            force_lazy_access=True,
            k_svd=9,
        )

        logger.info(
            f"A_hat: {np.array_str(lorenz_op.A_hat, precision=3, suppress_small=True)}"
        )
        logger.info(
            f"H_hat: {np.array_str(lorenz_op.H_hat, precision=3, suppress_small=True)}"
        )
        logger.info(
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
            plt.show()

        lorenz_op.construct_K_op()

        jacobian_estimative = lorenz_op.jacobian(np.array([1, 1, 1]))

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
            plt.show()

        for var in range(n_field):
            plt.title(f"Pointwise error {tags[var]}")
            plt.plot(t_test, np.abs(test_field[:, var] - estimated_field[:, var]))
            plt.xlabel("time (s)")
            plt.grid(True)
            plt.legend()
            plt.savefig(f"pointwise_error_integrated_var_{var}.png")
            plt.show()

        lorenz_op.save(save_path="/tmp", model_name="lorenz_63_koopman")
        lorenz_op.lean_save(save_path="/tmp", model_name="lorenz_63_koopman_lean")

    def test_koopman_nonlinear_int_lsoda(self):
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

        operator_config = {"bias_rescale": 1e-15, "solver": "pinv"}

        # lorenz_op = ExtendedOpInf(observables=['x'], operator_config=operator_config)
        # lorenz_op = ExtendedOpInf(observables=['x', 'sin(4*x)', 'cos(4*x)'], operator_config=operator_config)
        lorenz_op = ExtendedOpInf(
            observables=["x", "x**2", "tanh(x)"], operator_config=operator_config
        )
        # lorenz_op = ExtendedOpInf(observables=['x', 'Kronecker(x)'], operator_config=operator_config)
        # lorenz_op = ExtendedOpInf(observables=['x', 'Kronecker(x)', 'x**3'], operator_config=operator_config)
        # lorenz_op = ExtendedOpInf(observables=['x', 'Kronecker(Kronecker(x))'], operator_config=operator_config)
        # lorenz_op = ExtendedOpInf(observables=['x', 'tanh(sin(Kronecker(x)))'], operator_config=operator_config)

        lorenz_op.set(lambda_linear=lambda_linear, lambda_quadratic=lambda_quadratic)
        lorenz_op.fit(
            input_data=train_field,
            target_data=train_field_derivative,
            batch_size=10000,
            force_lazy_access=False,
            k_svd=51,
        )

        print(
            f"A_hat: {np.array_str(lorenz_op.A_hat, precision=2, suppress_small=True)}"
        )
        print(
            f"H_hat: {np.array_str(lorenz_op.H_hat, precision=2, suppress_small=True)}"
        )
        print(
            f"c_hat: {np.array_str(lorenz_op.c_hat, precision=2, suppress_small=True)}"
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
            plt.show()

        lorenz_op.construct_K_op()
        jacobian_estimative = lorenz_op.jacobian(np.array([1, 1, 1]))

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
            plt.show()

        for var in range(n_field):
            plt.title(f"Pointwise error {tags[var]}")
            plt.plot(t_test, np.abs(test_field[:, var] - estimated_field[:, var]))
            plt.xlabel("time (s)")
            plt.grid(True)
            plt.legend()
            plt.savefig(f"pointwise_error_integrated_var_{var}.png")
            plt.show()

        lorenz_op.save(save_path="/tmp", model_name="lorenz_63_koopman")
        lorenz_op.lean_save(save_path="/tmp", model_name="lorenz_63_koopman_lean")

    def test_koopman_nonlinear_int_lsoda_intervaled(self):
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

        operator_config = {"bias_rescale": 1e-15}

        lorenz_op = ExtendedOpInf(
            observables=["x", "x**2", "tanh(x)"],
            intervals=[-1, -1, -1],
            operator_config=operator_config,
        )

        lorenz_op.set(lambda_linear=lambda_linear, lambda_quadratic=lambda_quadratic)
        lorenz_op.fit(input_data=train_field, target_data=train_field_derivative)

        print(
            f"A_hat: {np.array_str(lorenz_op.A_hat, precision=2, suppress_small=True)}"
        )
        print(
            f"H_hat: {np.array_str(lorenz_op.H_hat, precision=2, suppress_small=True)}"
        )
        print(
            f"c_hat: {np.array_str(lorenz_op.c_hat, precision=2, suppress_small=True)}"
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
            plt.show()

        lorenz_op.construct_K_op()
        jacobian_estimative = lorenz_op.jacobian(np.array([1, 1, 1]))

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
            plt.show()

        for var in range(n_field):
            plt.title(f"Pointwise error {tags[var]}")
            plt.plot(t_test, np.abs(test_field[:, var] - estimated_field[:, var]))
            plt.xlabel("time (s)")
            plt.grid(True)
            plt.legend()
            plt.savefig(f"pointwise_error_integrated_var_{var}.png")
            plt.show()

    def test_opinf_nonlinear_incremental(self):
        dt = 0.0005
        T_max = 50
        rho = 28
        beta = 8 / 3
        beta_str = "8/3"
        sigma = 10
        n_field = 3
        t = np.arange(0.9 * T_max, T_max, dt)

        initial_state = np.array([1, 2, 3])[None, :]
        lorenz_data, derivative_lorenz_data = lorenz_solver(
            rho=rho,
            dt=dt,
            T=T_max,
            sigma=sigma,
            initial_state=initial_state,
            beta=beta,
            beta_str=beta_str,
            data_path="on_memory",
        )

        lambda_linear = 10
        lambda_quadratic = 100
        n_steps = int(T_max / dt)
        nt = int(0.9 * n_steps)
        nt_test = n_steps - nt

        train_field = lorenz_data[:nt]  # manufactured nonlinear oscillator data
        train_field_derivative = derivative_lorenz_data[:nt]

        test_field = lorenz_data[nt:]  # manufactured nonlinear oscillator data
        test_field_derivatives = derivative_lorenz_data[nt:]

        lorenz_op = OpInf(bias_rescale=1e-15)
        lorenz_op.set(lambda_linear=lambda_linear, lambda_quadratic=lambda_quadratic)
        lorenz_op.fit(
            input_data=train_field, target_data=train_field_derivative, batch_size=10000
        )

        print(
            f"A_hat: {np.array_str(lorenz_op.A_hat, precision=2, suppress_small=True)}"
        )
        print(
            f"H_hat: {np.array_str(lorenz_op.H_hat, precision=2, suppress_small=True)}"
        )
        print(
            f"c_hat: {np.array_str(lorenz_op.c_hat, precision=2, suppress_small=True)}"
        )

        init_state = train_field[-1:]

        estimated_field_derivatives = lorenz_op.eval(input_data=test_field)

        tags = ["x", "y", "z"]

        for var in range(n_field):
            plt.title(f"Time-derivative for variable {tags[var]}")
            plt.plot(t, test_field_derivatives[:, var], label="Exact")
            plt.plot(t, estimated_field_derivatives[:, var], label="Approximated")
            plt.xlabel("time (s)")
            plt.legend()
            plt.grid(True)
            plt.savefig(f"derivatives_var_{var}.png")
            plt.show()

    def test_opinf_nonlinear_with_linear_forcing(self):
        dt = 0.0025
        T_max = 50
        rho = 28
        beta = 8 / 3
        beta_str = "8/3"
        sigma = 10
        A = 2
        t = np.arange(0.9 * T_max, T_max, dt)

        n_steps = int(T_max / dt)
        nt = int(0.9 * n_steps)
        nt_test = n_steps - nt

        forcings = A * np.random.rand(n_steps, 3)
        initial_state = np.array([1, 2, 3])[None, :]

        n_field = 3  # number of field values to predict
        lambda_linear = 10
        lambda_quadratic = 100

        lorenz_data, derivative_lorenz_data = lorenz_solver_forcing(
            rho=rho,
            dt=dt,
            T=T_max,
            sigma=sigma,
            initial_state=initial_state,
            forcing=forcings,
            beta=beta,
            beta_str=beta_str,
            data_path="on_memory",
        )

        train_field = lorenz_data[:nt]  # manufactured nonlinear oscillator data
        train_field_derivative = derivative_lorenz_data[:nt]
        train_forcings = forcings[:nt]

        test_field = lorenz_data[nt:]  # manufactured nonlinear oscillator data
        test_field_derivatives = derivative_lorenz_data[nt:]
        test_forcings = forcings[nt:]

        lorenz_op = OpInf(forcing="linear", bias_rescale=1e-15)
        lorenz_op.set(lambda_linear=lambda_linear, lambda_quadratic=lambda_quadratic)

        lorenz_op.fit(
            input_data=train_field,
            target_data=train_field_derivative,
            forcing_data=train_forcings,
        )

        logger.info(
            f"A_hat: {np.array_str(lorenz_op.A_hat, precision=2, suppress_small=True)}"
        )
        logger.info(
            f"H_hat: {np.array_str(lorenz_op.H_hat, precision=2, suppress_small=True)}"
        )
        logger.info(
            f"B_hat: {np.array_str(lorenz_op.B_hat, precision=2, suppress_small=True)}"
        )
        logger.info(
            f"c_hat: {np.array_str(lorenz_op.c_hat, precision=2, suppress_small=True)}"
        )

        init_state = train_field[-1:]

        estimated_field_derivatives = lorenz_op.eval(
            input_data=test_field, forcing_data=test_forcings
        )

        tags = ["x", "y", "z"]

        for var in range(n_field):
            plt.title(f"Time-derivative for variable {tags[var]}")
            plt.plot(t, test_field_derivatives[:, var], label="Exact")
            plt.plot(t, estimated_field_derivatives[:, var], label="Approximated")
            plt.xlabel("time (s)")
            plt.legend()
            plt.grid(True)
            plt.savefig(f"derivatives_var_{var}.png")
            plt.show()

        # Using the derivatives surrogate for time-integrating
        solver = RK4(lorenz_op.eval)

        time = 0.90 * T_max
        estimated_variables = list()
        initial_state = init_state
        N_steps = int(T_max / dt)
        n_steps = test_field.shape[0]

        ii = 0

        while time < T_max - dt:
            state, derivative_state = solver.step_with_forcings_separated(
                initial_state, test_forcings[ii : ii + 1], dt
            )
            estimated_variables.append(state)
            initial_state = state
            sys.stdout.write("\rIteration {}".format(ii))
            sys.stdout.flush()
            time += dt
            ii += 1

        estimated_field = np.vstack(estimated_variables)

        for var in range(n_field):
            plt.title(f"Variable {tags[var]}")
            plt.plot(t, test_field[:, var], label="Exact")
            plt.plot(t, estimated_field[:, var], label="Approximated")
            plt.xlabel("time (s)")
            plt.grid(True)
            plt.legend()
            plt.savefig(f"integrated_var_{var}.png")
            plt.show()

    def test_opinf_nonlinear_with_linear_forcing_lsoda(self):
        dt = 0.0025
        T_max = 50
        rho = 28
        beta = 8 / 3
        beta_str = "8/3"
        sigma = 10
        A = 0.1
        t = np.arange(0.9 * T_max, T_max, dt)

        n_steps = int(T_max / dt)
        nt = int(0.9 * n_steps)
        nt_test = n_steps - nt

        forcings = A * np.random.rand(n_steps, 3)
        initial_state = np.array([1, 2, 3])[None, :]

        n_field = 3  # number of field values to predict
        lambda_linear = 10
        lambda_quadratic = 100

        lorenz_data, derivative_lorenz_data = lorenz_solver_forcing(
            rho=rho,
            dt=dt,
            T=T_max,
            sigma=sigma,
            initial_state=initial_state,
            forcing=forcings,
            beta=beta,
            beta_str=beta_str,
            data_path="on_memory",
        )

        train_field = lorenz_data[:nt]  # manufactured nonlinear oscillator data
        train_field_derivative = derivative_lorenz_data[:nt]
        train_forcings = forcings[:nt]

        test_field = lorenz_data[nt:]  # manufactured nonlinear oscillator data
        test_field_derivatives = derivative_lorenz_data[nt:]
        test_forcings = forcings[nt:]

        lorenz_op = OpInf(forcing="linear", bias_rescale=1e-15)
        lorenz_op.set(lambda_linear=lambda_linear, lambda_quadratic=lambda_quadratic)

        lorenz_op.fit(
            input_data=train_field,
            target_data=train_field_derivative,
            forcing_data=train_forcings,
        )

        logger.info(
            f"A_hat: {np.array_str(lorenz_op.A_hat, precision=2, suppress_small=True)}"
        )
        logger.info(
            f"H_hat: {np.array_str(lorenz_op.H_hat, precision=2, suppress_small=True)}"
        )
        logger.info(
            f"B_hat: {np.array_str(lorenz_op.B_hat, precision=2, suppress_small=True)}"
        )
        logger.info(
            f"c_hat: {np.array_str(lorenz_op.c_hat, precision=2, suppress_small=True)}"
        )

        init_state = train_field[-1:]

        tags = ["x", "y", "z"]

        # Using the derivatives surrogate for time-integrating
        time = 0.90 * T_max
        estimated_variables = list()
        initial_state = init_state
        N_steps = int(T_max / dt)
        n_steps = test_field.shape[0]
        t_test = np.linspace(time, T_max, n_steps)

        # Using the derivatives surrogate for time-integrating
        right_operator = ClassWrapper(lorenz_op)

        solver = LSODA(right_operator)

        initial_state = init_state[0]

        estimated_field = solver.run_forcing(
            initial_state, t_test, forcing=test_forcings
        )

        for var in range(n_field):
            plt.title(f"Variable {tags[var]}")
            plt.plot(t_test, test_field[:, var], label="Exact")
            plt.plot(t_test, estimated_field[:, var], label="Approximated")
            plt.xlabel("time (s)")
            plt.grid(True)
            plt.legend()
            plt.savefig(f"integrated_var_{var}.png")
            plt.show()

    def test_basic_opinf_nonlinear_with_linear_forcing(self):
        n_steps = 100000
        A = 1
        T = 50
        dt = T / n_steps

        forcings = A * np.random.rand(n_steps, 2)
        initial_state = np.array([2, 0])[None, :]

        n_field = 2  # number of field values to predict
        n_forcing = 2  # number of forcing terms
        lambda_linear = 10
        lambda_quadratic = 1

        nt = int(0.9 * n_steps)
        nt_test = n_steps - nt

        oscillator_data, derivative_oscillator_data = oscillator_solver_forcing(
            T, dt, initial_state, forcing=forcings
        )

        # Choosing observables (simple Koopman operator)
        # g = (x^3, y^3)
        observables = np.power(oscillator_data, 3)

        train_field = observables[:nt]  # manufactured nonlinear oscillator data
        train_field_derivative = derivative_oscillator_data[:nt]
        train_forcings = forcings[:nt]

        test_field = observables[nt:]  # manufactured nonlinear oscillator data
        test_field_derivatives = derivative_oscillator_data[nt:]
        test_forcings = forcings[nt:]

        non_op = OpInf(forcing="linear")
        non_op.set(lambda_linear=lambda_linear, lambda_quadratic=lambda_quadratic)
        non_op.fit(
            input_data=train_field,
            target_data=train_field_derivative,
            forcing_data=train_forcings,
        )

        init_state = train_field[-1:]

        estimated_field_derivatives = non_op.eval(
            input_data=test_field, forcing_data=test_forcings
        )

        print(f"A_hat: {np.array_str(non_op.A_hat, precision=2, suppress_small=True)}")
        print(f"H_hat: {np.array_str(non_op.H_hat, precision=2, suppress_small=True)}")
        print(f"c_hat: {np.array_str(non_op.c_hat, precision=2, suppress_small=True)}")

        for var in range(n_field):
            plt.plot(test_field_derivatives[:, var])
            plt.plot(estimated_field_derivatives[:, var])

            plt.show()
