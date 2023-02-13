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

import matplotlib.pyplot as plt
import numpy as np

from examples.utils.lorenz_solver import lorenz_solver
from simulai.math.integration import LSODA, ClassWrapper
from simulai.regression import OpInf


def jacobian(sigma=None, rho=None, beta=None, x=None, y=None, z=None):
    return np.array([[-sigma, sigma, 0], [rho - z, -1, -x], [y, x, -beta]])


# Set up
dt = 0.0025
T_max = 100
rho = 28
beta = 8 / 3
beta_str = "8/3"
sigma = 10
n_field = 3
lambda_linear = 1e-3
lambda_quadratic = 1e-3
train_fraction = 0.8
n_procs = 4

# Generating datasets
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

# preparing datasets
train_field = lorenz_data[:nt]  # manufactured nonlinear oscillator data
train_field_derivative = derivative_lorenz_data[:nt]

test_field = lorenz_data[nt:]  # manufactured nonlinear oscillator data
test_field_derivatives = derivative_lorenz_data[nt:]

# Instantiating OpInf
parallel = None  #'mpi'
lorenz_op = OpInf(bias_rescale=1e-15, solver="lstsq", parallel=parallel)

# Training
lorenz_op.set(lambda_linear=lambda_linear, lambda_quadratic=lambda_quadratic)
lorenz_op.fit(
    input_data=train_field, target_data=train_field_derivative, batch_size=1000
)

# Post-processing just when the MPI process is finished
if lorenz_op.success is True or (parallel is None and not lorenz_op.success):
    print(f"A_hat: {np.array_str(lorenz_op.A_hat, precision=2, suppress_small=True)}")
    print(f"H_hat: {np.array_str(lorenz_op.H_hat, precision=2, suppress_small=True)}")
    print(f"c_hat: {np.array_str(lorenz_op.c_hat, precision=2, suppress_small=True)}")

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

    # Using the derivatives surrogate for time-integrating
    right_operator = ClassWrapper(lorenz_op)

    solver = LSODA(right_operator)

    initial_state = init_state[0]

    estimated_field = solver.run(initial_state, t_test)

    eigs = list()

    for r in estimated_field:
        jac = jacobian(sigma=sigma, beta=beta, rho=rho, x=r[0], y=r[1], z=r[2])

        eigenvalues = np.linalg.eig(jac)[0]
        eigs.append(eigenvalues)

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
