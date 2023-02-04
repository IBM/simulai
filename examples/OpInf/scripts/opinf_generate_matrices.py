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

import numpy as np

from examples.utils.lorenz_solver import lorenz_solver
from simulai.regression import OpInf

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

lorenz_op.construct(input_data=train_field, target_data=train_field_derivative)

D, R_matrix = lorenz_op._generate_data_matrices(
    input_data=train_field, target_data=train_field_derivative
)

print("Matrices generated.")
