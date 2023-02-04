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

#!/usr/bin/env python
import os
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from scipy.integrate import odeint

from simulai.math.differentiation import CollocationDerivative
from simulai.math.integration import LSODA, ClassWrapper
from simulai.regression import ELM, OpInf


def NRMSE(exact, approximated):
    return np.sqrt(
        np.mean(np.square(exact - approximated) / exact.std(axis=0) ** 2, axis=1)
    )


def NRSE(exact, approximated):
    return np.sqrt(np.square(exact - approximated) / exact.std(axis=0) ** 2)


# Reading command line arguments.
parser = ArgumentParser(description="Reading input parameters")

parser.add_argument("--save_path", type=str, help="Save path", default="/tmp")
parser.add_argument("--F", type=float, help="Forcing value", default=8)

args = parser.parse_args()

save_path = args.save_path
F = args.F

initial_states_file = os.path.join(save_path, "initial_random.npy")

tol = 0.5

# These are our constants
N = 40  # Number of variables

n_initial = 100

if F == 8:
    lambda_1 = 1 / 1.68
else:
    lambda_1 = 1 / 2.27


def Lorenz96(x, t):
    """Lorenz 96 model with constant forcing"""
    # Setting up vector
    d = np.zeros(N)
    # Loops over indices (with operations and Python underflow indexing handling edge cases)
    for i in range(N):
        d[i] = (x[(i + 1) % N] - x[i - 2]) * x[i - 1] - x[i] + F
    return d


x0 = F * np.ones(N)  # Initial state (equilibrium)

if os.path.isfile(initial_states_file):
    initial_states = np.load(initial_states_file)
    print("No new initial_states file is being created.")
else:
    initial_states_list = list()
    for nn in range(n_initial):
        x0_nn = x0 + 0.01 * np.random.rand(
            N
        )  # Add small perturbation to the first variable
        initial_states_list.append(x0_nn[None, :])
    initial_states = np.vstack(initial_states_list)

    np.save(initial_states_file, initial_states)

vpt_list = dict()
nrmse_list = list()
case = 1

x0 = initial_states[case]

dt = 0.01
t = np.arange(0.0, 2000.0, dt)
lorenz_data = odeint(Lorenz96, x0, t)[t >= 1000]

diff = CollocationDerivative(config={})
derivative_lorenz_data = diff.solve(data=lorenz_data, x=t[t >= 1000])

n_steps = t[t >= 1000].shape[0]
nt = int(0.5 * n_steps)
nt_test = n_steps - nt
t_test = t[t >= 1000][nt:]
n_field = N

train_field = lorenz_data[:nt]  # manufactured nonlinear oscillator data
train_field_derivative = derivative_lorenz_data[:nt]

test_field = lorenz_data[nt:]  # manufactured nonlinear oscillator data
test_field_derivatives = derivative_lorenz_data[nt:]

lorenz_op = OpInf(bias_rescale=1, solver="pinv")
lorenz_op.fit(input_data=train_field, target_data=train_field_derivative)

init_state = train_field[-1:]
estimated_field_derivatives_train = lorenz_op.eval(input_data=train_field)
estimated_field_derivatives_test = lorenz_op.eval(input_data=test_field)

train_discrepancy = estimated_field_derivatives_train - train_field_derivative
test_discrepancy = estimated_field_derivatives_test - test_field_derivatives

tags = [rf"x_{i}" for i in range(n_field)]

# Construcing jacobian tensor (It could be used during the time-integrations, but seemingly it is not).
lorenz_op.construct_K_op()

# Time-integrating the trained model and visualizing the output.

# Using the derivatives surrogate for time-integrating
right_operator = ClassWrapper(lorenz_op)

solver = LSODA(right_operator)

initial_state = init_state[0]

estimated_field = solver.run(initial_state, t_test)

# Estimating the number of Lyapunov units for the extrapolation.
nrmse = NRMSE(test_field, estimated_field)
nrmse_ = nrmse[nrmse <= tol]

time_ref = (t_test - t_test[0]) / lambda_1
t_ref = time_ref[nrmse <= tol]
VPT = t_ref[-1]

print(f"VPT is {VPT}")

discrepancy = ELM(n_i=N, n_o=N, h=5000)
discrepancy.fit(input_data=train_field, target_data=train_discrepancy, lambd=1e-5)

evaluation = discrepancy.eval(input_data=test_field)

plt.plot(evaluation[:, 0])
plt.plot(test_field[:, 0])
plt.show()
