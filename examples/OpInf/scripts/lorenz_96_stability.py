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
import pickle

#!/usr/bin/env python
import warnings
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from scipy.integrate import odeint

from simulai.math.differentiation import CollocationDerivative
from simulai.math.integration import LSODA, ClassWrapper
from simulai.regression import OpInf


def NRMSE(exact, approximated):
    return np.sqrt(
        np.mean(np.square(exact - approximated) / exact.std(axis=0) ** 2, axis=1)
    )


def NRSE(exact, approximated):
    return np.sqrt(np.square(exact - approximated) / exact.std(axis=0) ** 2)


class OpInfError:
    def __init__(self, A_hat: np.ndarray = None, H_hat: np.ndarray = None) -> None:
        self.A_hat = A_hat
        self.H_hat = H_hat

        self.n = A_hat.shape[0]

        epsilon = sp.MatrixSymbol("epsilon", self.n, 1)
        u = sp.MatrixSymbol("u", self.n, 1)

        # u*e.T
        ue = sp.MatMul(u, epsilon.T)
        # (u*e.T).T
        ue_T = ue.T
        # e*e.T
        ee = sp.MatMul(epsilon, epsilon.T)

        # U(u*e.T + (u*e.T).T) + e X e
        v_u = np.array(ue + ue_T + ee)[np.triu_indices(self.n)]
        v = sp.Matrix(v_u)

        H_symb = sp.Matrix(self.H_hat)
        A_symb = sp.Matrix(self.A_hat)

        v_jac = sp.Matrix.jacobian(H_symb @ v, epsilon)

        # A + H*(U(u*e.T + (u*e.T).T) + e X e)
        self.jac_expressions = A_symb + v_jac
        self.error_expressions = A_symb @ epsilon + H_symb @ v

        self.epsilon = epsilon
        self.u = u

        self.jac = None
        self.error = None

        self.compile()

    def compile(self):
        self.jac = self.lambdify(expression=self.jac_expressions)
        self.error = self.lambdify(expression=self.error_expressions)

    def lambdify(self, expression=None):
        return sp.lambdify([self.epsilon, self.u], expression, "numpy")

    def eval_jacobian(self, u: np.array = None, epsilon: np.array = None):
        u = u.T
        epsilon = epsilon.T

        return self.jac(epsilon, u)

    def eval_error(self, u: np.array = None, epsilon: np.array = None):
        u = u.T
        epsilon = epsilon.T

        return self.error(epsilon, u)

    def save(self, name: str = None, path: str = None) -> None:
        blacklist = ["jac", "error"]

        for item in blacklist:
            delattr(self, item)

        with open(os.path.join(path, name + ".pkl"), "wb") as fp:
            pickle.dump(self, fp, protocol=4)


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
estimated_field_derivatives = lorenz_op.eval(input_data=test_field)
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

if os.path.isfile("jacobian_evaluator.pkl") is False:
    jac = OpInfError(A_hat=lorenz_op.A_hat, H_hat=lorenz_op.H_hat)
else:
    with open("jacobian_evaluator.pkl", "rb") as fp:
        jac = pickle.load(fp)

    jac.compile()

n_samples = test_field.shape[0]
eigs = list()

derrors = list()
errors = list()

"""
for i in range(0, n_samples, 10):
    derror = jac.eval_error(u=test_field[i:i+1], epsilon=estimated_field[i:i+1] - test_field[i:i+1]).T
    error = estimated_field[i:i+1] - test_field[i:i+1]

    print(f"Running case: {i}/{n_samples}")

    derrors.append(derror)
    errors.append(error)
"""

for i in range(0, n_samples):
    jacobian = jac.eval_jacobian(
        u=test_field[i : i + 1],
        epsilon=estimated_field[i : i + 1] - test_field[i : i + 1],
    )

    print(f"Running case: {i}/{n_samples}")

    eig = np.linalg.eig(jacobian)[0].real.max()
    eigs.append(eig)

eigs = np.vstack(eigs)
np.save("eigs.npy", eigs)


errors = np.vstack(errors)
np.save("errors.npy", errors)
np.save("derrors.npy", derrors)

jac.save(name="jacobian_evaluator", path=".")
