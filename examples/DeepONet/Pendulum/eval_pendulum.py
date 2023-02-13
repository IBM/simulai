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
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.integrate import odeint

from simulai.file import SPFile
from simulai.metrics import L2Norm

# Reading command line arguments.
parser = ArgumentParser(description="Reading input parameters")

parser.add_argument("--save_path", type=str, help="Save path", default="/tmp")
args = parser.parse_args()

save_path = args.save_path
model_name = os.path.basename(save_path)


# Pendulum numerical solver
class Pendulum:
    def __init__(self, rho: float = None, b: float = None, m: float = None) -> None:
        self.rho = rho
        self.b = b
        self.m = m

    def eval(self, state: np.ndarray = None, t: float = None) -> np.ndarray:
        x = state[0]
        y = state[1]

        x_residual = y
        y_residual = -self.b * y / self.m - self.rho * np.sin(x)

        return np.array([x_residual, y_residual])

    def run(self, initial_state, t):
        solution = odeint(self.eval, initial_state, t)

        return np.vstack(solution)


Q = 1000
N = int(100)
dt = 1 / Q

t = np.arange(0, 100, dt)

initial_state_0 = np.array([1, 1])

s_intv = np.stack([[-2, -2], [2, 2]], axis=0)
U_s = np.random.uniform(low=s_intv[0], high=s_intv[1], size=(N, 2))
U_s = np.vstack([U_s, np.array([[1, 1]])])

solver = Pendulum(rho=9.81, m=1, b=0.05)

saver = SPFile(compact=False)
rober_net = saver.read(model_path=save_path)

for j in range(N + 1):
    exact_data = solver.run(U_s[j], t)

    initial_state_test = U_s[j]

    n_outputs = 2
    n_times = 100

    branch_input_test = np.tile(initial_state_test[None, :], (Q, 1))
    trunk_input_test = np.linspace(0, 1, Q)[:, None]

    approximated_data = rober_net.eval(
        trunk_data=trunk_input_test, branch_data=branch_input_test
    )
    data_ = torch.from_numpy(branch_input_test.astype("float32")).to("cuda")
    # print(rober_net.branch_network.gate(input_data=data_).cpu().detach().numpy())

    eval_list = list()

    for i in range(0, n_times):
        branch_input_test = np.tile(initial_state_test[None, :], (Q, 1))

        approximated_data = rober_net.eval(
            trunk_data=trunk_input_test, branch_data=branch_input_test
        )
        initial_state_test = approximated_data[-1]

        eval_list.append(approximated_data)

    evaluation = np.vstack(eval_list)
    time = np.linspace(0, n_times, evaluation.shape[0])

    l2_norm = L2Norm()

    error_s1 = 100 * l2_norm(
        data=evaluation[:, 0], reference_data=exact_data[:, 0], relative_norm=True
    )
    error_s2 = 100 * l2_norm(
        data=evaluation[:, 1], reference_data=exact_data[:, 1], relative_norm=True
    )

    print(f"State {j}, {U_s[j]}.")
    print(f"Approximation errors, s1: {error_s1} %, s2: {error_s2} ")

    if j % 1 == 0:
        plt.plot(time, evaluation[:, 0], label="Approximated")
        plt.plot(time, exact_data[:, 0], label="Exact", ls="--")
        plt.xlabel("t (s)")
        plt.ylabel("Angle")

        plt.xticks(np.arange(0, 100, 20))
        plt.legend()
        plt.ylim(1.5 * exact_data[:, 0].min(), 1.5 * exact_data[:, 0].max())
        plt.savefig(f"{model_name}_s1_time_int_{j}.png")
        plt.close()

        plt.plot(time, evaluation[:, 1], label="Approximated")
        plt.plot(time, exact_data[:, 1], label="Exact", ls="--")
        plt.xlabel("t (s)")
        plt.ylabel("Angular Speed")

        plt.xticks(np.arange(0, 100, 20))
        plt.legend()
        plt.ylim(1.5 * exact_data[:, 1].min(), 1.5 * exact_data[:, 1].max())
        plt.savefig(f"{model_name}_s2_time_int_{j}.png")
        plt.close()
