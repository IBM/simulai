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
from scipy.integrate import odeint


class ROBER:
    def __init__(self, k1: float = None, k2: float = None, k3: float = None) -> None:
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3

    def eval(self, state: np.ndarray = None, t: float = None) -> np.ndarray:
        s1 = state[0]
        s2 = state[1]
        s3 = state[2]

        s1_residual = -self.k1 * s1 + self.k3 * s2 * s3
        s2_residual = self.k1 * s1 - self.k2 * (s2**2) - self.k3 * s2 * s3
        s3_residual = self.k2 * (s2**2)

        return np.array([s1_residual, s2_residual, s3_residual])

    def run(self, initial_state, t):
        print("Solving ROBER ...")
        solution = odeint(self.eval, initial_state, t)

        return np.vstack(solution)


k1 = 0.04
k2 = 3e7
k3 = 1e4

T_max = 500
dt = 0.1

t = np.arange(0, T_max, dt)
initial_state = np.array([1, 0, 0])

output_dataset = list()
input_dataset = list()

solver = ROBER(k1=k1, k2=k2, k3=k3)

solution = solver.run(initial_state, t)

solution = solution * np.array([1, 1e4, 1])

ground_truth = np.load("evaluation.npy")

plt.plot(t, solution)
plt.show()
