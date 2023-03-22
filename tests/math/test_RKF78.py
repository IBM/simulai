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

from unittest import TestCase

import numpy as np

from simulai.math.integration import LSODA, RKF78, ClassWrapper
from simulai.metrics import L2Norm


class Pendulum:
    def __init__(self, k=1, u=None):
        self.k = k
        self.u = u

        if u is None:
            self.call = self._call_no_forcing
        else:
            self.call = self._call_with_forcing

    def _call_no_forcing(self, data):
        s1 = data[0, 0]
        s2 = data[0, 1]

        return np.array([s2, -self.k * np.sin(s1)])

    def _call_with_forcing(self, data):
        s1 = data[0, 0]
        s2 = data[0, 1]
        u = data[0, 2]

        return np.array([s2, -self.k * np.sin(s1) + u])

    def __call__(self, data):
        return self.call(data)


class LorenzSystem:
    def __init__(self, sigma=10.0, beta=8 / 3, rho=28.0):
        self.sigma = sigma
        self.beta = beta
        self.rho = rho

    def __call__(self, data):
        x = data[0, 0]
        y = data[0, 1]
        z = data[0, 2]

        return np.array(
            [self.sigma * (y - x), x * (self.rho - z) - y, x * y - self.beta * z]
        )

    def eval(self, data):
        return self.__call__(data)


class TestRKF78Integrator(TestCase):
    def setUp(self) -> None:
        pass

    def forcing(self, x):
        A = 0.5
        return A * (np.cos(2 * x) + np.sin(2 * x))

    def test_integration_without_forcings(self):
        N = 1000
        t = np.linspace(0, 10 * np.pi, N)
        dt = t[1] - t[0]

        initial_state = np.array([1, 0, 0])[None, :]

        lorenz = LorenzSystem()
        integrator = RKF78(right_operator=lorenz, adaptive=False)
        output_array = integrator.run(
            initial_state=initial_state, t_f=10 * np.pi, dt=1e-3, n_eq=3
        )

        print("Extrapolation concluded.")

        assert isinstance(
            output_array, np.ndarray
        ), "The output of the integration must be a np.ndarray."

    def test_integration_comparative(self):
        N = 1000
        t = np.linspace(0, 10 * np.pi, N)
        dt = t[1] - t[0]

        initial_state = np.array([1, 0, 0])[None, :]

        lorenz = LorenzSystem()
        integrator = RKF78(right_operator=lorenz, adaptive=False)
        output_array_rkf78 = integrator.run(
            initial_state=initial_state, t_f=10 * np.pi, dt=dt, n_eq=3
        )

        integrator = LSODA(right_operator=ClassWrapper(lorenz))
        output_array_lsoda = integrator.run(
            current_state=initial_state[0], t=np.arange(0, 10 * np.pi, dt)
        )

        error = L2Norm()(
            data=output_array_rkf78[1:],
            reference_data=output_array_lsoda,
            relative_norm=True,
        )
        print(f"{100*error} %")

        print("Extrapolation concluded.")

        assert isinstance(
            output_array_rkf78, np.ndarray
        ), "The output of the integration must be a np.ndarray."
