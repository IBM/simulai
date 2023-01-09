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
from unittest import TestCase

from simulai.math.integration import RK4, LSODA, ClassWrapper

class Pendulum:

    def __init__(self, k=1, u=None):

        self.k = k
        self.u = u

        if u is None:
            self.eval = self._eval_no_forcing
        else:
            self.eval = self._eval_with_forcing

    def _eval_no_forcing(self, data):

        s1 = data[0, 0]
        s2 = data[0, 1]

        return np.array([s2, -self.k * np.sin(s1)])

    def _eval_with_forcing(self, data, forcing_data=None):

        s1 = data[0, 0]
        s2 = data[0, 1]
        u = forcing_data[0,0]

        return np.array([s2, -self.k * np.sin(s1) + u])

    def jacobian(self, x):

        return np.array([[0, 1],
                         [self.k*np.cos(x[0]), 0]])

    def __call__(self, data):

        return self.eval(data)

class TestLSODAIntegrator(TestCase):

    def setUp(self) -> None:
        pass

    def forcing(self, x):
        A = 0.5
        return A*(np.cos(2*x) + np.sin(2*x))

    def test_integration_without_forcings(self):

        N = 1000
        t = np.linspace(0, 10*np.pi, N)
        dt = (t[1] - t[0])

        initial_state = np.array([0, 1])

        pendulum = Pendulum(k=1)
        right_operator = ClassWrapper(pendulum)

        solver = LSODA(right_operator)

        estimated_field = solver.run(initial_state, t)

        print("Extrapolation concluded.")

        assert isinstance(estimated_field, np.ndarray), "The output of the integration must be a np.ndarray."

    def test_integration_with_forcings(self):

        N = 1000
        t = np.linspace(0, 10*np.pi, N)

        initial_state = np.array([0, 1])
        forcings = self.forcing(t)[:, None]

        pendulum = Pendulum(k=1, u=forcings)
        right_operator = ClassWrapper(pendulum)

        solver = LSODA(right_operator)

        estimated_field = solver.run_forcing(initial_state, t, forcings)

        print("Extrapolation concluded.")

        assert isinstance(estimated_field, np.ndarray), "The output of the integration must be a np.ndarray."

    def test_integration_with_forcings_stiffness(self):

        N = 1000
        t = np.linspace(0, 10*np.pi, N)

        initial_state = np.array([0, 1])
        forcings = 1e7*self.forcing(t)[:, None]

        pendulum = Pendulum(k=1, u=forcings)
        right_operator = ClassWrapper(pendulum)

        solver = LSODA(right_operator)

        estimated_field = solver.run_forcing(initial_state, t, forcings)

        print("Extrapolation concluded.")

        assert isinstance(estimated_field, np.ndarray), "The output of the integration must be a np.ndarray."