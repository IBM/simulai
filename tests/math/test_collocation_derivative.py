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

from simulai.metrics import L2Norm
from simulai.math.progression import gp
from simulai.math.differentiation import CollocationDerivative

''' Testing the effectiveness of the collocation differentiation
    (using spline interpolation) using manufactured data:
    U = exp(-omega_t*t)*(x**2*cos(y) + x*y)
'''

class TestCollocationDerivative(TestCase):

    def setUp(self) -> None:

        self.max_steps = 3

    def generate_field(self, x, y, t, omega_t):
        X, T, Y = np.meshgrid(x, t, y)

        # Generic function U = exp(-lambd*t)*(x**2*cos(y) + x*y)
        U = X ** 2 * np.cos(omega_t * T * Y) + (T ** 2) * X * Y
        U_t = -omega_t * Y * np.sin(omega_t * T * Y) * (X ** 2) + 2 * T * X * Y

        return U, U_t

    def test_collocation_derivative(self) -> None:

        # Constructing data
        N_series = gp(init=25, factor=2, n=self.max_steps)
        Nt_series = gp(init=25, factor=2, n=self.max_steps)

        omega_t = 10*np.pi

        x_max = 1
        x_min = 0

        y_max = 1
        y_min = 0

        t_max = 1
        t_min = 0

        for N, Nt in zip(N_series, Nt_series):

            x = np.linspace(x_min, x_max, N)
            y = np.linspace(y_min, y_max, N)
            t = np.linspace(t_min, t_max, Nt)

            dt_ = (1/10)*(t_max - t_min)/Nt
            dt = 10*dt_

            X, T, Y = np.meshgrid(x, t, y)

            # Generic function U = exp(-lambd*t)*(x**2*cos(y) + x*y)
            U = X**2*np.cos(omega_t*T*Y) + (T**2)*X*Y
            U_t = -omega_t*Y*np.sin(omega_t*T*Y)*(X**2) + 2*T*X*Y

            config = {'step': dt}
            diff_op = CollocationDerivative(config=config)

            U_t_est = diff_op.solve(data=U)

            print("Derivatives evaluated.")

            l2_norm = L2Norm()

            error = l2_norm(data=U_t_est, reference_data=U_t, relative_norm=True)

            print(f"Evaluation relative error for N={N} and N_t={Nt}: {100*error} %")

    # Testing to interpolate and differentiate
    def test_collocation_derivative_interpolation(self) -> None:

        # Constructing data
        N_series = gp(init=25, factor=2, n=self.max_steps)
        Nt_series = gp(init=25, factor=2, n=self.max_steps)

        omega_t = 10*np.pi

        x_max = 1
        x_min = 0

        y_max = 1
        y_min = 0

        t_max = 1
        t_min = 0

        multiplier = 2

        for N, Nt in zip(N_series, Nt_series):

            x = np.linspace(x_min, x_max, N)
            y = np.linspace(y_min, y_max, N)
            t = np.linspace(t_min, t_max, Nt)
            t_ = np.linspace(t_min, t_max, multiplier*Nt)

            U, U_t = self.generate_field(x, y, t, omega_t)
            U_, U_t_ = self.generate_field(x, y, t_, omega_t)

            kk = [1, 2, 3, 4, 5]

            for k in kk:

                print(f"Using splines of degree k={k}")

                diff_op = CollocationDerivative(config={}, k=k)

                _, U_t_est = diff_op.interpolate_and_solve(data=U, x_grid=t, x=t_)

                print("Derivatives evaluated.")

                l2_norm = L2Norm()

                error = l2_norm(data=U_t_est, reference_data=U_t_, relative_norm=True)

                print(f"Evaluation relative error for N={N} and N_t={Nt}: {100*error} %")

