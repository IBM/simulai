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

import copy
import sys
from typing import Tuple

import numpy as np
from scipy.integrate import odeint

from simulai.abstract import Integral


# Parent class for explicit time-integrators
class ExplicitIntegrator(Integral):
    name = "int"

    def __init__(
        self, coeffs: np.ndarray, weights: np.ndarray, right_operator: callable
    ) -> None:
        """
        Explicit time-integrator parent class.

        Parameters
        ----------
        coeffs: np.ndarray
            Coefficients to shift time during integration.
        weights: np.ndarray
            The weights for penalizing each shifted state.
        right_operator: callable
            The callable for evaluating the residual in each iteration.

        Returns
        -------
        None

        """
        super().__init__()

        self.coeffs = coeffs
        self.weights = weights
        self.right_operator = right_operator
        self.n_stages = len(self.coeffs)
        self.log_phrase = "Executing integrator "

    def step(
        self, variables_state_initial: np.ndarray, dt: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Marches a step in the time-integration.

        Parameters
        ----------
        variables_state_initial: np.ndarray
            Initial state.
        dt: float
            Timestep size.

        Returns
        -------
        (np.ndarray, np.ndarray)
            The integrated state and its time-derivative.
        """
        variables_state = variables_state_initial
        residuals_list = np.zeros((self.n_stages,) + variables_state.shape[1:])

        k_weighted = None
        for stage in range(self.n_stages):
            k = self.right_operator(variables_state)
            residuals_list[stage, :] = k
            k_weighted = self.weights[stage].dot(residuals_list)
            variables_state = (
                variables_state_initial + self.coeffs[stage] * dt * k_weighted
            )

        return variables_state, k_weighted

    def step_with_forcings(
        self, variables_state_initial: np.ndarray, forcing_state: np.ndarray, dt: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Marches a step in the time-integration using a concatenated [variables, forcings] state.

        Parameters
        ----------
        variables_state_initial: np.ndarray
            Initial state.
        forcing_state: np.ndarray
            The state of the forcing terms.
        dt: float
            Timestep size.

        Returns
        -------
        (np.ndarray, np.ndarray)
            The integrated state and its time-derivative.
        """
        variables_state = np.concatenate(
            [variables_state_initial, forcing_state], axis=-1
        )
        residuals_list = np.zeros((self.n_stages,) + variables_state_initial.shape[1:])

        k_weighted = None
        for stage in range(self.n_stages):
            k = self.right_operator(variables_state)
            residuals_list[stage, :] = k
            k_weighted = self.weights[stage].dot(residuals_list)
            variables_state_ = (
                variables_state_initial + self.coeffs[stage] * dt * k_weighted
            )
            variables_state = np.concatenate([variables_state_, forcing_state], axis=1)
        return variables_state_, k_weighted

    def step_with_forcings_separated(
        self, variables_state_initial: np.ndarray, forcing_state: np.ndarray, dt: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        March a single step in the time-integration process, with variables and forcings being treated separately.

        Parameters
        ----------
        variables_state_initial : np.ndarray
            Initial state of the variables.
        forcing_state : np.ndarray
            State of the forcing terms.
        dt : float
            Timestep size.

        Returns
        -------
        tuple
            Integrated state and its time-derivative.
        """

        variables_state = {
            "input_data": variables_state_initial,
            "forcing_data": forcing_state,
        }
        residuals_list = np.zeros((self.n_stages,) + variables_state_initial.shape[1:])

        k_weighted = None

        for stage in range(self.n_stages):
            k = self.right_operator(**variables_state)
            residuals_list[stage, :] = k
            k_weighted = self.weights[stage].dot(residuals_list)
            variables_state_ = (
                variables_state_initial + self.coeffs[stage] * dt * k_weighted
            )
            variables_state = {
                "input_data": variables_state_,
                "forcing_data": forcing_state,
            }

        return variables_state_, k_weighted

    # Looping over multiple steps without using forcings
    def _loop(self, initial_state: np.ndarray, epochs: int, dt: float) -> list:
        """
        Time-integration loop.

        Parameters
        ----------
        initial_state : np.ndarray
            Initial state of the system.
        epochs : int
            Number of steps to be used in the time-integration.
        dt : float
            Timestep size.

        Returns
        -------
        list
            List of integrated states.
        """

        ii = 0
        integrated_variables = list()

        while ii < epochs:
            state, derivative_state = self.step(initial_state, dt)
            integrated_variables.append(state)
            initial_state = state
            sys.stdout.write(
                "\r {}, iteration: {}/{}".format(self.log_phrase, ii + 1, epochs)
            )
            sys.stdout.flush()
            ii += 1

        return integrated_variables

    # Looping over multiple steps using forcings
    def _loop_forcings(
        self, initial_state: np.ndarray, forcings: np.ndarray, epochs: int, dt: float
    ) -> list:
        """
        Forced time-integration loop.

        Parameters
        ----------
        initial_state : np.ndarray
            Initial state of the system.
        forcings : np.ndarray
            Array containing all the forcing states.
        epochs : int
            Number of steps to be used in the time-integration.
        dt : float
            Timestep size.

        Returns
        -------
        list
            List of integrated states.
        """

        ii = 0
        integrated_variables = list()

        while ii < epochs:
            forcings_state = forcings[ii][None, :]
            state, derivative_state = self.step_with_forcings(
                initial_state, forcings_state, dt
            )
            integrated_variables.append(state)
            initial_state = state
            sys.stdout.write(
                "\r {}, iteration: {}/{}".format(self.log_phrase, ii + 1, epochs)
            )
            sys.stdout.flush()
            ii += 1

        return integrated_variables

    def __call__(
        self,
        initial_state: np.ndarray = None,
        epochs: int = None,
        dt: float = None,
        resolution: float = None,
        forcings: np.ndarray = None,
    ) -> np.ndarray:
        """
        Determine the proper time-integration loop to use and execute it.

        Parameters
        ----------
        initial_state : np.ndarray, optional
            Initial state of the system.
        forcings : np.ndarray, optional
            Array containing all the forcing states.
        epochs : int, optional
            Number of steps to be used in the time-integration.
        dt : float, optional
            Timestep size.
        resolution : float, optional
            Resolution at which to return the integrated states.

        Returns
        -------
        np.ndarray
            Array of integrated states.
        """

        if forcings is None:
            integrated_variables = self._loop(initial_state, epochs, dt)
        else:
            integrated_variables = self._loop_forcings(
                initial_state, forcings, epochs, dt
            )

        if resolution is None:
            resolution_step = 1
        elif resolution >= dt:
            resolution_step = int(resolution / dt)
        else:
            raise Exception("Resolution is lower than the discretization step.")

        return np.vstack(integrated_variables)[::resolution_step]


# Built-in Runge-Kutta 4th order
class RK4(ExplicitIntegrator):
    name = "rk4_int"

    def __init__(self, right_operator: callable = None) -> None:
        """
        Initialize a 4th-order Runge-Kutta time-integrator.

        Parameters
        ----------
        right_operator : callable
            An operator representing the right-hand side of a dynamic system.
        """
        weights = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [1 / 6, 2 / 6, 2 / 6, 1 / 6]]
        )

        coeffs = np.array([0.5, 0.5, 1, 1])

        super().__init__(coeffs, weights, right_operator)

        self.log_phrase += "Runge-Kutta 4th order "


### In construction
# Runge-Kutta 7[8]
class RKF78:
    def __init__(
        self,
        right_operator: callable = None,
        tetol: float = 1e-4,
        adaptive: bool = True,
        C: float = 0.8,
    ) -> None:
        self.right_operator = right_operator

        self.log_phrase = "Runge-Kutta 78"

        self.n_stages = 13
        self.n_stages_aux = 12

        self.beta = np.zeros((self.n_stages, self.n_stages_aux))

        self.ch = np.array(
            [
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [34.0 / 105],
                [9.0 / 35],
                [9.0 / 35],
                [9.0 / 280],
                [9.0 / 280],
                [0.0],
                [41.0 / 840],
                [41.0 / 840],
            ]
        )

        self.alpha = np.array(
            [
                [0.0],
                [2.0 / 27],
                [1.0 / 9],
                [1.0 / 6],
                [5.0 / 12],
                [0.5],
                [5.0 / 6],
                [1.0 / 6],
                [2.0 / 3],
                [1.0 / 3],
                [1.0],
                [0.0],
                [1.0],
            ]
        )

        self.beta[1, 0] = 2.0 / 27
        self.beta[2, 0] = 1.0 / 36
        self.beta[3, 0] = 1.0 / 24
        self.beta[4, 0] = 5.0 / 12
        self.beta[5, 0] = 0.05
        self.beta[6, 0] = -25.0 / 108
        self.beta[7, 0] = 31.0 / 300
        self.beta[8, 0] = 2.0
        self.beta[9, 0] = -91.0 / 108
        self.beta[10, 0] = 2383.0 / 4100
        self.beta[11, 0] = 3.0 / 205
        self.beta[12, 0] = -1777.0 / 4100
        self.beta[2, 1] = 1.0 / 12
        self.beta[3, 2] = 1.0 / 8
        self.beta[4, 2] = -25.0 / 16
        self.beta[4, 3] = -self.beta[4, 2]
        self.beta[5, 3] = 0.25
        self.beta[6, 3] = 125.0 / 108
        self.beta[8, 3] = -53.0 / 6
        self.beta[9, 3] = 23.0 / 108
        self.beta[10, 3] = -341.0 / 164
        self.beta[12, 3] = self.beta[10, 3]
        self.beta[5, 4] = 0.2
        self.beta[6, 4] = -65.0 / 27
        self.beta[7, 4] = 61.0 / 225
        self.beta[8, 4] = 704.0 / 45
        self.beta[9, 4] = -976.0 / 135
        self.beta[10, 4] = 4496.0 / 1025
        self.beta[12, 4] = self.beta[10, 4]
        self.beta[6, 5] = 125.0 / 54
        self.beta[7, 5] = -2.0 / 9
        self.beta[8, 5] = -107.0 / 9
        self.beta[9, 5] = 311.0 / 54
        self.beta[10, 5] = -301.0 / 82
        self.beta[11, 5] = -6.0 / 41
        self.beta[12, 5] = -289.0 / 82
        self.beta[7, 6] = 13.0 / 900
        self.beta[8, 6] = 67.0 / 90
        self.beta[9, 6] = -19.0 / 60
        self.beta[10, 6] = 2133.0 / 4100
        self.beta[11, 6] = -3.0 / 205
        self.beta[12, 6] = 2193.0 / 4100
        self.beta[8, 7] = 3.0
        self.beta[9, 7] = 17.0 / 6
        self.beta[10, 7] = 45.0 / 82
        self.beta[11, 7] = -3.0 / 41
        self.beta[12, 7] = 51.0 / 82
        self.beta[9, 8] = -1.0 / 12
        self.beta[10, 8] = 45.0 / 164
        self.beta[11, 8] = 3.0 / 41
        self.beta[12, 8] = 33.0 / 164
        self.beta[10, 9] = 18.0 / 41
        self.beta[11, 9] = 6.0 / 41
        self.beta[12, 9] = 12.0 / 41
        self.beta[12, 11] = 1.0

        self.tetol = tetol
        self.C = C
        self.adaptive = adaptive

    def run(
        self,
        initial_state: np.ndarray = None,
        dt: float = None,
        n_eq: int = None,
        t_f: float = None,
    ) -> np.ndarray:
        initial_state = initial_state.T

        stop_criterion = False

        dt_ = dt
        t_i = 0

        f = np.zeros((n_eq, 13)).astype(float)

        xdot = np.zeros((n_eq, 1)).astype(float)

        xwrk = np.zeros((n_eq, 1)).astype(float)

        xwrk[:, 0] = initial_state[:, 0]
        x = initial_state
        x = x.astype(float)

        solutions = list()

        while not stop_criterion:
            twrk = t_i
            xwrk[:, 0] = x[:, 0]

            if t_i == t_f:
                stop_criterion = True
            else:
                pass

            if np.abs(dt_) > np.abs(t_f - t_i):
                dt_ = t_f - t_i
            else:
                pass

            sys.stdout.write("\r T_i : {}, dt : {}, T_f : {}".format(t_i, dt_, t_f))
            sys.stdout.flush()

            f_state = self.right_operator(x.T)

            f_state_tra = np.transpose(f_state)
            f[:, 0] = f_state_tra

            for k in range(1, self.n_stages):
                for i in range(n_eq):
                    x[i, 0] = xwrk[i, 0] + dt_ * sum(self.beta[k, 0:k] * f[i, 0:k])
                    t_i = twrk + self.alpha[k, 0] * dt_
                    xdot = self.right_operator(x.T)
                    xdot_tra = np.transpose(xdot)
                    f[:, k] = xdot_tra

            xerr = self.tetol

            for i in range(0, n_eq):
                f_tra = np.transpose(f)
                x[i, 0] = xwrk[i, 0] + dt_ * sum(self.ch[:, 0] * f_tra[:, i])

                # truncation error calculations
                ter = abs(
                    (f[i, 0] + f[i, 10] - f[i, 11] - f[i, 12]) * self.ch[11, 0] * dt_
                )
                tol = abs(x[i, 0]) * self.tetol + self.tetol
                tconst = ter / tol

            if tconst > xerr:
                xerr = tconst

            if self.adaptive:
                # compute new step size
                dt_ = self.C * dt_ * ((1.0 / xerr) ** (1.0 / 5))
            else:
                pass

            if xerr > 1:
                # Timestep rejected
                t_i = twrk
                x = xwrk
            else:
                solutions.append(copy.copy(x[None, :, 0]))

        return np.vstack(solutions)


# Wrapper for using the SciPy LSODA (LSODA itself is a wrapper for ODEPACK)
class LSODA:
    def __init__(self, right_operator: callable) -> None:
        self.right_operator = right_operator

        self.log_phrase = "LSODA with forcing"

    def run(self, current_state: np.ndarray = None, t: np.ndarray = None) -> np.ndarray:
        """

        Parameters
        ----------
        current_state : np.ndarray, optional
            The initial state of the system.
        t : np.ndarray, optional
            The time points at which to solve the system.

        Returns
        -------
        solution : np.ndarray
            The solution to the system at the specified time points.

        """
        if hasattr(self.right_operator, "jacobian"):
            Jacobian = self.right_operator.jacobian
        else:
            Jacobian = None

        solution = odeint(self.right_operator.eval, current_state, t, Dfun=Jacobian)
        return solution

    def run_forcing(
        self,
        current_state: np.ndarray = None,
        t: np.ndarray = None,
        forcing: np.ndarray = None,
    ) -> np.ndarray:
        """

        Parameters
        ----------
        current_state : np.ndarray, optional
            The initial state of the system.
        t : np.ndarray, optional
            The time points at which to solve the system.
        forcing : np.ndarray, optional
            The forcing terms to use in the differential equation.

        Returns
        -------
        solutions : np.ndarray
            The solution to the system at the specified time points with the forcing terms applied.

        """
        assert isinstance(
            forcing, np.ndarray
        ), "When running with forcing, a forcing array must be provided."

        if hasattr(self.right_operator, "jacobian"):
            print("Integrating using external Jacobian function")
            Jacobian = self.right_operator.jacobian
        else:
            print("Numerical Jacobian")
            Jacobian = None

        epochs = forcing.shape[0]
        self.right_operator.set(forcing=forcing)

        solutions = [current_state]

        for step in range(epochs):
            solution = odeint(
                self.right_operator.eval_forcing,
                current_state,
                t[step : step + 2],
                args=(step,),
                Dfun=Jacobian,
            )

            solutions.append(solution[1:])
            current_state = solution[-1]

            sys.stdout.write(
                "\r {}, iteration: {}/{}".format(self.log_phrase, step + 1, epochs)
            )
            sys.stdout.flush()

        return np.vstack(solutions)


# Wrapper for handling function objects in time-integrators
class FunctionWrapper:
    def __init__(self, function: callable, extra_dim: bool = True) -> None:
        self.function = function

        if extra_dim is True:
            self.prepare_input = self._extra_dim_prepare_input
        else:
            self.prepare_input = self._no_extra_dim_prepare_input

        if extra_dim is True:
            self.prepare_output = self._extra_dim_prepare_output
        else:
            self.prepare_output = self._no_extra_dim_prepare_output

    def _extra_dim_prepare_input(self, input_data: np.ndarray) -> np.ndarray:
        return input_data[None, :]

    def _no_extra_dim_prepare_input(self, input_data: np.ndarray) -> np.ndarray:
        return input_data

    def _extra_dim_prepare_output(self, output_data: np.ndarray) -> np.ndarray:
        return output_data[0, :]

    def _no_extra_dim_prepare_output(self, output_data: np.ndarray) -> np.ndarray:
        return output_data

    def __call__(self, input_data: np.ndarray) -> np.ndarray:
        input_data = self.prepare_input(input_data)

        return self.prepare_output(self.function(input_data))


# Wrapper for handling class objects in time-integrators
class ClassWrapper:
    def __init__(self, class_instance: callable) -> None:
        assert hasattr(
            class_instance, "eval"
        ), f"The object class_instance={class_instance} has no attribute eval."

        self.class_instance = class_instance

        if hasattr(self.class_instance, "jacobian"):
            self.jacobian = self._jacobian
        else:
            pass

        self.forcing = None

    def _squeezable(self, input: np.ndarray) -> np.ndarray:
        try:
            output = np.squeeze(input, axis=0)
        except:
            output = input

        return output

    def set(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __call__(self, input_data: np.ndarray) -> np.ndarray:
        return self.class_instance(input_data)[0, :]

    def eval(self, input_data: np.ndarray, t: float) -> np.ndarray:
        input_data = input_data
        evaluation = self.class_instance.eval(input_data[None, :])

        return self._squeezable(evaluation)

    def eval_forcing(self, input_data: np.ndarray, t: float, i: int) -> np.ndarray:
        evaluation = self.class_instance.eval(
            input_data[None, :], forcing_data=self.forcing[i : i + 1, :]
        )

        return self._squeezable(evaluation)

    def _jacobian(self, input_data: np.ndarray, *args, **kwargs) -> np.ndarray:
        print("Using jacobian: stiffness alert.")

        return self.class_instance.jacobian(input_data)
