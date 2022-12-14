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
import sys
from scipy.integrate import odeint

from simulai.abstract import Integral

# Parent class for explicit time-integrators
class ExplicitIntegrator(Integral):

    name = "int"

    def __init__(self, coeffs:np.ndarray, weights:np.ndarray, right_operator:callable) -> None:

        """ Explicit time-integrator parent class
        :param coeffs: coefficients to shift time during integration
        :type coeffs: np.ndarray
        :param weights: the weights for penalizing each shifted state
        :type weights: np.ndarray
        :param right_operator: the callable for evaluating the residual in each iteration
        :returns: nothing
        """

        super().__init__()

        self.coeffs = coeffs
        self.weights = weights
        self.right_operator = right_operator
        self.n_stages = len(self.coeffs)
        self.log_phrase = "Executing integrator "

    def step(self, variables_state_initial:np.ndarray, dt:float) -> (np.ndarray, np.ndarray):

        """It marches a step in the time-integration
        :param variables_state_initial: initial state
        :type variables_state_initial: np.ndarray
        :param dt: timestep size
        :type dt: float
        :returns: the integrated state and its time-derivative
        :rtype: (np.ndarray, np.ndarray)
        """

        variables_state = variables_state_initial
        residuals_list = np.zeros((self.n_stages,) + variables_state.shape[1:])

        k_weighted = None
        for stage in range(self.n_stages):

            k = self.right_operator(variables_state)
            residuals_list[stage, :] = k
            k_weighted = self.weights[stage].dot(residuals_list)
            variables_state = variables_state_initial + self.coeffs[stage] * dt * k_weighted

        return variables_state, k_weighted

    def step_with_forcings(self, variables_state_initial:np.ndarray,
                                 forcing_state:np.ndarray, dt:float) -> (np.ndarray, np.ndarray):

        """It marches a step in the time-integration using a concatenated [variables, forcings] state
        :param variables_state_initial: initial state
        :type variables_state_initial: np.ndarray
        :param forcing_state: the state of the forcing terms
        :type forcing_state: np.ndarray
        :param dt: timestep size
        :type dt: float
        :returns: the integrated state and its time-derivative
        :rtype: (np.ndarray, np.ndarray)
        """

        variables_state = np.concatenate([variables_state_initial, forcing_state],
                                         axis=-1)

        residuals_list = np.zeros((self.n_stages,) + variables_state_initial.shape[1:])

        k_weighted = None

        for stage in range(self.n_stages):

            k = self.right_operator(variables_state)
            residuals_list[stage, :] = k
            k_weighted = self.weights[stage].dot(residuals_list)
            variables_state_ = variables_state_initial + self.coeffs[stage] * dt * k_weighted
            variables_state = np.concatenate([variables_state_, forcing_state],
                                             axis=1)
        return variables_state_, k_weighted

    def step_with_forcings_separated(self, variables_state_initial:np.ndarray,
                                           forcing_state:np.ndarray, dt:float) -> (np.ndarray, np.ndarray):

        """It marches a step in the time-integration for variables and forcings being operated separately
        :param variables_state_initial: initial state
        :type variables_state_initial: np.ndarray
        :param forcing_state: the state of the forcing terms
        :type forcing_state: np.ndarray
        :param dt: timestep size
        :type dt: float
        :returns: the integrated state and its time-derivative
        :rtype: (np.ndarray, np.ndarray)
        """

        variables_state = {'input_data': variables_state_initial, 'forcing_data': forcing_state}
        residuals_list = np.zeros((self.n_stages,) + variables_state_initial.shape[1:])

        k_weighted = None

        for stage in range(self.n_stages):

            k = self.right_operator(**variables_state)
            residuals_list[stage, :] = k
            k_weighted = self.weights[stage].dot(residuals_list)
            variables_state_ = variables_state_initial + self.coeffs[stage] * dt * k_weighted
            variables_state = {'input_data': variables_state_, 'forcing_data': forcing_state}

        return variables_state_, k_weighted

    # Looping over multiple steps without using forcings
    def _loop(self, initial_state:np.ndarray, epochs:int, dt:float) -> list:

        """No forced time-integration loop
        :param initial_state: the initial state
        :type initial_state: np.ndarray
        :param epochs: number of steps to be used in the time-integration
        :type epochs: int
        :param dt: the timestep size
        :type dt: float
        :returns: the list of integrated states
        :rtype: list
        """

        ii = 0
        integrated_variables = list()

        while ii < epochs:

            state, derivative_state = self.step(initial_state, dt)
            integrated_variables.append(state)
            initial_state = state
            sys.stdout.write("\r {}, iteration: {}/{}".format(self.log_phrase, ii+1, epochs))
            sys.stdout.flush()
            ii += 1

        return integrated_variables

    # Looping over multiple steps using forcings
    def _loop_forcings(self,  initial_state:np.ndarray, forcings:np.ndarray, epochs:int, dt:float) -> list:

        """Forced time-integration loop
        :param initial_state: the initial state
        :type initial_state: np.ndarray
        :param forcings: the array containing all the forcing states
        :type forcings: np.ndarray
        :param epochs: number of steps to be used in the time-integration
        :type epochs: int
        :param dt: the timestep size
        :type dt: float
        :returns: the list of integrated states
        :rtype: list
        """

        ii = 0
        integrated_variables = list()

        while ii < epochs:
            forcings_state = forcings[ii][None, :]
            state, derivative_state = self.step_with_forcings(initial_state, forcings_state, dt)
            integrated_variables.append(state)
            initial_state = state
            sys.stdout.write("\r {}, iteration: {}/{}".format(self.log_phrase, ii + 1, epochs))
            sys.stdout.flush()
            ii += 1

        return integrated_variables

    def __call__(self, initial_state:np.ndarray=None, epochs:int=None,
                       dt:float=None, resolution:float=None, forcings:np.ndarray=None) -> np.ndarray:

        """It determined the proper time-integration loop to be used and executes it.
        :param initial_state: the initial state
        :type initial_state: np.ndarray
        :param forcings: the array containing all the forcing states
        :type forcings: np.ndarray
        :param epochs: number of steps to be used in the time-integration
        :type epochs: int
        :param dt: the timestep size
        :type dt: float
        :returns: the list of integrated states
        :rtype: list
        """

        if forcings is None:
            integrated_variables = self._loop(initial_state, epochs, dt)
        else:
            integrated_variables = self._loop_forcings(initial_state, forcings, epochs, dt)

        if resolution is None:
            resolution_step = 1
        elif resolution >= dt:
            resolution_step = int(resolution/dt)
        else:
            raise Exception("Resolution is lower than the discretization step.")

        return np.vstack(integrated_variables)[::resolution_step]

# Built-in Runge-Kutta 4th order
class RK4(ExplicitIntegrator):

    name = "rk4_int"

    def __init__(self, right_operator):

        """ Runge-Kutta 4th-order time-integrator
        :param right_operator: an operator for representing the right-hand side
        of a dynamic system
        :type right_operator: callable (def or lambda)
        """

        weights = np.array(
                           [[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 1, 0],
                           [1/6, 2/6, 2/6, 1/6]]
                          )

        coeffs = np.array([0.5, 0.5, 1, 1])

        super().__init__(coeffs, weights, right_operator)

        self.log_phrase += "Runge-Kutta 4th order "

# Wrapper for using the SciPy LSODA (LSODA itself is a wrapper for ODEPACK)
class LSODA:

    def __init__(self, right_operator:callable) -> None:

        self.right_operator = right_operator

        self.log_phrase = "LSODA with forcing"

    def run(self, current_state:np.ndarray=None, t:np.ndarray=None) -> np.ndarray:

        if hasattr(self.right_operator, "jacobian"):
            Jacobian = self.right_operator.jacobian
        else:
            Jacobian = None

        solution = odeint(self.right_operator.eval, current_state, t,
                          Dfun=Jacobian)
        return solution

    def run_forcing(self, current_state: np.ndarray = None, t: np.ndarray = None,
                          forcing:np.ndarray=None) -> np.ndarray:

        assert isinstance(forcing, np.ndarray), "When running with forcing, a forcing array must be provided."
        
        if hasattr(self.right_operator, "jacobian"):
            Jacobian = self.right_operator.jacobian
        else:
            Jacobian = None

        epochs = forcing.shape[0]
        self.right_operator.set(forcing=forcing)

        solutions = [current_state]

        for step in range(epochs):
            solution = odeint(self.right_operator.eval_forcing, current_state, t[step:step+2], args=(step,),
                              Dfun=Jacobian)

            solutions.append(solution[1:])
            current_state = solution[-1]

            sys.stdout.write("\r {}, iteration: {}/{}".format(self.log_phrase, step + 1, epochs))
            sys.stdout.flush()

        return np.vstack(solutions)

# Wrapper for handling function objects in time-integrators
class FunctionWrapper:

    def __init__(self, function:callable, extra_dim:bool=True) -> None:

        self.function = function

        if extra_dim is True:
            self.prepare_input = self._extra_dim_prepare_input
        else:
            self.prepare_input = self._no_extra_dim_prepare_input

        if extra_dim is True:
            self.prepare_output = self._extra_dim_prepare_output
        else:
            self.prepare_output = self._no_extra_dim_prepare_output

    def _extra_dim_prepare_input(self, input_data:np.ndarray) -> np.ndarray:

        return input_data[None, :]

    def _no_extra_dim_prepare_input(self, input_data:np.ndarray) -> np.ndarray:

        return input_data

    def _extra_dim_prepare_output(self, output_data:np.ndarray) -> np.ndarray:

        return output_data[0, :]

    def _no_extra_dim_prepare_output(self, output_data:np.ndarray) -> np.ndarray:

        return output_data

    def __call__(self, input_data:np.ndarray) -> np.ndarray:

        input_data = self.prepare_input(input_data)

        return self.prepare_output(self.function(input_data))

# Wrapper for handling class objects in time-integrators
class ClassWrapper:

    def __init__(self, class_instance:callable) -> None:

        assert hasattr(class_instance, 'eval'), f"The objet class_instance={class_instance} has no attribute eval."

        self.class_instance = class_instance

        if hasattr(self.class_instance, 'jacobian'):
            self.jacobian = self._jacobian
        else:
            pass

        self.forcing = None

    def _squeezable(self, input:np.ndarray) -> np.ndarray:

        try:
            output = np.squeeze(input, axis=0)
        except:
            output = input

        return output

    def set(self, **kwargs):

        for key, value in kwargs.items():
            setattr(self, key, value)

    def __call__(self, input_data:np.ndarray) -> np.ndarray:

        return self.class_instance(input_data)[0, :]

    def eval(self, input_data:np.ndarray, t:float) -> np.ndarray:

        input_data = input_data
        evaluation = self.class_instance.eval(input_data[None, :])

        return self._squeezable(evaluation)

    def eval_forcing(self, input_data: np.ndarray, t: float, i:int) -> np.ndarray:

        evaluation = self.class_instance.eval(input_data[None, :], forcing_data=self.forcing[i:i+1,:])

        return self._squeezable(evaluation)

    def _jacobian(self, input_data:np.ndarray, t:float) -> np.ndarray:

        print("Using jacobian: stiffness alert.")

        return self.class_instance.jacobian(input_data)
