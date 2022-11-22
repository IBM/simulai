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

import sys, os
import numpy as np
from simulai.utilities.problem_classes import NonlinearOscillator
from simulai.math.integration import RK4
from argparse import ArgumentParser
sys.path.insert(0, '.')

# Testing to solve a nonlinear oscillator problem using
# a 4th order and a four steps Runge-Kutta
def oscillator_solver(T, dt, initial_state, extra_params=None):
    """

    Parameters
    ----------
    T: int
    dt: float
    initial_state: np.array

    Returns
    -------
    np.ndarray, np.ndarray
    """

    if extra_params is None:
        problem = NonlinearOscillator()
    elif type(extra_params) == dict:
        problem = NonlinearOscillator(**extra_params)
    else:
        raise Exception(f"extra_params it is expected to be a dict or None but received {type(extra_params)}.")

    solver = RK4(problem)

    time = np.arange(0, T, dt)

    variables_timesteps = list()
    derivatives_timesteps = list()

    current_state = initial_state

    for tt in range(time.shape[0]):

        variables_state, derivatives_state = solver.step(current_state, dt)
        variables_timesteps.append(variables_state)
        derivatives_timesteps.append(derivatives_state)
        current_state = variables_state
        sys.stdout.write("\rIteration {}".format(tt))
        sys.stdout.flush()

    variables_matrix = np.vstack(variables_timesteps)
    derivatives_matrix = np.vstack(derivatives_timesteps)

    return variables_matrix, derivatives_matrix

# Testing to solve a nonlinear oscillator problem using
# a 4th order and a four steps Runge-Kutta
def oscillator_solver_forcing(T, dt, initial_state, forcing=None, p=3, extra_params=None):
    """

    Parameters
    ----------
    T: int
    dt: float
    initial_state: np.array

    Returns
    -------
    np.ndarray, np.ndarray
    """

    if extra_params is None:
        problem = NonlinearOscillator(forcing=True, p=p)
    elif type(extra_params) == dict:
        problem = NonlinearOscillator(forcing=True, p=p, **extra_params)
    else:
        raise Exception(f"extra_params it is expected to be a dict or None but received {type(extra_params)}.")

    solver = RK4(problem)

    time = np.arange(0, T, dt)

    variables_timesteps = list()
    derivatives_timesteps = list()

    current_state = initial_state

    for tt in range(time.shape[0]):

        variables_state, derivatives_state = solver.step_with_forcings(current_state, forcing[tt:tt+1], dt)
        variables_timesteps.append(variables_state)
        derivatives_timesteps.append(derivatives_state)
        current_state = variables_state
        sys.stdout.write("\rIteration {}".format(tt))
        sys.stdout.flush()

    variables_matrix = np.vstack(variables_timesteps)
    derivatives_matrix = np.vstack(derivatives_timesteps)

    return variables_matrix, derivatives_matrix

def main():
    parser = ArgumentParser(description="Reading input arguments")
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--time', type=float)
    parser.add_argument('--dt', type=float)

    args = parser.parse_args()

    data_path = args.data_path
    T = args.time
    dt = args.dt

    initial_state = np.array([2, 0])[None, :]

    variables, derivatives = oscillator_solver(T, dt, initial_state)


if __name__ == "__main__":
    main()


