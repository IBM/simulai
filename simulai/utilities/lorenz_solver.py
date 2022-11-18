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

import sys
import os
import numpy as np
from simulai.utilities.problem_classes import LorenzSystem
from simulai.math.integration import RK4
import matplotlib.pyplot as plt
from fractions import Fraction
from argparse import ArgumentParser

# Testing to solve a nonlinear oscillator problem using
# a 4th order and 4 steps Runge-Kutta


def lorenz_solver(rho, beta, sigma, T, dt, data_path, initial_state, solver='RK4', **kwargs) -> object:
    """

    Parameters
    ----------
    rho: float
    beta:
    sigma
    T: int
    dt: float
    data_path: str
    initial_state: List[float]

    Returns
    -------
    (np.ndarray, np.ndarray)
        the full path to the directory that was created to store the results of the simulation
    """
    if not os.path.isdir(data_path):
        os.makedirs(data_path, exist_ok=True)
    elif data_path=="on_memory":
        data_path = "no_path"

    beta_str = kwargs.get('beta_str').replace('/', '_')

    # Using the built-in RK45 implementation
    if solver == 'RK4':

        problem = LorenzSystem(rho, sigma, beta)
        time = np.arange(0, T, dt)

        solver = RK4(problem)

        variables_timesteps = list()
        derivatives_timesteps = list()

        current_state = initial_state

        iter = 0

        for tt in range(time.shape[0]):

            variables_state, derivatives_state = solver.step(current_state, dt)

            variables_timesteps.append(variables_state)
            derivatives_timesteps.append(derivatives_state)

            current_state = variables_state

            sys.stdout.write("\rIteration {}".format(iter))
            sys.stdout.flush()
            iter += 1

        variables_matrix = np.vstack(variables_timesteps)
        derivatives_matrix = np.vstack(derivatives_timesteps)

    # Using the SciPy RK4 implementation
    elif solver == 'RK45':

        from scipy.integrate import solve_ivp
        from simulai.math.differentiation import CollocationDerivative

        problem = LorenzSystem(rho, sigma, beta, use_t=True)
        time = np.arange(0, T, dt)

        current_state = initial_state

        solution = solve_ivp(problem.eval, [0, T], current_state[0], max_step=dt, method='RK45')

        variables_matrix = solution['y'].T
        time = solution['t']

        # After the time-integrations, it is necessary to evaluate the derivatives numerically
        diff = CollocationDerivative(config={})
        derivatives_matrix = diff.solve(data=variables_matrix, x=time)


    plt.plot(time, variables_matrix[:, 0], label="x")
    plt.plot(time, variables_matrix[:, 1], label="y")
    plt.plot(time, variables_matrix[:, 2], label="z")

    label_string = "rho_{}_sigma_{}_beta_{}_T_{}_dt_{}/".format(rho, sigma, beta_str, T, dt)

    all_variables_matrix = np.hstack([variables_matrix, derivatives_matrix])
    n_col = all_variables_matrix.shape[1]
    all_variables_record = np.core.records.fromarrays(np.split(all_variables_matrix, n_col, axis=1),
                                                      names='x, y, z, dx_dt, dy_dt, dz_dt',
                                                      formats='f8, f8, f8, f8, f8, f8')

    if data_path != "no_path":
        dir_path = os.path.join(data_path, label_string)
        if not os.path.isdir(dir_path):
             os.mkdir(dir_path)

        np.save(os.path.join(data_path, "{}Lorenz_data.npy".format(label_string)), all_variables_record)

    plt.xlabel("Time(s)")
    plt.title("Lorenz System")
    plt.legend()
    plt.grid(True)
    plt.close()

    if solver == 'RK45':
        return variables_matrix, derivatives_matrix, time
    else:
        return variables_matrix, derivatives_matrix


def lorenz_solver_forcing(rho, beta, sigma, T, dt, data_path, initial_state, forcing=None, **kwargs):
    """

    Parameters
    ----------
    rho: float
    beta:
    sigma
    T: int
    dt: float
    data_path: str
    initial_state: List[float]

    Returns
    -------
    (np.ndarray, np.ndarray)
        the full path to the directory that was created to store the results of the simulation
    """
    if not os.path.isdir(data_path):
        os.makedirs(data_path, exist_ok=True)
    elif data_path=="on_memory":
        data_path = "no_path"

    beta_str = kwargs.get('beta_str').replace('/', '_')

    problem = LorenzSystem(rho, sigma, beta, forcing=True)

    solver = RK4(problem)

    time = np.arange(0, T, dt)

    variables_timesteps = list()
    derivatives_timesteps = list()

    current_state = initial_state

    iter = 0

    for tt in range(time.shape[0]):

        variables_state, derivatives_state = solver.step_with_forcings(current_state, forcing[tt:tt+1], dt)

        variables_timesteps.append(variables_state)
        derivatives_timesteps.append(derivatives_state)
        current_state = variables_state

        sys.stdout.write("\rIteration {}".format(iter))
        sys.stdout.flush()
        iter += 1

    variables_matrix = np.vstack(variables_timesteps)
    derivatives_matrix = np.vstack(derivatives_timesteps)

    plt.plot(time, variables_matrix[:, 0], label="x")
    plt.plot(time, variables_matrix[:, 1], label="y")
    plt.plot(time, variables_matrix[:, 2], label="z")

    label_string = "rho_{}_sigma_{}_beta_{}_T_{}_dt_{}/".format(rho, sigma, beta_str, T, dt)

    all_variables_matrix = np.hstack([variables_matrix, derivatives_matrix])
    n_col = all_variables_matrix.shape[1]
    all_variables_record = np.core.records.fromarrays(np.split(all_variables_matrix, n_col, axis=1),
                                                      names='x, y, z, dx_dt, dy_dt, dz_dt',
                                                      formats='f8, f8, f8, f8, f8, f8')

    if data_path != "no_path":
        dir_path = os.path.join(data_path, label_string)
        if not os.path.isdir(dir_path):
             os.mkdir(dir_path)

        np.save(os.path.join(data_path, "{}Lorenz_data.npy".format(label_string)), all_variables_record)

    plt.xlabel("Time(s)")
    plt.title("Lorenz System")
    plt.legend()
    plt.grid(True)
    plt.close()

    return variables_matrix, derivatives_matrix

if __name__ == "__main__":

    parser = ArgumentParser(description="Reading input arguments")
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--time', type=float)
    parser.add_argument('--dt', type=float)
    parser.add_argument('--rho', type=float)
    parser.add_argument('--sigma', type=float)
    parser.add_argument('--beta', type=str)

    args = parser.parse_args()

    data_path = args.data_path
    T = args.time
    dt = args.dt

    initial_state = np.array([1, 2, 3])[None, :]

    rho = args.rho #28

    beta_str = args.beta
    beta = float(Fraction(beta_str)) #8/3

    sigma = args.sigma #10

    lorenz_solver(rho=rho, beta=beta, sigma=sigma,
                  T=T, dt=dt,
                  data_path=data_path,
                  initial_state=initial_state,
                  beta_str=beta_str
                  )


