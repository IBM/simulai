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

from simulai.templates import ReservoirComputing

def time_function(t):

    omega = np.pi
    return np.sin(omega*t) + np.sin(10*omega*t) + np.sin(20*omega*t)

def bidimensional_map_nonlin_1(h, t, x, y, x_0, y_0):

    gamma = 3*np.pi
    r = np.sqrt(np.square(x - x_0) + np.square(y - y_0))

    f = lambda r: gamma*r + (gamma/2)*r**2 + np.sqrt(r)

    return h(t - f(r))

def bidimensional_map_nonlin_2(h, t, x, y, x_0, y_0):

    gamma = 3*np.pi
    r = np.sqrt(np.square(x - x_0) + np.square(y - y_0))

    f = lambda r: gamma*r

    return h(t - f(r))

def bidimensional_map_nonlin_3(h, t, x, y, x_0, y_0):

    gamma = 3*np.pi
    r = np.sqrt(np.square(x - x_0) + np.square(y - y_0))

    f = lambda r: gamma*r + (gamma/2)*r**3 + np.sqrt(r)

    return h(t - f(r))

def bidimensional_map_nonlin_4(h, t, x, y, x_0, y_0):

    gamma = 3*np.pi
    r = np.sqrt(np.square(x - x_0) + np.square(y - y_0))

    f = lambda r: gamma*r + (gamma/2)*r**3 + np.sqrt(r)

    return h(t - (1/3)*(np.sin(np.pi*t)+1)*f(r))

def bidimensional_map_nonlin_5(h, t, x, y, x_0, y_0):

    gamma = 3*np.pi

    r = np.sqrt(np.square(x - x_0) + np.square(y - y_0))

    x_ = x_0 * (0.99 + 0.01 * np.sin(np.pi*r * t))
    y_ = y_0 * (0.99 + 0.01 * np.sin(np.pi*r * t))

    r = np.sqrt(np.square(x - x_) + np.square(y - y_))

    f = lambda r: gamma*r + (gamma/2)*r**3 + np.sqrt(r)

    return h(t - (1/3)*(np.sin(np.pi*t)+1)*f(r))

def bidimensional_map_nonlin_6(h, t, x, y, x_0, y_0):

    gamma = 3*np.pi
    r = np.sqrt(np.square(x - x_0 + 0.01*x_0*np.sin(gamma*t))
                + np.square(y - y_0 + 0.01*y_0*np.sin(2*gamma*t)))

    f = lambda r: gamma*r + (gamma/2)*r**3 + np.sqrt(r)

    return h(t - f(r))

def bidimensional_map_lin(h, t, x, y, x_0, y_0):

    gamma = 3*np.pi
    r = np.sqrt(np.square(x - x_0) + np.square(y - y_0))

    f = lambda r: gamma*r + (gamma/2)*r**2 + np.sqrt(r)

    return h(t)*f(r)

def reservoir_generator(number_of_reservoirs=None, sparsity_level=None, reservoir_dim=None):

    reservoir_computing = ReservoirComputing(reservoir_dim=reservoir_dim,
                                             sparsity_level=sparsity_level)

    return [reservoir_computing.create_reservoir() for n in range(number_of_reservoirs)]


class Scattering:

    def __init__(self, root=None, scatter_op=None):

        self.root = root
        self.scatter_op = scatter_op

    def exec(self, data=None, scatter_data=None):

        return self.scatter_op(self.root, data, *scatter_data)




