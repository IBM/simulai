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
    """
    Computes the sum of three sine waves at frequencies of pi, 10*pi, and 20*pi.

    Parameters
    ----------
    t : float or array-like
        The time value(s) at which to compute the function.

    Returns
    -------
    output : float or array-like
        The value(s) of the function at the specified time(s).
    """

    omega = np.pi
    return np.sin(omega * t) + np.sin(10 * omega * t) + np.sin(20 * omega * t)


def bidimensional_map_nonlin_1(h, t, x, y, x_0, y_0):
    """
    Computes a bidimensional nonlinear map using the given parameters.

    Parameters
    ----------
    h : function
        The function to use for mapping.
    t : float or array-like
        The time value(s) at which to compute the function.
    x : float or array-like
        The x-coordinate(s) of the map.
    y : float or array-like
        The y-coordinate(s) of the map.
    x_0 : float
        The x-coordinate of the center of the map.
    y_0 : float
        The y-coordinate of the center of the map.

    Returns
    -------
    output : float or array-like
        The value(s) of the bidimensional map at the specified time(s) and coordinates.
    """

    gamma = 3 * np.pi
    r = np.sqrt(np.square(x - x_0) + np.square(y - y_0))

    f = lambda r: gamma * r + (gamma / 2) * r**2 + np.sqrt(r)

    return h(t - f(r))


def bidimensional_map_nonlin_2(h, t, x, y, x_0, y_0):
    """
    Computes a bidimensional nonlinear map using the given parameters.

    Parameters
    ----------
    h : function
        The function to use for mapping.
    t : float or array-like
        The time value(s) at which to compute the function.
    x : float or array-like
        The x-coordinate(s) of the map.
    y : float or array-like
        The y-coordinate(s) of the map.
    x_0 : float
        The x-coordinate of the center of the map.
    y_0 : float
        The y-coordinate of the center of the map.

    Returns
    -------
    output : float or array-like
        The value(s) of the bidimensional map at the specified time(s) and coordinates.
    """

    gamma = 3 * np.pi
    r = np.sqrt(np.square(x - x_0) + np.square(y - y_0))

    f = lambda r: gamma * r

    return h(t - f(r))


def bidimensional_map_nonlin_3(h, t, x, y, x_0, y_0):
    """
    Computes the nonlinear bidimensional map with function h.

    Parameters
    ----------
    h : function
        The function to use for mapping.
    t : float or array-like
        The time value(s) at which to compute the function.
    x : float or array-like
        The x-coordinate(s) of the map.
    y : float or array-like
        The y-coordinate(s) of the map.
    x_0 : float
        The x-coordinate of the center of the map.
    y_0 : float
        The y-coordinate of the center of the map.

    Returns
    -------
    float
        Result of applying the map.
    """
    gamma = 3 * np.pi
    r = np.sqrt(np.square(x - x_0) + np.square(y - y_0))

    f = lambda r: gamma * r + (gamma / 2) * r**3 + np.sqrt(r)

    return h(t - f(r))


def bidimensional_map_nonlin_4(h, t, x, y, x_0, y_0):
    """
    Computes the nonlinear bidimensional map with function h and a temporal modulation.

    Parameters
    ----------
    h : function
        The function to use for mapping.
    t : float or array-like
        The time value(s) at which to compute the function.
    x : float or array-like
        The x-coordinate(s) of the map.
    y : float or array-like
        The y-coordinate(s) of the map.
    x_0 : float
        The x-coordinate of the center of the map.
    y_0 : float
        The y-coordinate of the center of the map.

    Returns
    -------
    float
        Result of applying the map.
    """
    gamma = 3 * np.pi
    r = np.sqrt(np.square(x - x_0) + np.square(y - y_0))

    f = lambda r: gamma * r + (gamma / 2) * r**3 + np.sqrt(r)

    return h(t - (1 / 3) * (np.sin(np.pi * t) + 1) * f(r))


def bidimensional_map_nonlin_5(h, t, x, y, x_0, y_0):
    """
    Computes the nonlinear bidimensional map function.

    Parameters
    ----------
    h : function
        The function to use for mapping.
    t : float or array-like
        The time value(s) at which to compute the function.
    x : float or array-like
        The x-coordinate(s) of the map.
    y : float or array-like
        The y-coordinate(s) of the map.
    x_0 : float
        The x-coordinate of the center of the map.
    y_0 : float
        The y-coordinate of the center of the map.

    Returns
    -------
    result : float
        Result of the bidimensional map function.
    """
    gamma = 3 * np.pi

    r = np.sqrt(np.square(x - x_0) + np.square(y - y_0))

    x_ = x_0 * (0.99 + 0.01 * np.sin(np.pi * r * t))
    y_ = y_0 * (0.99 + 0.01 * np.sin(np.pi * r * t))

    r = np.sqrt(np.square(x - x_) + np.square(y - y_))

    f = lambda r: gamma * r + (gamma / 2) * r**3 + np.sqrt(r)

    return h(t - (1 / 3) * (np.sin(np.pi * t) + 1) * f(r))


def bidimensional_map_nonlin_6(h, t, x, y, x_0, y_0):
    """
    Computes the nonlinear bidimensional map function.

    Parameters
    ----------
    h : function
        The function to use for mapping.
    t : float or array-like
        The time value(s) at which to compute the function.
    x : float or array-like
        The x-coordinate(s) of the map.
    y : float or array-like
        The y-coordinate(s) of the map.
    x_0 : float
        The x-coordinate of the center of the map.
    y_0 : float
        The y-coordinate of the center of the map.

    Returns
    -------
    result : float
        Result of the bidimensional map function.
    """
    gamma = 3 * np.pi
    r = np.sqrt(
        np.square(x - x_0 + 0.01 * x_0 * np.sin(gamma * t))
        + np.square(y - y_0 + 0.01 * y_0 * np.sin(2 * gamma * t))
    )

    f = lambda r: gamma * r + (gamma / 2) * r**3 + np.sqrt(r)

    return h(t - f(r))


def bidimensional_map_lin(h, t, x, y, x_0, y_0):
    """
    Compute the bidimensional linear map.

    Parameters
    ----------
    h : function
        The function to use for mapping.
    t : float or array-like
        The time value(s) at which to compute the function.
    x : float or array-like
        The x-coordinate(s) of the map.
    y : float or array-like
        The y-coordinate(s) of the map.
    x_0 : float
        The x-coordinate of the center of the map.
    y_0 : float
        The y-coordinate of the center of the map.

    Returns
    -------
    float
        The result of the bidimensional linear map for the given input.
    """
    gamma = 3 * np.pi
    r = np.sqrt(np.square(x - x_0) + np.square(y - y_0))

    f = lambda r: gamma * r + (gamma / 2) * r**2 + np.sqrt(r)

    return h(t) * f(r)


def reservoir_generator(
    number_of_reservoirs=None, sparsity_level=None, reservoir_dim=None
):
    """Generate a list of reservoirs.

    Parameters
    ----------
    number_of_reservoirs : int, optional
        The number of reservoirs to generate, by default None.
    sparsity_level : float, optional
        The sparsity level of the reservoirs, by default None.
    reservoir_dim : int, optional
        The dimension of the reservoirs, by default None.

    Returns
    -------
    list
        A list containing the generated reservoirs.
    """
    reservoir_computing = ReservoirComputing(
        reservoir_dim=reservoir_dim, sparsity_level=sparsity_level
    )

    return [reservoir_computing.create_reservoir() for n in range(number_of_reservoirs)]


class Scattering:
    """
    Class for applying a scattering operator on data using a root and scatter_data.

    Parameters
    ----------
    root : int
        Index of the root node to apply the scatter operation.
    scatter_op : function
        Function that takes the root node, input data and scatter_data as input and returns the scattered data.

    Methods
    -------
    exec(data=None, scatter_data=None)
        Applies the scatter operation on data using the specified root and scatter_data.
    """

    def __init__(self, root=None, scatter_op=None):
        """
        Initialize a Scattering instance.

        Parameters
        ----------
        root : int
            Index of the root node to apply the scatter operation.
        scatter_op : function
            Function that takes the root node, input data and scatter_data as input and returns the scattered data.
        """
        self.root = root
        self.scatter_op = scatter_op

    def exec(self, data=None, scatter_data=None):
        """
        Apply scatter operation on input data using specified root and scatter_data.

        Parameters
        ----------
        data : array_like
            Input data on which scatter operation will be applied.
        scatter_data : tuple
            Tuple of additional data that is needed for scatter operation.

        Returns
        ----s---
        array_like
            Scattered data.
        """
        return self.scatter_op(self.root, data, *scatter_data)
