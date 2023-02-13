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

from typing import Optional, Tuple

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as ius
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

"""differentiation.py

    This module contains derivative approximation algorithms
    commonly used in numerical analysis
"""


# This is like an identifier
class Derivative(object):
    """Parent class for the derivative operations
    It is used for identification purposes
    """

    def __init__(self):
        pass


# This is like an identifier
class TimeDerivative(Derivative):
    """Numerical time derivative based on centered differences."""

    def __init__(self):
        super().__init__()

    def __call__(self, u: np.ndarray = None, delta: float = 1) -> np.ndarray:
        return np.gradient(u, delta)


# This is like an identifier
class SpaceDerivative(Derivative):
    def __init__(self):
        super().__init__()


class LeleDerivative:
    def __init__(self, N: int = 1, h: float = 1) -> None:
        """
        Initialize the 10th-order derivative algorithm purposed by Lele.

        Parameters
        ----------
        N : int, optional
            The number of mesh nodes. The default value is 1.
        h : float, optional
            The mesh size. The default value is 1.
        """
        self.RightVector = np.array(N)

        # Constructing the left operator
        LeftMatrix = lil_matrix((N + 4, N + 4))
        RightMatrix = lil_matrix((N + 6, N + 6))

        for j in range(2, N + 2):
            i = j - 2
            delta_1 = self.delta_1(i, N)
            delta_2 = self.delta_2(i, N)

            LeftMatrix[j, j - 1] = delta_1
            LeftMatrix[j, j + 1] = delta_1

            LeftMatrix[j, j - 2] = delta_2
            LeftMatrix[j, j + 2] = delta_2

            LeftMatrix[j, j] = 1.0

        self.LeftMatrix = LeftMatrix[2:-2, 2:-2]

        for j in range(3, N + 3):
            i = j - 3
            delta_3 = self.delta_3(i, N, h)
            delta_4 = self.delta_4(i, N, h)
            delta_5 = self.delta_5(i, N, h)
            delta_0 = self.delta_0(i, N, h)

            RightMatrix[j, j - 1] = -delta_5
            RightMatrix[j, j + 1] = delta_5

            RightMatrix[j, j - 2] = -delta_4
            RightMatrix[j, j + 2] = delta_4

            RightMatrix[j, j - 3] = -delta_3
            RightMatrix[j, j + 3] = delta_3

            RightMatrix[j, j] = delta_0

        self.RightMatrix = RightMatrix[3:-3, 3:-3]

    def solve(self, f: np.ndarray) -> np.ndarray:
        """
        Perform Lele differentiation.

        Parameters
        ----------
        f : np.ndarray
            The variable values to be differentiated.

        Returns
        -------
        np.ndarray
            The derivatives of f.
        """

        b = self.RightMatrix @ f

        print("Performing Lele Derivation.")
        return spsolve(self.LeftMatrix, b)

    def delta_1(self, j: int, N: int) -> float:
        """
        Calculate the term delta_1 from the Lele's expression.

        Parameters
        ----------
        j : int
            The node index.
        N : int
            The total number of nodes.

        Returns
        -------
        float
            The evaluation of delta_1.
        """

        if j == 0 or j == N - 1:
            return 2
        elif j == 1 or j == N - 2:
            return 1.0 / 4
        elif j == 2 or j == N - 3:
            return 4.7435 / 10.67175
        elif j == 3 or j == N - 4:
            return 4.63271875 / 9.38146875
        else:
            return 1.0 / 2

    def delta_2(self, j: int, N: int) -> float:
        """
        Calculate the term delta_2 from the Lele's expression.

        Parameters
        ----------
        j : int
            The node index.
        N : int
            The total number of nodes.

        Returns
        -------
        float
            The evaluation of delta_2.
        """

        if j == 0 or j == N - 1:
            return 0
        elif j == 1 or j == N - 2:
            return 0
        elif j == 2 or j == N - 3:
            return 0.2964375 / 10.67175
        elif j == 3 or j == N - 4:
            return 0.451390625 / 9.38146875
        else:
            return 1.0 / 20

    def delta_3(self, j: int, N: int, h: float) -> float:
        """
        Calculate the term delta_3 from the Lele's expression.

        Parameters
        ----------
        j : int
            The node index.
        N : int
            The total number of nodes.
        h : float
            The mesh discretization size.

        Returns
        -------
        float
            The evaluation of delta_3.
        """

        if j == 0 or j == N - 1:
            return 0
        elif j == 1 or j == N - 2:
            return 0
        elif j == 2 or j == N - 3:
            return 0
        elif j == 3 or j == N - 4:
            return 6 * (0.015 / 9.38146875) / (6 * h)
        else:
            return 1.0 / 100 * (1 / (6 * h))

    def delta_4(self, j: int, N: int, h: float) -> float:
        """
        Term delta_4 from the Lele's expression

        Parameters
        ----------
        j : int
            The node index.
        N : int
            The total number of nodes.
        h : float
            The mesh discretization size.

        Returns
        -------
        float
            The evaluation of delta_4.
        """

        if j == 0:
            return 1.0 / (2 * h)
        if j == N - 1:
            return -1.0 / (2 * h)
        elif j == 1 or j == N - 2:
            return 0
        elif j == 2 or j == N - 3:
            return 4 * (1.23515625 / 10.67175) / (4 * h)
        elif j == 3 or j == N - 4:
            return 4 * (1.53 / 9.38146875) / (4 * h)
        else:
            return (101.0 / 105) * (1.0 / (4 * h))

    def delta_5(self, j: int, N: int, h: float) -> float:
        """
        Term delta_5 from the Lele's expression

        Parameters
        ----------
        j : int
            The node index.
        N : int
            The total number of nodes.
        h : float
            The mesh discretization size.

        Returns
        -------
        float
            The evaluation of delta_5.
        """

        if j == 0:
            return 2.0 / h
        elif j == N - 1:
            return -2.0 / h
        elif j == 1 or j == N - 2:
            return 3.0 / 2 * (1.0 / (2 * h))
        elif j == 2 or j == N - 3:
            return 2 * (7.905 / 10.67175) / (2 * h)
        elif j == 3 or j == N - 4:
            return 2 * (6.66984375 / 9.38146875) / (2 * h)
        else:
            return 17.0 / 12 * (1.0 / (2 * h))

    def delta_0(self, j: int, N: int, h: float) -> float:
        """
        Compute the term delta_0 from the Lele's expression.

        Parameters
        ----------
        j : int
            Node index.
        N : int
            Total number of nodes.
        h : float
            Mesh discretization size.

        Returns
        -------
        float
            The evaluation of delta_0.

        Examples
        --------
        >>> delta_0(0, 10, 0.1)
        -5.0
        >>> delta_0(9, 10, 0.1)
        5.0
        >>> delta_0(5, 10, 0.1)
        0.0
        """
        if j == 0:
            return -5 / (2 * h)
        elif j == N - 1:
            return 5 / (2 * h)
        else:
            return 0


class CenteredDerivative:
    def __init__(self, config: dict = None) -> None:
        """
        Initializes the object for performing second-order centered differentiation.

        Parameters
        ----------
        config : dict, optional
            Configuration dictionary. If not provided, default value is None.

        Returns
        -------
        None
        """
        self.step = config["step"]
        assert self.step, "A value for the differentiation step must be provided."

    def solve(self, data: np.ndarray = None, axis: int = 0) -> np.ndarray:
        """
        Computes the centered derivative of the input data.

        Parameters
        ----------
        data : np.ndarray, optional
            The values to be differentiated. If not provided, default value is None.
        axis : int, optional
            The axis along which the derivative is taken. Default is 0.

        Returns
        -------
        np.ndarray
            The derivatives of data.
        """
        print("Performing Second-Order Centered Derivation.")
        return np.gradient(data, self.step, axis=axis)

    def __call__(self, data: np.ndarray = None) -> np.ndarray:
        """
        Wrapper function for executing self.solve.

        Parameters
        ----------
        data : np.ndarray, optional
            The values to be differentiated. If not provided, default value is None.

        Returns
        -------
        np.ndarray
            The derivatives of data.
        """
        return self.solve(data=data)


# It interpolates the derivatives using splines and derive it
class CollocationDerivative:
    def __init__(self, config: dict = None, k: int = 3) -> None:
        """Initialize a derivative calculator using spline interpolation.

        Parameters
        ----------
        config : dict, optional
            A dictionary containing some parameters to be passed to the Scipy engine.
        k : int, optional
            The degree of the spline interpolation.
        """
        self.step = None
        self.t = None
        self.k = k

        if "step" in config.keys():
            self.step = config["step"]

        self.original_shape = None

    @staticmethod
    def _guarantee_correct_shape(data: np.ndarray) -> np.ndarray:
        """Ensure that the input data is a one-dimensional array.

        This method ensures that the input data is a one-dimensional array, since the scipy.interpolate.InterpolatedUnivariateSpline class requires it. If the input data is already one-dimensional, it is returned as is. If the input data has more than one dimension, it is reshaped into a one-dimensional array by collapsing all dimensions except the first one.

        Parameters:
        ----------
            data (np.ndarray): The input data to be reshaped.

        Returns:
        ----------
            The input data (np.ndarray), reshaped as a one-dimensional array if necessary.
        """
        if data.ndim <= 1:
            return data
        else:
            immutable_dim = data.shape[0]
            collapsible_dims = data.shape[1:]
            collapsed_shape = np.prod(collapsible_dims)
            return data.reshape((immutable_dim, collapsed_shape))

    def solve(self, data: np.ndarray, x: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Perform differentiation using a non-linear method.

        Parameters:
        ----------
            data (np.ndarray): The values to be differentiated.
            x (np.ndarray, optional): The axis to be used for executing the differentiation. If not provided, the 'step' attribute of the object will be used.

        Returns:
        ----------
            The derivatives of the input data along the chosen axis.

        Raises:
        ----------
            Exception: If there is no way to evaluate the derivatives (i.e., 'x' and 'step' attributes are not provided).
        """
        print("Performing Collocation Derivation.")

        intp_list = []
        N = data.shape[0]

        if self.step:
            self.t = [self.step * ti for ti in range(N)]
        elif x is not None:
            self.t = x
        else:
            raise Exception("There is no way for evaluating derivatives.")

        data = self._guarantee_correct_shape(data)

        for i in range(data.shape[1]):
            # Interpolate data using a non-linear method
            interpolation = ius(self.t, data[:, i], k=self.k)
            # Differentiate interpolated data
            differentiated_interpolation = interpolation.derivative()
            # Append differentiated data to list
            intp_list.append(differentiated_interpolation(self.t)[:, None])

        intp_arr = np.hstack(intp_list)

        return intp_arr.reshape(self.original_shape)

    def interpolate_and_solve(
        self, data: np.ndarray = None, x_grid: np.ndarray = None, x: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform interpolation and differentiation using a non-linear method.

        Parameters:
        ----------
            data (np.ndarray): The values to be interpolated and differentiated.
            x_grid (np.ndarray): The grid in which the input data is defined.
            x (np.ndarray): The grid in which to interpolate the input data.

        Returns:
        ----------
            A tuple containing the interpolated and differentiated data.

        Raises:
        ----------
            AssertionError: If the length of the data array is greater than the length of the x array.
        """
        if data.shape[0] > x.shape[0]:
            raise AssertionError(
                "In order to perform interpolation, it is necessary dim(x) > dim(data)."
            )

        print("Performing Interpolation and Collocation Derivation.")

        self.t = x_grid
        data = self._guarantee_correct_shape(data)
        interpolated_data_list = []
        differentiated_data_list = []

        for i in range(data.shape[1]):
            # Interpolate data using a non-linear method
            interpolation = ius(self.t, data[:, i], k=self.k)
            # Differentiate interpolated data
            differentiated_interpolation = interpolation.derivative()
            # Append interpolated and differentiated data to lists
            interpolated_data_list.append(interpolation(x)[:, None])
            differentiated_data_list.append(differentiated_interpolation(x)[:, None])

        # Concatenate interpolated and differentiated data lists into arrays
        interpolated_data_array = np.hstack(interpolated_data_list)
        differentiated_data_array = np.hstack(differentiated_data_list)

        if self.original_shape is not None:
            return interpolated_data_array.reshape(
                (-1,) + self.original_shape[1:]
            ), differentiated_data_array.reshape((-1,) + self.original_shape[1:])
        else:
            return interpolated_data_array.reshape(
                self.original_shape
            ), differentiated_data_array.reshape(self.original_shape)

    def __call__(self, data: np.ndarray = None) -> np.ndarray:
        """
        Wrapper for executing self.solve.

        Parameters
        ----------
        data : np.ndarray, optional
            The values to be differentiated. The default value is None.

        Returns
        -------
        np.ndarray
            The derivatives of data.
        """

        derivative_data = self.solve(data=data)

        return derivative_data
