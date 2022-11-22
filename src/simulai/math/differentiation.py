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

from typing import Optional

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
    """Numerical time derivative based on centered differences.
    """
    def __init__(self):
        super().__init__()

    def __call__(self, u:np.ndarray=None, delta:float=1) -> np.ndarray:
        return np.gradient(u, delta)

# This is like an identifier
class SpaceDerivative(Derivative):
    def __init__(self):
        super().__init__()

class LeleDerivative:

    def __init__(self, N:int=1, h:float=1) -> None:

        """10th-order derivative algorithm purposed by Lele.

        :param N: number of mesh nodes
        :type param: int
        :param h: mesh size
        :type h: float

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

            LeftMatrix[j, j] = 1.

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

    def solve(self, f:np.ndarray) -> np.ndarray:

        """Performing Lele differentiation

        :param f: variable values to be differentiated
        :type f: np.ndarray
        :return: the derivatives of f
        :rtype: np.ndarray
        """

        b = self.RightMatrix @ f

        print("Performing Lele Derivation.")
        return spsolve(self.LeftMatrix, b)

    def delta_1(self, j:int, N:int) -> float:

        """term delta_1 from the Lele's expression

        :param j: node index
        :type j: int
        :param N: total number of nodes
        :type N: int
        :return: the evaluation of delta_1
        :rtype: float
        """

        if j == 0 or j == N - 1:
            return 2
        elif j == 1 or j == N - 2:
            return 1. / 4
        elif j == 2 or j == N - 3:
            return 4.7435 / 10.67175
        elif j == 3 or j == N - 4:
            return 4.63271875 / 9.38146875
        else:
            return 1. / 2

    def delta_2(self, j:int, N:int) -> float:

        """term delta_2 from the Lele's expression

        :param j: node index
        :type j: int
        :param N: total number of nodes
        :type N: int
        :return: the evaluation of delta_2
        :rtype: float
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
            return 1. / 20

    def delta_3(self, j:int, N:int, h:float) -> float:

        """term delta_3 from the Lele's expression

        :param j: node index
        :type j: int
        :param N: total number of nodes
        :type N: int
        :param h: mesh discretization size
        :type h: float
        :return: the evaluation of delta_3
        :rtype: float
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
            return 1. / 100 * (1 / (6 * h))

    def delta_4(self, j:int, N:int, h:float) -> float:

        """term delta_4 from the Lele's expression

        :param j: node index
        :type j: int
        :param N: total number of nodes
        :type N: int
        :param h: mesh discretization size
        :type h: float
        :return: the evaluation of delta_4
        :rtype: float
        """

        if j == 0:
            return 1. / (2 * h)
        if j == N - 1:
            return -1. / (2 * h)
        elif j == 1 or j == N - 2:
            return 0
        elif j == 2 or j == N - 3:
            return 4 * (1.23515625 / 10.67175) / (4 * h)
        elif j == 3 or j == N - 4:
            return 4 * (1.53 / 9.38146875) / (4 * h)
        else:
            return (101. / 105) * (1. / (4 * h))

    def delta_5(self, j:int, N:int, h:float) -> float:

        """term delta_5 from the Lele's expression

        :param j: node index
        :type j: int
        :param N: total number of nodes
        :type N: int
        :param h: mesh discretization size
        :type h: float
        :return: the evaluation of delta_5
        :rtype: float
        """

        if j == 0:
            return 2. / h
        elif j == N - 1:
            return -2. / h
        elif j == 1 or j == N - 2:
            return 3. / 2 * (1. / (2 * h))
        elif j == 2 or j == N - 3:
            return 2 * (7.905 / 10.67175) / (2 * h)
        elif j == 3 or j == N - 4:
            return 2 * (6.66984375 / 9.38146875) / (2 * h)
        else:
            return 17. / 12 * (1. / (2 * h))


    def delta_0(self, j:int, N:int, h:float) -> float:

        """term delta_0 from the Lele's expression

        :param j: node index
        :type j: int
        :param N: total number of nodes
        :type N: int
        :param h: mesh discretization size
        :type h: float
        :return: the evaluation of delta_0
        :rtype: float
        """

        if j == 0:
            return -5 / (2 * h)
        elif j == N - 1:
            return 5 / (2 * h)
        else:
            return 0

class CenteredDerivative:

    def __init__(self, config:dict=None) -> None:

        """Performing second-order centered differentiation

        :param config: Configuration dictionary
        :type config: dict
        :return: nothing
        """

        self.step = config['step']

        assert self.step, "A value for the differentiation step must be provided."

    def solve(self, data:np.ndarray=None, axis:int=0) -> np.ndarray:

        """It solves the centered derivative

        :param data: the values to be differentiated
        :type data: np.ndarray
        :param axis: axis to perform the differentiation
        :type axis: int
        :return: the derivatives of data
        :rtype: np.ndarray
        """

        print("Performing Second-Order Centered Derivation.")
        return np.gradient(data, self.step, axis=axis)

    def __call__(self, data:np.ndarray=None) -> np.ndarray:

        """__call__ wrapper for executing self.solve

        :param data: the values to be differentiated
        :type data: np.ndarray
        :return: the derivatives of data
        :rtype: np.ndarray
        """
        return self.solve(data=data)

# It interpolates the derivatives using splines and derive it
class CollocationDerivative:

    def __init__(self, config:dict=None) -> None:

        """Derivative using splines to interpolate the function to be differentiated

        :param config: a dictionary containing some parameters to be passed to ScipY engine
        :type config: dict
        :return: nothing
        """

        self.step = None
        self.t = None

        if 'step' in config.keys():
            self.step = config['step']

        self.original_shape = None


    def _guaratee_correct_shape(self, data:np.ndarray) -> np.ndarray:

        """This method ensures the input as one-dimensional arrays
           since the scipy.interpolate.InterpolatedUnivariateSpline class
           requires it.

        :param data: The input dataset to be differentiated
        :type data: np.ndarray
        :return: the input data properly reshaped (if necessary)
        :rtype: np.ndarray
        """

        if len(data.shape) <= 2:
            return data
        else:
            self.original_shape = data.shape
            immutable_dim = data.shape[0]
            collapsible_dims = data.shape[1:]
            collapsed_shape = np.prod(collapsible_dims)
            return data.reshape((immutable_dim, collapsed_shape))

    def solve(self, data:np.ndarray=None, x: Optional[np.ndarray]=None) -> np.ndarray:

        """Performing collocation differentiation

        :param data: values to be differentiated
        :type data: np.ndarray
        :param x: the axis to be used for executing the differentiation (Optional)
        :return: the derivatives of the input data along the chosen axis
        :rtype: np.ndarray
        """

        print("Performing Collocation Derivation.")

        intp_list = list()
        N = data.shape[0]

        if self.step:
            self.t = [self.step * ti for ti in range(N)]
        elif x is not None:
            self.t = x
        else:
            raise Exception("There is no way for evaluating derivatives.")

        data = self._guaratee_correct_shape(data)

        for i in range(data.shape[1]):

            intp = ius(self.t, data[:, i])
            dintp = intp.derivative()
            intp_list.append(dintp(self.t)[:, None])

        intp_array = np.hstack(intp_list)

        return intp_array.reshape(self.original_shape)

    def interpolate_and_solve(self, data:np.ndarray=None,
                                    x_grid:np.ndarray=None, x:np.ndarray=None) -> (np.ndarray, np.ndarray):

        """Performing collocation differentiation

        :param data: values to be differentiated
        :type data: np.ndarray
        :param x_grid: the grid in which the input is defined
        :type x_grid: np.ndarray
        :param x: the grid in which to interpolate the input
        :return: a pair (interpolated, derivated)
        :rtype: (np.ndarray, np.ndarray)
        """

        assert data.shape[0] <= x.shape[0], "In order to perform interpolation, it is necessary dim(x) > dim(data)."

        print("Performing Interpolation and Collocation Derivation.")

        intp_list = list()
        diff_intp_list = list()

        self.t = x_grid

        data = self._guaratee_correct_shape(data)

        for i in range(data.shape[1]):

            intp = ius(self.t, data[:, i])
            dintp = intp.derivative()

            intp_list.append(intp(x)[:, None])
            diff_intp_list.append(dintp(x)[:, None])

        intp_array = np.hstack(intp_list)
        diff_intp_array = np.hstack(diff_intp_list)

        if self.original_shape is not None:
            return intp_array.reshape((-1,) + self.original_shape[1:]), \
                   diff_intp_array.reshape((-1,) +  self.original_shape[1:])
        else:
            return intp_array.reshape(self.original_shape), diff_intp_array.reshape(self.original_shape)

    def __call__(self, data:np.ndarray=None) -> np.ndarray:

        """__call__ wrapper for executing self.solve

        :param data: the values to be differentiated
        :type data: np.ndarray
        :return: the derivatives of data
        :rtype: np.ndarray
        """

        derivative_data = self.solve(data=data)

        return derivative_data
