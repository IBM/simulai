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
import sklearn.gaussian_process as gp
from scipy import interpolate


class GaussianRandomFields:
    def __init__(
        self, x_interval=(0, 1), kernel="RBF", length_scale=1, N=None, interp="cubic"
    ):
        """
        Initialize a Gaussian process object.

        Parameters
        ----------
        x_interval: tuple, optional (default=(0, 1))
            A tuple of two floats representing the range of the independent variable `x` over which the Gaussian process will be defined.
        kernel: str, optional (default='RBF')
            The type of kernel to use for the Gaussian process. Must be a string representing a kernel function available in `sklearn.gaussian_process.kernels`.
        length_scale: int or float, optional (default=1)
            The length scale parameter for the kernel function.
        N: int, optional (default=None)
            The number of points to sample from the independent variable `x`. If not provided, the number of points will be determined based on the `interp` parameter.
        interp: str, optional (default='cubic')
            The type of interpolation to use when generating points for the independent variable `x`. Must be a string representing a valid interpolation method.

        Attributes
        ----------
        x_interval: tuple
            The range of the independent variable `x` over which the Gaussian process will be defined.
        length_scale: int or float
            The length scale parameter for the kernel function.
        N: int
            The number of points to sample from the independent variable `x`.
        interp: str
            The type of interpolation to use when generating points for the independent variable `x`.
        tol: float
            A small positive number used to ensure positive definiteness of the covariance matrix.
        x: ndarray, shape (N, 1)
            An array of `N` points sampled from the independent variable `x` using the specified interpolation method.
        kernels: ndarray, shape (N, N)
            The covariance matrix for the Gaussian process, calculated using the specified kernel function and length scale.
        space: ndarray, shape (N, N)
            The Cholesky decomposition of the covariance matrix.
        """

        self.x_interval = x_interval
        self.length_scale = length_scale
        self.N = N
        self.interp = interp
        self.tol = 1e-13

        self.x = np.linspace(*self.x_interval, self.N)[:, None]

        kernels_function = getattr(gp.kernels, kernel, None)
        assert (
            kernels_function is not None
        ), f"The kernel {kernel} is not in sklearn.gaussian_process.kernels"

        kernels = kernels_function(length_scale=self.length_scale)
        self.kernels = kernels(self.x)

        self.space = np.linalg.cholesky(self.kernels + self.tol * np.eye(self.N))

    def random_u(self, n_features=None):
        """Generate random latent features using NumPy.

        Parameters
        ----------
        n_features : int, optional
            Number of latent features to generate. If None, defaults to the number of features in `self.space`.

        Returns
        -------
        u_ : ndarray, shape (n_features, N)
            Array of random latent features.
        """
        u_ = np.random.randn(self.N, n_features)
        return np.dot(self.space, u_).T

    def generate_u(self, features, sensors):
        """Generate latent features using NumPy and scipy.interpolate.

        Parameters
        ----------
        features : ndarray, shape (n_features, M)
            Array of input features.
        sensors : ndarray, shape (N, )
            Array of sensor locations.

        Returns
        -------
        u : ndarray, shape (n_features, N)
            Array of latent features.
        """
        values = map(
            lambda y: interpolate.interp1d(
                np.ravel(self.x), y, kind=self.interp, copy=False, assume_sorted=True
            )(sensors)[:, None],
            features,
        )

        return np.hstack(list(values))
