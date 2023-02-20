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

from typing import Tuple, Union

import numexpr as ne
import numpy as np
import scipy as sp
import scipy.spatial as spatial
import sympy as sy
from scipy.sparse import csc_matrix


class Kansas:
    def __init__(
        self,
        points: np.ndarray = None,
        centers: np.ndarray = None,
        sigma2: Union[str, float] = "auto",
        kernel: str = "gaussian",
        eps: float = 1e-8,
    ) -> None:
        """
        Initialize the radial basis function interpolator.

        Parameters
        ----------
        points : np.ndarray
            The points at which to interpolate.
        centers : np.ndarray
            The centers of the kernel functions.
        sigma2 : Union[str, float], optional
            The variance of the kernel functions. If set to "auto", the variance is calculated from the maximum distance
            between centers.
        kernel : str, optional
            The type of kernel function to use.
        eps : float, optional
            The tolerance for eliminating elements from the interpolation matrices.

        """

        self.Nk = centers.shape[0]  # number of centers of kernels
        self.kernel = (
            kernel  # kernel function or string specifiying the type of kernel function
        )
        self.eps = eps  # tolerance to take off interpolation matrices elements
        self.points = (points).astype("float32")  # interpolation points
        self.centers = (centers).astype("float32")  # kernel centers
        self.ndim = centers.shape[1]  # dimension of points

        if self.ndim != points.shape[1]:
            print("mismatch dimension of kernel centers and points")

        self.f1_was_gen = False

        # Calculate the radial function for the combination of centers and interpolation points
        self.r2 = (spatial.distance.cdist(points, centers, "sqeuclidean")).astype(
            "float32"
        )

        # calculate sigma based on the mean maximal distance beteewn 2 kernels or use the informed sigma
        if sigma2 == "auto":
            d2 = spatial.distance.cdist(centers, centers, "sqeuclidean")

            d2Max = np.max(d2)

            self.sigma2 = d2Max / (2 * self.Nk)

        else:
            self.sigma2 = sigma2

        self.use_optimized = False
        self.kernel_list = [
            "gaussian",
            "MQ",
            "IMQ",
        ]  # declare optimized implemented kernels

        if self.kernel in self.kernel_list:
            self.use_optimized = True

            # initialize interpolation matrices lists
            # self.Dx = [None] * self.ndim
            # self.Dxx = [None] * self.ndim

        else:
            self.x = sy.symbols("x")
            x = self.x
            sigma2 = self.sigma2
            self.expr = eval(self.kernel)
        return

    def get_interpolation_matrix(self) -> None:
        """
        Calculates and returns the interpolation matrix for the given set of points. If `self.use_optimized` is True,
        the optimized version of the interpolation matrix is calculated and returned. Otherwise, the interpolation matrix
        is calculated using a lambda function that evaluates the kernel at every point on the radial function, and then
        returning the resulting matrix after setting all elements with absolute value less than `self.eps` to 0.0 and
        converting the matrix to a sparse CSC format.
        """
        if self.use_optimized:
            G = self.get_interpolation_matrix_optimized()
        else:
            g = sy.lambdify(
                self.x, self.expr, "numpy"
            )  # create element to element function for evalueate kernel at every point on the radial function

            G = g(self.r2)  # evaluate radial function on the kernel

        G[np.abs(G) < self.eps] = 0.0  # take off elements to an specified tolerance
        G = sp.sparse.csc_matrix(G)  # transform in csc matrix

        return G

    def get_interpolation_matrix_optimized(self) -> np.ndarray:
        """
        Calculate the interpolation matrix using an optimized method.

        Returns:
        ----------
            G : ndarray
            The interpolation matrix.
        """
        G = self.Kernel(self.r2, self.sigma2, self.kernel)

        return G

    def get_first_derivative_matrix(self, var_index: int) -> np.ndarray:
        """
        Calculate the first derivative matrix of a specific variable.

        Parameters
        ----------
            var_index : int
            The index of the variable for which to calculate the derivative matrix.

        Returns:
        ----------

            ndarray: The first derivative matrix.
        """
        if var_index > self.ndim:
            print("index of variable are higher than ")

        if self.use_optimized:
            Dx = self.get_first_derivative_matrix_optimized(var_index)
        else:
            Dx, _ = self.get_first_derivative_matrix_aux(var_index)

        return Dx

    def get_first_derivative_matrix_optimized(
        self, var_index: int = None
    ) -> csc_matrix:
        """
        Compute the first derivative matrix of the kernel function with respect to the variable at the given index.

        Parameters
        ----------
        var_index : int
            The index of the variable to compute the first derivative with respect to.

        Returns
        -------
        Dx : scipy.sparse.csc_matrix
            The first derivative matrix.

        """

        # rx = sp.spatial.distance.cdist(self.points[:, [var_index]], self.centers[:, [var_index]], lambda u, v: (u - v))

        rx = self.points[:, [var_index]] - self.centers[:, var_index]

        Dx = self.kernel_Dx(self.r2, rx, self.sigma2, self.kernel)

        Dx[np.abs(Dx) < self.eps] = 0.0

        Dx = sp.sparse.csc_matrix(Dx)

        return Dx

    def get_first_derivative_matrix_aux(
        self, var_index: int = None
    ) -> Tuple[csc_matrix, np.ndarray]:
        """
        Compute the auxiliary matrices needed to compute the first derivative matrix of the kernel function with respect to the variable at the given index.

        Parameters
        ----------
        var_index : int
            The index of the variable to compute the first derivative with respect to.

        Returns
        -------
        Dx : scipy.sparse.csc_matrix
            The first derivative matrix.
        dr2dx : numpy.ndarray
            The auxiliary array needed to compute the first derivative matrix.

        """
        self.gen_f1()

        # dr2dx = sp.spatial.distance.cdist(self.points[:, [var_index]], self.centers[:, [var_index]], lambda u, v: 2*(u - v))

        dr2dx = 2 * (self.points[:, [var_index]] - self.centers[:, var_index])

        # if self.Dx[var_index] == None:

        Dx = self.f1 * dr2dx

        Dx[np.abs(Dx) < self.eps] = 0.0

        Dx = sp.sparse.csc_matrix(Dx)

        # self.Dx[var_index] = Dx

        return Dx, dr2dx

    def get_cross_derivative_matrix(
        self, var_index1: int = None, var_index2: int = None
    ) -> csc_matrix:
        """
        Compute the cross derivative matrix of the kernel function with respect to the variables at the given indices.

        Parameters
        ----------
        var_index1 : int
            The index of the first variable to compute the cross derivative with respect to.
        var_index2 : int
            The index of the second variable to compute the cross derivative with respect to.

        Returns
        -------
        Dxy : scipy.sparse.csc_matrix
            The cross derivative matrix.

        """
        if var_index1 > self.ndim or var_index2 > self.ndim:
            print("index of variable are higher than dimension")

        if self.use_optimized:
            Dxy = self.get_cross_derivative_matrix_optimized(var_index1, var_index2)

        else:
            d2expr = self.expr.diff(self.x, 2)

            d2g = sy.lambdify(self.x, d2expr, "numpy")

            # if self.Dx[var_index]== None :
            _, dr2dx = self.get_first_derivative_matrix_aux(var_index1)
            _, dr2dy = self.get_first_derivative_matrix_aux(var_index2)

            Dxy = d2g(self.r2) * (dr2dx * dr2dy) + 2 * self.f1

        Dxy[np.abs(Dxy) < self.eps] = 0.0

        Dxy = sp.sparse.csc_matrix(Dxy)

        # self.Dxx[var_index] = Dxx

        return Dxy

    def get_cross_derivative_matrix_optimized(
        self, var_index1: int = None, var_index2: int = None
    ) -> np.ndarray:
        """
        Compute the cross derivative matrix of the kernel function with respect to the variables at the given indices using an optimized method.

        Parameters
        ----------
        var_index1 : int
            The index of the first variable to compute the cross derivative with respect to.
        var_index2 : int
            The index of the second variable to compute the cross derivative with respect to.

        Returns
        -------
        Dxy : numpy.ndarray
            The cross derivative matrix.

        """
        rx = self.points[:, [var_index1]] - self.centers[:, var_index1]

        ry = self.points[:, [var_index2]] - self.centers[:, var_index2]

        Dxy = self.kernel_Dxy(self.r2, rx, ry, self.sigma2, self.kernel)

        return Dxy

    def get_second_derivative_matrix(self, var_index: int) -> csc_matrix:
        """
        Compute the second derivative matrix of the kernel function with respect to the variable at the given index.

        Parameters
        ----------
        var_index : int
            The index of the variable to compute the second derivative with respect to.

        Returns
        -------
        Dxx : scipy.sparse.csc_matrix
            The second derivative matrix.

        """
        if var_index > self.ndim:
            print("index of variable are higher than dimension")

        if self.use_optimized:
            Dxx = self.get_second_derivative_matrix_optimized(var_index)

        else:
            d2expr = self.expr.diff(self.x, 2)

            d2g = sy.lambdify(self.x, d2expr, "numpy")

            # if self.Dx[var_index]== None :
            _, dr2dx = self.get_first_derivative_matrix_aux(var_index)

            Dxx = d2g(self.r2) * (dr2dx**2) + 2 * self.f1

        Dxx[np.abs(Dxx) < self.eps] = 0.0

        Dxx = sp.sparse.csc_matrix(Dxx)

        # self.Dxx[var_index] = Dxx

        return Dxx

    def get_second_derivative_matrix_optimized(self, var_index: int) -> np.ndarray:
        """
        Compute the second derivative matrix of the kernel function with respect to the variable at the given index using an optimized method.

        Parameters
        ----------
        var_index : int
            The index of the variable to compute the second derivative with respect to.

        Returns
        -------
        Dxx : numpy.ndarray
            The second derivative matrix.

        """
        # rx2 = sp.spatial.distance.cdist(self.points[:, [var_index]], self.centers[:, [var_index]], lambda u, v: (u - v)**2)

        rx2 = (self.points[:, [var_index]] - self.centers[:, var_index]) ** 2

        Dxx = self.kernel_Dxx(self.r2, rx2, self.sigma2, self.kernel)

        return Dxx

    def get_laplacian_matrix(self) -> csc_matrix:
        """
        Compute the Laplacian matrix of the kernel function.

        Returns
        -------
        L : scipy.sparse.csc_matrix
            The Laplacian matrix.

        """
        if self.use_optimized:
            L = self.kernel_Laplacian(self.r2, self.sigma2, self.kernel)
        else:
            d2expr = self.expr.diff(self.x, 2)

            d2g = sy.lambdify(self.x, d2expr, "numpy")

            self.gen_f1()

            L = d2g(self.r2) * (4 * self.r2) + 2 * self.ndim * self.f1

        L[np.abs(L) < self.eps] = 0.0
        L = sp.sparse.csc_matrix(L)

        return L

    def gen_f1(self) -> None:
        """
        Generate the first derivative of the kernel function.

        """
        if self.f1_was_gen == False:
            d1expr = self.expr.diff(self.x)

            dg = sy.lambdify(self.x, d1expr, "numpy")

            self.f1 = dg(self.r2)

            self.f1_was_gen = True

        return

    def Kernel(
        self, r2: np.ndarray = None, sigma2: float = None, kernel_type: str = None
    ) -> np.ndarray:
        """
        Compute the kernel function.

        Parameters
        ----------
        r2 : numpy.ndarray
            The array of squared distances.
        sigma2 : float
            The variance parameter of the kernel function.
        kernel_type : str
            The type of kernel function to use. Can be "gaussian", "MQ", or "IMQ".

        Returns
        -------
        G : numpy.ndarray
            The kernel function evaluated at the given squared distances.
        """

        if kernel_type == "gaussian":
            G = ne.evaluate("exp(-r2/(2.0*sigma2))")

            # G = np.exp(-r2/(2.0*sigma2))

        elif kernel_type == "MQ":
            G = ne.evaluate("sqrt(r2 + sigma2)")

        elif kernel_type == "IMQ":
            G = ne.evaluate("1.0/sqrt(r2 + sigma2)")

        else:
            print(" this kernel does not exist: ", kernel_type)

        G = (G).astype("float32")

        return G

    def kernel_Dx(
        self,
        r2: float = None,
        rx: float = None,
        sigma2: float = None,
        kernel_type: str = None,
    ) -> float:
        """
        Calculate the derivative of a kernel function with respect to a distance value.

        Parameters
        ----------
        r2 : float
            Squared distance between two points.
        rx : float
            Distance between two points in a single dimension.
        sigma2 : float
            Hyperparameter for the kernel function.
        kernel_type : str
            Type of kernel to use. Must be one of 'gaussian', 'MQ', or 'IMQ'.

        Returns
        -------
        Dx : float
            Derivative of the kernel function with respect to `rx`.

        """

        if kernel_type == "gaussian":
            Dx = ne.evaluate("-(rx/sigma2)*exp(-r2/(2.0*sigma2))")

            # Dx = -(rx/sigma2)*np.exp(-r2/(2.0*sigma2))

        elif kernel_type == "MQ":
            Dx = ne.evaluate("rx/sqrt(r2 + sigma2)")

        elif kernel_type == "IMQ":
            Dx = ne.evaluate("-rx*((r2 + sigma2)**(-1.5))")

        else:
            print(" this kernel does not exist: ", kernel_type)

        Dx = (Dx).astype("float32")

        return Dx

    def kernel_Dxy(
        self,
        r2: float = None,
        rx: float = None,
        ry: float = None,
        sigma2: float = None,
        kernel_type: str = None,
    ) -> float:
        """
        Calculate the mixed second partial derivative of a kernel function.

        Parameters
        ----------
        r2 : float
            Squared distance between two points.
        rx : float
            Distance between two points in a single dimension.
        ry : float
            Distance between two points in a single dimension.
        sigma2 : float
            Hyperparameter for the kernel function.
        kernel_type : str
            Type of kernel to use. Must be one of 'gaussian', 'MQ', or 'IMQ'.

        Returns
        -------
        Dxy : float
            Mixed second partial derivative of the kernel function with respect to `rx` and `ry`.
        """
        if kernel_type == "gaussian":
            Dxy = ne.evaluate("((rx*ry)/(sigma2**2))*exp(-r2/(2*sigma2))")

        elif kernel_type == "MQ":
            Dxy = ne.evaluate("3.0*rx*ry*((r2 + sigma2)**(-2.5))")

        elif kernel_type == "IMQ":
            Dxy = ne.evaluate("5.0*rx*ry*((r2 + sigma2)**(-3.5))")

        else:
            print(" this kernel does not exist: ", kernel_type)

        Dxy = (Dxy).astype("float32")

        return Dxy

    def kernel_Dxx(
        self,
        r2: float = None,
        rx2: float = None,
        sigma2: float = None,
        kernel_type: str = None,
    ) -> float:
        """
        Calculate the Dxx value for a given kernel type.

        Parameters
        ----------
        r2 : float
            The square of the distance between two points.
        rx2 : float
            The square of the distance between two points in the x direction.
        sigma2 : float
            The square of the kernel width.
        kernel_type : str
            The type of kernel to use. Can be either "gaussian", "MQ", or "IMQ".

        Returns
        -------
        Dxx : float
            The Dxx value for the given kernel type.

        """
        if kernel_type == "gaussian":
            Dxx = ne.evaluate(
                "((rx2/(sigma2**2)) - (1.0/sigma2) )*exp(-r2/(2.0*sigma2))"
            )

        elif kernel_type == "MQ":
            Dxx = ne.evaluate("(1.0/sqrt(r2 + sigma2))-rx2*((r2 + sigma2)**(-1.5))")
        elif kernel_type == "IMQ":
            Dxx = ne.evaluate(
                "-((r2 + sigma2)**(-1.5))+3.0*rx2*((r2 + sigma2)**(-2.5))"
            )
        else:
            print(" thins kernel does not exist: ", kernel_type)

        Dxx = (Dxx).astype("float32")

        return Dxx

    def kernel_Laplacian(
        self, r2: float = None, sigma2: float = None, kernel_type: str = None
    ) -> float:
        """
        Calculate the Laplacian value for a given kernel type.

        Parameters
        ----------
        r2 : float
            The square of the distance between two points.
        sigma2 : float
            The square of the kernel width.
        kernel_type : str
            The type of kernel to use. Can be either "gaussian", "MQ", or "IMQ".

        Returns
        -------
        L : float
            The Laplacian value for the given kernel type.

        """
        ndim = float(self.ndim)

        if kernel_type == "gaussian":
            L = ne.evaluate(
                "((r2/(sigma2**2.0)) - (ndim/sigma2) )*exp(-r2/(2.0*sigma2))"
            )
            # L =((r2 / (sigma2 ** 2.0)) - (ndim / sigma2)) * np.exp(-r2 / (2.0 * sigma2))

        elif kernel_type == "MQ":
            L = ne.evaluate("(ndim/sqrt(r2 + sigma2))-r2*((r2 + sigma2)**(-1.5))")
            # L = 2/np.sqrt(r2 + (2 * sigma2)) - r2*np.power(r2 + (2 * sigma2),1.5)

        elif kernel_type == "IMQ":
            L = ne.evaluate(
                "-ndim*((r2 + sigma2)**(-1.5))+3.0*r2*((r2 + sigma2)**(-2.5))"
            )
            # L = -2*np.power(r2 + (2 * sigma2),-1.5)+3*r2*np.power(r2 + (2 * sigma2),-2.5)
        else:
            print(" thins kernel does not exist: ", kernel_type)

        L = (L).astype("float32")

        return L
