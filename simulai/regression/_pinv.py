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

from typing import List

import dask.array as da
import h5py
import numpy as np


class CompressedPinv:
    """
    Compressed Pseudo Inverse of a matrix using SVD (Singular Value Decomposition).

    Attributes:
        D (h5py.Dataset): input matrix
        k (int): number of singular values to consider when computing the pseudo inverse
        u (dask.array): left singular matrix
        s (dask.array): singular values
        v_star (dask.array): right singular matrix
        s_pinv (dask.array): inverse of the singular values
        u_star (dask.array): conjugate transpose of the left singular matrix
        v (dask.array): conjugate transpose of the right singular matrix
        v_bar (dask.array): v scaled by the inverse of s
        n_rows (int): number of rows in the left singular matrix

    Args:
        D (h5py.Dataset, optional): input matrix. Defaults to None.
        chunks (tuple, optional): chunks to break up the input matrix into. Defaults to None.
        k (int, optional): number of singular values to consider when computing the pseudo inverse. Defaults to 100.

    """

    def __init__(
        self, D: h5py.Dataset = None, chunks: tuple = None, k: int = 100
    ) -> None:
        """
        Initialize the CompressedPinv object.

        Args:
            D (h5py.Dataset, optional): input matrix. Defaults to None.
            chunks (tuple, optional): chunks to break up the input matrix into. Defaults to None.
            k (int, optional): number of singular values to consider when computing the pseudo inverse. Defaults to 100.

        """
        self.D = da.from_array(D, chunks=chunks)
        self.k = k
        self.u, self.s, self.v_star = da.linalg.svd_compressed(self.D, k=k)

        self.s_pinv = 1 / self.s.compute()
        self.u_star = self.u.conj().T
        self.v = self.v_star.conj().T

        self.v_bar = self.v * self.s_pinv

        self.n_rows = self.u_star.shape[0]

    def __call__(self, Y: h5py.Dataset = None, batches: List[slice] = None) -> None:
        """
        Compute the compressed pseudo inverse of the input matrix.

        Args:
            Y (h5py.Dataset, optional): input matrix. Defaults to None.
            batches (List[slice], optional): list of slices for the input matrix. Defaults to None.

        Returns:
            numpy.ndarray: the compressed pseudo inverse of the input matrix

        """
        output = np.zeros((self.n_rows, Y.shape[-1]))

        for batch in batches:
            output += self.u_star[:, batch].compute() @ Y[batch]

        return self.v_bar.compute() @ output
