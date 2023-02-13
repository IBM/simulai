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

import os
import pickle
import sys
import warnings
from copy import deepcopy
from typing import List, Optional, Union

import h5py
import numpy as np
import psutil
from scipy.linalg import solve as solve

MPI_GLOBAL_AVAILABILITY = True

try:
    from mpi4py import MPI
except:
    MPI_GLOBAL_AVAILABILITY = False
    warnings.warn(f"Trying to import MPI in {__file__}.")
    warnings.warn(
        "mpi4py is not installed. If you want to execute MPI jobs, we recommend you install it."
    )

from simulai.parallel import PipelineMPI

from ._pinv import CompressedPinv


# Operator inference using quadratic non-linearity
class OpInf:
    def __init__(
        self,
        forcing: str = None,
        bias_rescale: float = 1,
        solver: Union[str, callable] = "lstsq",
        parallel: Union[str, None] = None,
        show_log: bool = False,
        engine: str = "numpy",
    ) -> None:
        """Operator Inference (OpInf)

        :param forcing: the kind of forcing to be used, 'linear' or 'nonlinear'
        :type forcing: str
        :param bias_rescale: factor for rescaling the linear coefficients (c_hat)
        :type bias_rescale: float
        :param solver: solver to be used for solving the global system, e. g. 'lstsq'.
        :type solver: Union[str, callable]
        :param parallel: the kind of parallelism to be used (currently, 'mpi' or None)
        :type parallel: str
        :param engine: the engine to be used for constructing the global system (currently just 'numpy')
        :type engine: str
        :return: nothing
        """

        # forcing is chosen among (None, 'linear', 'nonlinear')
        self.forcing = forcing
        self.bias_rescale = bias_rescale
        self.solver = solver
        self.parallel = parallel
        self.show_log = show_log
        self.engine = engine

        if self.forcing is not None:
            self.eval_op = self._eval_forcing
        else:
            self.eval_op = self._eval

        if self.forcing == "nonlinear":
            self.kronecker_product = self._augmented_kronecker_product
        else:
            self.kronecker_product = self._simple_kronecker_product

        if self.parallel == None:
            self.dispatcher = self._serial_operators_construction_dispatcher
        elif self.parallel == "mpi":
            if MPI_GLOBAL_AVAILABILITY == True:
                self.dispatcher = self._parallel_operators_construction_dispatcher
            else:
                raise Exception(
                    "Trying to execute a MPI job but there is no MPI distribution available."
                )
        else:
            raise Exception(
                f"The option {self.parallel} for parallel is not valid. It must be None or mpi"
            )

        self.lambda_linear = 0
        self.lambda_quadratic = 0

        self.n_inputs = None
        self.n_outputs = None
        self.n_samples = None
        self.n_quadratic_inputs = None
        self.n_forcing_inputs = 0

        self.jacobian = None
        self.jacobian_op = None

        self.D_o = None
        self.R_matrix = None

        # OpInf adjustable operators
        self.c_hat = None  # Bias
        self.A_hat = None  # Coefficients for the linear field variable terms
        self.H_hat = None  # Coefficients for the nonlinear quadratic terms
        self.B_hat = None  # Coefficients for the linear forcing terms

        self.success = None
        self.continuing = 1

        self.raw_model = True
        self.tmp_data_path = "/tmp"

    # Matrix containing all the model parameters
    @property
    def O_hat(self) -> np.ndarray:
        """The concatenation of all the coefficients matrices"""

        valid = [
            m for m in [self.c_hat, self.A_hat, self.H_hat, self.B_hat] if m is not None
        ]

        return np.hstack(valid)

    @property
    def D_matrix_dim(self) -> np.ndarray:
        """The dimension of the data matrix"""

        return np.array([self.n_samples, self.n_linear_terms + self.n_quadratic_inputs])

    @property
    def Res_matrix_dim(self) -> np.ndarray:
        """The dimension of the right-hand side residual matrix"""

        return np.array([self.n_samples, self.n_outputs])

    @property
    def m_indices(self) -> list:
        """Indices for the non-repeated observables in the Kronecker
        product output
        """
        return np.vstack([self.i_u, self.j_u]).T.tolist()

    @property
    def solver_nature(self) -> str:
        """It classifies the solver used
           in 'lazy' (when data is stored on disk)
           and 'memory' (when data is all allocated in memory)

        :return: the solver classification
        :rtype:str
        """

        if self.solver == "pinv":
            return "lazy"
        else:
            return "memory"

    # Splitting the global solution into corresponding operators
    def set_operators(self, global_matrix: np.ndarray = None) -> None:
        """Setting up each operator using the global system solution

        :param global_matrix: the solution of the global system
        :type global_matrix: np.ndarray
        :return: nothing
        """
        if self.n_inputs == None and self.n_outputs == None:
            self.n_inputs = self.n_outputs = global_matrix.shape[1]

        if self.raw_model == True:
            self.construct()

        if self.forcing is not None:
            self.c_hat = global_matrix[:1].T
            self.A_hat = global_matrix[1 : self.n_inputs + 1].T
            self.B_hat = global_matrix[
                self.n_inputs + 1 : self.n_inputs + 1 + self.n_forcing_inputs
            ].T
            self.H_hat = global_matrix[self.n_inputs + 1 + self.n_forcing_inputs :].T
        else:
            self.c_hat = global_matrix[:1].T
            self.A_hat = global_matrix[1 : self.n_inputs + 1].T
            self.H_hat = global_matrix[self.n_inputs + 1 :].T

    # Setting up model parameters
    def set(self, **kwargs):
        """Setting up extra parameters (as regularization terms)

        :param kwargs: dictionary containing extra parameters
        :type kwargs:dict
        :return: nothing
        """

        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def check_fits_in_memory(self) -> str:
        """It checks if the data matrices, D and Res_matrix, can fit on memory

        :return: the method for dealing with the data matrix, 'batch-wise' or 'global'
        :rtype: str
        """

        total_size = np.prod(self.D_matrix_dim) + np.prod(self.Res_matrix_dim)
        item_size = np.array([0]).astype("float64").itemsize

        allocated_memory = total_size * item_size
        available_memory = psutil.virtual_memory().available

        if allocated_memory >= available_memory:
            print("The data matrices does not fit in memory. Using batchwise process.")
            return "batchwise"
        else:
            print("The data matrices fits in memory.")
            return "global"

    # It checks if a matrix is symmetric
    def _is_symmetric(self, matrix: np.ndarray = None) -> bool:
        """It checks if the system matrix is symmetric

        :param matrix: the global system matrix
        :type matrix: np.ndarray
        :return: Is the matrix symmetric ? True or False
        :rtype: bool
        """

        return np.array_equal(matrix, matrix.T)

    def _kronecker_product(
        self, a: np.ndarray = None, b: np.ndarray = None
    ) -> np.ndarray:
        """Kronecker product between two arrays

        :param a: first element of the Kronecker product
        :type a: np.ndarray
        :param b: second element of the Kronecker product
        :type b: np.ndarray
        :return: the result of the kronecker product
        :rtype: np.ndarray
        """

        assert (
            a.shape == b.shape
        ), f"a and b must have the same shape, but received {a.shape} and {b.shape}"

        kron_output = np.einsum("bi, bj->bij", a, b)

        assert (
            np.isnan(kron_output).max() == False
        ), "There are NaN in the Kronecker output"

        # Checking if the Kronecker output tensor is symmetric or not
        if np.array_equal(kron_output, kron_output.transpose(0, 2, 1)):
            return kron_output[:, self.i_u, self.j_u]
        else:
            shapes = kron_output.shape[1:]
            return kron_output.reshape(-1, np.prod(shapes))

    # Kronecker product augmented using extra variables (such as forcing terms)
    def _augmented_kronecker_product(
        self, a: np.ndarray = None, b: np.ndarray = None
    ) -> np.ndarray:
        """Kronecker product between two arrays with self products for a and b

        :param a: first element of the Kronecker product
        :type a: np.ndarray
        :param b: second element of the Kronecker product
        :type b: np.ndarray
        :return: the result of the kronecker product
        :rtype: np.ndarray
        """

        ab = np.concatenate([a, b], axis=-1)
        kron_ab = self._kronecker_product(a=ab, b=ab)

        return kron_ab

    # Kronecker product for the variables themselves
    def _simple_kronecker_product(self, a: np.ndarray = None, **kwargs) -> np.ndarray:
        """Kronecker product with a=b

        :param a: first element of the Kronecker product
        :type a: np;ndarray
        :return: the result of the kronecker product
        :rtype: np.ndarray
        """

        kron_aa = self._kronecker_product(a=a, b=a)

        return kron_aa

    # Serially constructing operators
    def _serial_operators_construction_dispatcher(
        self,
        input_chunks: list = None,
        target_chunks: list = None,
        forcing_chunks: list = None,
        D_o: np.ndarray = None,
        R_matrix: np.ndarray = None,
    ) -> (np.ndarray, np.ndarray):
        """Dispatching the batch-wise global data matrix evaluation in a serial way

        :param input_chunks: list of input data chunks
        :type input_chunks: List[np.ndarray]
        :param target_chunks: list of target data chunks
        :type target_chunks: List[np.ndarray]
        :param D_o: pre-allocated global matrix used for receiving the chunk-wise evaluation
        :type D_o: np.ndarray
        :param R_matrix: pre-allocated residual matrix used for receiving the chunk-wise evaluation
        :type R_matrix: np.ndarray
        :return: the pair (data_matrix, residual_matrix) evaluated for all the chunks/batches
        :rtype: (np.ndarray, np.ndarray)
        """

        for ii, (i_chunk, t_chunk, f_chunk) in enumerate(
            zip(input_chunks, target_chunks, forcing_chunks)
        ):
            sys.stdout.write(
                "\rProcessing chunk {} of {}".format(ii + 1, len(input_chunks))
            )
            sys.stdout.flush()

            D_o_ii, R_matrix_ii = self._construct_operators(
                input_data=i_chunk, target_data=t_chunk, forcing_data=f_chunk
            )

            D_o += D_o_ii
            R_matrix += R_matrix_ii

        return D_o, R_matrix

    # Parallely constructing operators
    def _parallel_operators_construction_dispatcher(
        self,
        input_chunks: list = None,
        target_chunks: list = None,
        forcing_chunks: list = None,
        D_o: np.ndarray = None,
        R_matrix: np.ndarray = None,
    ) -> (np.ndarray, np.ndarray):
        """Dispatching the batch-wise global data matrix evaluation in a parallel way

        :param input_chunks: list of input data chunks
        :type input_chunks: List[np.ndarray]
        :param forcing_chunks: list of forcing data chunks
        :type forcing_chunks: List[np.ndarray]
        :param target_chunks: list of target data chunks
        :type target_chunks: List[np.ndarray]
        :param D_o: pre-allocated global matrix used for receiving the chunk-wise evaluation
        :type D_o: np.ndarray
        :param R_matrix: pre-allocated residual matrix used for receiving the chunk-wise evaluation
        :type R_matrix: np.ndarray
        :return: the pair (data_matrix, residual_matrix) evaluated for all the chunks/batches
        :rtype: (np.ndarray, np.ndarray)
        """

        # All the datasets list must have the same length in order to allow the compatibility and the partitions
        # between workers.
        assert len(input_chunks) == len(target_chunks) == len(forcing_chunks), (
            "All the list must have the same"
            "length, but received "
            f"{len(input_chunks)}, "
            f"{len(target_chunks)} and"
            f"{len(forcing_chunks)}"
        )

        keys = list()

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        n_chunks = len(input_chunks)

        if rank == 0:
            for batch_id in range(n_chunks):
                print("Preparing the batch {}".format(batch_id))

                keys.append(f"batch_{batch_id}")

        input_chunks = comm.bcast(input_chunks, root=0)
        target_chunks = comm.bcast(target_chunks, root=0)
        forcing_chunks = comm.bcast(forcing_chunks, root=0)
        keys = comm.bcast(keys, root=0)

        comm.barrier()

        kwargs = {
            "input_chunks": input_chunks,
            "target_chunks": target_chunks,
            "forcing_chunks": forcing_chunks,
            "key": keys,
        }

        # Pipeline for executing MPI jobs for independent sub-processes
        mpi_run = PipelineMPI(
            exec=self._parallel_exec_wrapper, collect=True, show_log=self.show_log
        )

        # Fitting the model instances in parallel
        mpi_run.run(kwargs=kwargs)

        # When MPI finishes a run it outputs a dictionary containing status_dict the
        # partial result of each worker
        if mpi_run.success:
            out = mpi_run.status_dict

            values = out.values()

            # Each field in the output dictionary contains a tuple (D_0, R_matrix)
            # with the partial values of the OpInf system matrices
            D_o = sum([v[0] for v in values])
            R_matrix = sum([v[1] for v in values])

            self.success = True
        else:
            self.continuing = 0

        return D_o, R_matrix

    # Wrapper for the independent parallel process
    def _parallel_exec_wrapper(
        self,
        input_chunks: np.ndarray = None,
        target_chunks: np.ndarray = None,
        forcing_chunks: list = None,
        key: str = None,
    ) -> dict:
        D_o_ii, R_matrix_ii = self._construct_operators(
            input_data=input_chunks,
            target_data=target_chunks,
            forcing_data=forcing_chunks,
        )

        return {key: [D_o_ii, R_matrix_ii]}

    def _generate_data_matrices(
        self,
        input_data: np.ndarray = None,
        target_data: np.ndarray = None,
        forcing_data: np.ndarray = None,
        **kwargs,
    ) -> (np.ndarray, np.ndarray):
        # If forcing_data is None, the Kronecker product is applied just for the field
        # variables, thus reducing to the no forcing term case
        # The field variables quadratic terms are used anyway.

        n_samples = input_data.shape[0]

        quadratic_input_data = self.kronecker_product(a=input_data, b=forcing_data)

        # Matrix used for including constant terms in the operator expression
        unitary_matrix = self.bias_rescale * np.ones((n_samples, 1))

        # Known data matrix (D)
        if forcing_data is not None:
            # Constructing D using purely linear forcing terms
            D = np.hstack(
                [unitary_matrix, input_data, forcing_data, quadratic_input_data]
            )

        else:
            D = np.hstack([unitary_matrix, input_data, quadratic_input_data])

        # Target data
        Res_matrix = target_data.T

        return D, Res_matrix

    # Creating datasets on disk with lazy access
    def _lazy_generate_data_matrices(
        self,
        input_data: np.ndarray = None,
        target_data: np.ndarray = None,
        forcing_data: np.ndarray = None,
        save_path: str = None,
        batch_size: int = None,
    ) -> (h5py.Dataset, h5py.Dataset, List[slice]):
        def batch_forcing(batch: np.ndarray = None) -> np.ndarray:
            return forcing_data[batch]

        def pass_forcing(*args) -> np.ndarray:
            return None

        if forcing_data is None:
            handle_forcing = pass_forcing
        else:
            handle_forcing = batch_forcing

        if save_path is None:
            save_path = self.tmp_data_path

        filename = os.path.join(save_path, "data_matrices.hdf5")
        f = h5py.File(filename, mode="w")

        Ddset = f.create_dataset("D", shape=tuple(self.D_matrix_dim), dtype="f")
        Rdset = f.create_dataset(
            "Res_matrix", shape=tuple(self.Res_matrix_dim), dtype="f"
        )

        max_batches = int(self.n_samples / batch_size)
        batches = [
            slice(item[0], item[-1])
            for item in np.array_split(np.arange(0, self.n_samples, 1), max_batches)
        ]

        for batch in batches:
            # Generating the data-driven matrices
            D, Res_matrix = self._generate_data_matrices(
                input_data=input_data[batch],
                target_data=target_data[batch],
                forcing_data=handle_forcing(batch),
            )
            Ddset[batch] = D
            Rdset[batch] = Res_matrix.T

        return Ddset, Rdset, batches, filename

    # Direct construction
    def _construct_operators(
        self,
        input_data: np.ndarray = None,
        target_data: np.ndarray = None,
        forcing_data: np.ndarray = None,
        **kwargs,
    ) -> (np.ndarray, np.ndarray):
        # Generating the data-driven matrices
        D, Res_matrix = self._generate_data_matrices(
            input_data=input_data, target_data=target_data, forcing_data=forcing_data
        )

        # Constructing the data-driven component of the left operator
        D_o = D.T @ D

        # Constructing the right residual matrix
        R_matrix = D.T @ Res_matrix.T

        return D_o, R_matrix

    # Operators can be constructed incrementally when the dimensions are too large to
    # fit in common RAM. It also can be parallelized without major issues
    def _incremental_construct_operators(
        self,
        input_data: np.ndarray = None,
        target_data: np.ndarray = None,
        forcing_data: np.ndarray = None,
        batch_size: int = None,
    ) -> (np.ndarray, np.ndarray):
        D_o = np.zeros(
            (
                self.n_linear_terms + self.n_quadratic_inputs,
                self.n_linear_terms + self.n_quadratic_inputs,
            )
        )

        R_matrix = np.zeros(
            (self.n_linear_terms + self.n_quadratic_inputs, self.n_outputs)
        )

        n_samples = input_data.shape[0]
        n_chunks = int(n_samples / batch_size)

        input_chunks = np.array_split(input_data, n_chunks, axis=0)
        target_chunks = np.array_split(target_data, n_chunks, axis=0)

        if forcing_data is not None:
            forcing_chunks = np.array_split(forcing_data, n_chunks, axis=0)
        else:
            forcing_chunks = n_chunks * [None]

        # The incremental dispatcher can be serial or parallel.
        D_o, R_matrix = self.dispatcher(
            input_chunks=input_chunks,
            target_chunks=target_chunks,
            forcing_chunks=forcing_chunks,
            D_o=D_o,
            R_matrix=R_matrix,
        )

        return D_o, R_matrix

    def _builtin_jacobian(self, x):
        return self.A_hat + (self.K_op @ x.T)

    def _external_jacobian(self, x):
        return self.jacobian_op(x)

    def _get_H_hat_column_position(self, i: int, j: int) -> Union[int, None]:
        jj = j - i

        return int((i / 2) * (2 * self.n_inputs + 1 - i) + jj)

    def _define_H_hat_coefficient_function(self, k: int, l: int, n: int, m: int):
        if m is not None:
            H_coeff = self.H_hat[k, m]
        else:
            H_coeff = 0

        if n == l:
            H_term = 2 * H_coeff
        else:
            H_term = H_coeff

        self.K_op[k, l, n] = H_term

    # Constructing a tensor for evaluating Jacobians
    def construct_K_op(self, op: callable = None) -> None:
        # Vector versions of the index functions
        get_H_hat_column_position = np.vectorize(self._get_H_hat_column_position)
        define_H_hat_coefficient_function = np.vectorize(
            self._define_H_hat_coefficient_function
        )

        if hasattr(self, "n_outputs") is False:
            self.n_outputs = self.n_inputs

        if op is None:
            self.K_op = np.zeros((self.n_outputs, self.n_inputs, self.n_inputs))
            K = np.zeros((self.n_outputs, self.n_inputs, self.n_inputs))

            for k in range(self.n_outputs):
                K[k, ...] = k

            K = K.astype(int)

            ll = np.arange(0, self.n_inputs, 1).astype(int)
            nn = np.arange(0, self.n_inputs, 1).astype(int)

            L, N = np.meshgrid(ll, nn, indexing="ij")

            M_ = get_H_hat_column_position(L, N)

            M_u = np.triu(M_)
            M = (M_u + M_u.T - M_u.diagonal() * np.eye(self.n_inputs)).astype(int)

            define_H_hat_coefficient_function(K, L, N, M)

            self.jacobian = self._builtin_jacobian

        else:
            self.jacobian_op = op

            self.jacobian = self._external_jacobian

    # Constructing the basic setup
    def construct(
        self,
        input_data: np.ndarray = None,
        target_data: np.ndarray = None,
        forcing_data: np.ndarray = None,
    ) -> None:
        # Collecting information dimensional information from the datasets
        if (
            isinstance(input_data, np.ndarray)
            == isinstance(target_data, np.ndarray)
            == True
        ):
            assert len(input_data.shape) == len(target_data.shape) == 2, (
                "The input and target data, "
                "must be two-dimensional but received shapes"
                f" {input_data.shape} and {target_data.shape}"
            )
            self.n_samples = input_data.shape[0]

            # When there are forcing variables there are extra operators in the model
            if self.forcing is not None:
                assert (
                    forcing_data is not None
                ), "If the forcing terms are used, forcing data must be provided."

                assert len(forcing_data.shape) == 2, (
                    "The forcing data must be two-dimensional,"
                    f" but received shape {forcing_data.shape}"
                )

                assert (
                    input_data.shape[0] == target_data.shape[0] == forcing_data.shape[0]
                ), (
                    "The number of samples is not the same for all the sets with"
                    f"{input_data.shape[0]}, {target_data.shape[0]} and {forcing_data.shape[0]}."
                )

                self.n_forcing_inputs = forcing_data.shape[1]
            # For no forcing cases, the classical form is adopted
            else:
                print("Forcing terms are not being used.")
                assert input_data.shape[0] == target_data.shape[0], (
                    "The number of samples is not the same for all the sets with"
                    f"{input_data.shape[0]} and {target_data.shape[0]}"
                )

            # Number of inputs or degrees of freedom
            self.n_inputs = input_data.shape[1]
            self.n_outputs = target_data.shape[1]

        # When no dataset is provided to fit, it is necessary directly setting up the dimension values
        elif (
            isinstance(input_data, np.ndarray)
            == isinstance(target_data, np.ndarray)
            == False
        ):
            assert self.n_inputs != None and self.n_outputs != None, (
                "It is necessary to provide some" " value to n_inputs and n_outputs"
            )

        else:
            raise Exception(
                "There is no way for executing the system construction"
                " if no dataset or dimension is provided."
            )

        # Defining parameters for the Kronecker product
        if (self.forcing is None) or (self.forcing == "linear"):
            # Getting the upper component indices of a symmetric matrix
            self.i_u, self.j_u = np.triu_indices(self.n_inputs)
            self.n_quadratic_inputs = self.i_u.shape[0]

        # When the forcing interaction is 'nonlinear', there operator H_hat is extended
        elif self.forcing == "nonlinear":
            # Getting the upper component indices of a symmetric matrix
            self.i_u, self.j_u = np.triu_indices(self.n_inputs + self.n_forcing_inputs)
            self.n_quadratic_inputs = self.i_u.shape[0]

        else:
            print(f"The option {self.forcing} is not allowed for the forcing kind.")

        # Number of linear terms
        if forcing_data is not None:
            self.n_forcing_inputs = forcing_data.shape[1]
            self.n_linear_terms = 1 + self.n_inputs + self.n_forcing_inputs
        else:
            self.n_linear_terms = 1 + self.n_inputs

        self.raw_model = False

    # Evaluating the model operators
    def fit(
        self,
        input_data: np.ndarray = None,
        target_data: np.ndarray = None,
        forcing_data: np.ndarray = None,
        batch_size: int = None,
        Lambda: np.ndarray = None,
        continuing: Optional[bool] = True,
        fit_partial: Optional[bool] = False,
        force_lazy_access: Optional[bool] = False,
        k_svd: Optional[int] = None,
        save_path: Optional[str] = None,
    ) -> None:
        """Solving an Operator Inference system from large dataset

        :param input_data: dataset for the input data
        :type input_data: np.ndarray
        :param target_data: dataset for the target data
        :type target_data: np.ndarray
        :param forcing_data: dataset for the forcing data
        :type forcing_data: np.ndarray
        :param batch_size: size of the batch used for creating the global system matrices
        :type batch_size: int
        :param Lambda: customized regularization matrix
        :type Lambda: np.ndarray
        """

        if type(self.solver) == str:
            self.construct(
                input_data=input_data,
                target_data=target_data,
                forcing_data=forcing_data,
            )

            # Constructing the system operators
            if self.solver_nature == "memory":
                # This operation can require a large memory footprint, so it also can be executed
                # in chunks and, eventually, in parallel.

                if isinstance(batch_size, int):
                    construct_operators = self._incremental_construct_operators
                else:
                    construct_operators = self._construct_operators

                if self.D_o is None and self.R_matrix is None:
                    D_o, R_matrix = construct_operators(
                        input_data=input_data,
                        target_data=target_data,
                        forcing_data=forcing_data,
                        batch_size=batch_size,
                    )
                    self.D_o = D_o
                    self.R_matrix = R_matrix

                if (
                    type(self.D_o) == np.ndarray
                    and type(self.R_matrix) == np.ndarray
                    and fit_partial is True
                ):
                    D_o, R_matrix = construct_operators(
                        input_data=input_data,
                        target_data=target_data,
                        forcing_data=forcing_data,
                        batch_size=batch_size,
                    )
                    self.D_o += D_o
                    self.R_matrix += R_matrix
                else:
                    D_o = self.D_o
                    R_matrix = self.R_matrix
                    self.continuing = 1

                # If just system matrices, D_o and R_matrix are desired, the execution can be interrupted
                # here.
                if self.continuing and continuing is not False:
                    # Regularization operator
                    if Lambda is None:
                        Lambda = np.ones(self.n_linear_terms + self.n_quadratic_inputs)
                        Lambda[: self.n_linear_terms] = self.lambda_linear
                        Lambda[self.n_linear_terms :] = self.lambda_quadratic
                    else:
                        print("Using an externally defined Lambda vector.")

                    Gamma = Lambda * np.eye(
                        self.n_linear_terms + self.n_quadratic_inputs
                    )

                    # Left operator
                    L_operator = D_o + Gamma.T @ Gamma

                    # Solving the linear system via least squares
                    print("Solving linear system ...")

                    if self._is_symmetric(L_operator) and self.solver is None:
                        print("L_operator is symmetric.")
                        solution = solve(L_operator, R_matrix, assume_a="sym")
                    elif self.solver == "pinv_close":
                        D_o_pinv = np.linalg.pinv(D_o)
                        solution = D_o_pinv @ R_matrix
                    else:
                        solution = np.linalg.lstsq(L_operator, R_matrix, rcond=None)[0]

                    # Setting up the employed matrix operators
                    self.set_operators(global_matrix=solution)

            # It corresponds to the case 'lazy' in which data is temporally stored on disk.
            # In case of using the Moore-Penrose pseudo-inverse it is necessary
            # to store the entire data matrices in order to solve the undetermined system
            else:
                if self.check_fits_in_memory == "global" and force_lazy_access is False:
                    D, Res_matrix = self._generate_data_matrices(
                        input_data=input_data,
                        target_data=target_data,
                        forcing_data=forcing_data,
                    )

                    D_pinv = np.linalg.pinv(D)
                    solution = D_pinv @ Res_matrix.T

                else:
                    if force_lazy_access is True:
                        print("The batchwise execution is being forced.")

                    assert (
                        batch_size is not None
                    ), f"It is necessary to define batch_size but received {batch_size}."
                    (
                        D,
                        Res_matrix,
                        batches,
                        filename,
                    ) = self._lazy_generate_data_matrices(
                        input_data=input_data,
                        target_data=target_data,
                        forcing_data=forcing_data,
                        save_path=save_path,
                        batch_size=batch_size,
                    )
                    if k_svd is None:
                        k_svd = self.n_inputs

                    pinv = CompressedPinv(
                        D=D, chunks=(batch_size, self.n_inputs), k=k_svd
                    )
                    solution = pinv(Y=Res_matrix, batches=batches)

                    # Removing the file stored in disk
                    os.remove(filename)

                # Setting up the employed matrix operators
                self.set_operators(global_matrix=solution)

        elif callable(self.solver):
            warnings.warn("Iterative solvers are not currently supported.")
            warnings.warn("Finishing fitting process without modifications.")

        else:
            raise Exception(
                f"The option {type(self.solver)} is not suported.\
                            it must be callable or str."
            )

        print("Fitting process concluded.")

    # Making residual evaluations using the trained operator without forcing terms
    def _eval(self, input_data: np.ndarray = None) -> np.ndarray:
        # If forcing_data is None, the Kronecker product is applied just for the field
        # variables, thus reducing to the no forcing term case
        quadratic_input_data = self.kronecker_product(a=input_data)

        output = input_data @ self.A_hat.T
        output += quadratic_input_data @ self.H_hat.T
        output += self.c_hat.T

        return output

    # Making residual evaluations using the trained operator with forcing terms
    def _eval_forcing(
        self, input_data: np.ndarray = None, forcing_data: np.ndarray = None
    ) -> np.ndarray:
        # If forcing_data is None, the Kronecker product is applied just for the field
        # variables, thus reducing to the no forcing term case
        quadratic_input_data = self.kronecker_product(a=input_data, b=forcing_data)

        output = input_data @ self.A_hat.T
        output += quadratic_input_data @ self.H_hat.T
        output += forcing_data @ self.B_hat.T
        output += self.c_hat.T

        return output

    def eval(self, input_data: np.ndarray = None, **kwargs) -> np.ndarray:
        """Evaluating using the trained model

        :param input_data: array containing the input data
        :type input_data: np.ndarray
        :return: output evaluation using the trained model
        :rtype: np.ndarray
        """

        return self.eval_op(input_data=input_data, **kwargs)

    # Saving to disk the complete model
    def save(self, save_path: str = None, model_name: str = None) -> None:
        """Complete saving

        :param save_path: path to the saving directory
        :type: str
        :param model_name: name for the model
        :type model_name: str
        :return: nothing
        """

        path = os.path.join(save_path, model_name + ".pkl")
        try:
            with open(path, "wb") as fp:
                pickle.dump(self, fp, protocol=4)
        except Exception as e:
            print(e, e.args)

    # Saving to disk a lean version of the model
    def lean_save(self, save_path: str = None, model_name: str = None) -> None:
        """Lean saving

        :param save_path: path to the saving directory
        :type: str
        :param model_name: name for the model
        :type model_name: str
        :return: nothing
        """

        # Parameters to be removed in a lean version of the model
        black_list = ["D_o", "R_matrix"]

        path = os.path.join(save_path, model_name + ".pkl")

        self_copy = deepcopy(self)

        for item in black_list:
            del self_copy.__dict__[item]

        try:
            with open(path, "wb") as fp:
                pickle.dump(self_copy, fp, protocol=4)
        except Exception as e:
            print(e, e.args)
