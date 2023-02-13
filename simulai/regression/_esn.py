# (C) Copyright IBM Corp. 2019, 2020, 2021, 2022.
import os
import pickle
import platform
import sys
import warnings
from inspect import signature
from multiprocessing import Pool
from typing import Union

import numpy as np
import psutil
import scipy.sparse as sparse
from scipy import linalg
from scipy.sparse.linalg import eigs

from simulai.templates import ReservoirComputing

#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at

#           http://www.apache.org/licenses/LICENSE-2.0

#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.


# It could be used for executing multi-core CPU or even GPU jobs
# with NumPy arrays
def tensor_dot(self, engine: str = "numba") -> callable:
    try:
        from numba import float64, guvectorize
    except:
        raise Exception("Numba is not installed in your system.")

    @guvectorize([(float64[:, :], float64[:, :])], "(n,m)->(n,n)", target="cpu")
    def _tensor_dot(vector: np.ndarray, out: np.ndarray) -> None:
        for index in range(vector.shape[1]):
            out += vector[:, index][:, None] @ (vector[:, index][:, None]).transpose()

    return _tensor_dot


# Numba loop for shared-memory parallel execution
def outer_dot(engine: str = "numba") -> callable:
    if engine == "numba":
        try:
            from numba import njit
        except:
            raise Exception("Numba is not installed in your system.")

        print("Using numba engine.")

        @njit(parallel=True)
        def _outer_dot(vector: np.ndarray, out: np.ndarray) -> np.ndarray:
            for ii in range(vector.shape[1]):
                out += np.outer(vector[:, ii], vector[:, ii])

            return out

    else:

        def _outer_dot(vector: np.ndarray, out: np.ndarray):
            pass

    return _outer_dot


# Combining Numba with multiprocessing
def multiprocessing_dot(vector: np.ndarray) -> np.ndarray:
    print("Using multiprocessing engine.")

    out = np.zeros((vector.shape[0], vector.shape[0]))

    for ii in range(vector.shape[1]):
        out += np.outer(vector[:, ii], vector[:, ii])

    return out


"""
    BEGIN Reservoir Computing (RC) and children classes
"""


class EchoStateNetwork(ReservoirComputing):

    """
    Echo State Network, a subclass of the Reservoir Computing methods
    """

    def __init__(
        self,
        reservoir_dim: int = None,
        sparsity_level: float = None,
        radius: float = 1.0,
        number_of_inputs: int = None,
        sigma: float = 1.0,
        beta: float = None,
        kappa: float = None,
        leak_rate: float = 1,
        activation: str = "tanh",
        Win_interval: list = [-1, 1],
        tau: float = None,
        Win_init: str = "global",
        transformation: str = "T1",
        solver: str = "direct_inversion",
        s_reservoir_matrix=None,
        W_in=None,
        eta=0,
        global_matrix_constructor_str="multiprocessing",
        A=None,
        b=None,
        estimate_linear_transition=False,
        estimate_bias_transition=False,
        n_workers=1,
        memory_percent=0.5,
        layerwise_train=None,
        show_log: bool = False,
        input_augmented_reservoir: bool = False,
        model_id: str = None,
    ) -> None:
        """
        :param reservoir_dim: dimension of the reservoir matrix
        :type reservoir_dim: int
        :param sparsity_level: level of sparsity at the matrices initialization
        :type sparsity_level: float
        :param radius:
        :param number_of_inputs: number of variables used as input
        :type number_of_inputs: int
        :param sigma: multiplicative constant for the interval of the input matrix initialization
        :type sigma: float
        :param beta: Ridge regularization parameter
        :type beta: float
        :param kappa:
        :param tau:
        :param activation:
        :param Win_interval:
        :param transformation:
        :param model_id:
        """

        super().__init__(reservoir_dim=reservoir_dim, sparsity_level=sparsity_level)

        self.sparsity_tolerance = 0.0025  # Default choice

        self.reservoir_dim = reservoir_dim

        self.sparsity_level = sparsity_level

        self.radius = radius
        self.number_of_inputs = number_of_inputs
        self.sigma = sigma
        self.beta = beta
        self.leak_rate = leak_rate
        self.set_activation(activation)
        self.Win_interval = Win_interval
        self.set_tau(tau)
        self.kappa = kappa
        self.Win_init = Win_init
        self.set_transformation(transformation)
        self.eta = eta
        self.global_matrix_constructor_str = global_matrix_constructor_str
        self.n_workers = n_workers
        self.memory_percent = memory_percent
        self.layerwise_train = layerwise_train
        self.input_augmented_reservoir = input_augmented_reservoir
        self.show_log = show_log

        self.s_reservoir_matrix = None
        self.W_in = None
        self.W_out = None
        self.current_state = None
        self.reference_state = None
        self.A = A
        self.b = b
        self.estimate_linear_transition = estimate_linear_transition
        self.estimate_bias_transition = estimate_bias_transition

        self.model_id = model_id

        # Echo state networks does not use history during fitting
        self.depends_on_history = False

        # Linear system solver
        self.solver_op = None
        self.solver = None
        self.set_solver(solver)

        if self.global_matrix_constructor_str == "direct":
            self.global_matrix_constructor = self._construct_global_matrix_direct
        elif self.global_matrix_constructor_str == "numba":
            if platform.system() == "Darwin":
                warnings.warn(
                    "You are using MacOS. Numba have presented some bugs in this platform."
                )
                warnings.warn(
                    "You are using MacOS. If you got some issue, test to use multiprocessing instead."
                )

            self.global_matrix_constructor = self._construct_global_matrix_numba
        elif self.global_matrix_constructor_str == "multiprocessing":
            self.global_matrix_constructor = (
                self._construct_global_matrix_multiprocessing
            )
        else:
            raise Exception(
                f"Case {self.global_matrix_constructor_str} not available for creating the"
                f"global matrix."
            )

        # The matrices input-to-reservoir and reservoir can be
        # directly passed to the class instance
        if s_reservoir_matrix is None and W_in is None:
            print("Initializing ESN matrices ...")

            self.s_reservoir_matrix, self.W_in = self._initialize_parameters()
        else:
            self.s_reservoir_matrix = s_reservoir_matrix
            self.W_in = W_in

        self.size_default = np.array([0]).astype("float64").itemsize

        if show_log == True:
            self.logger = self._show_log
        else:
            self.logger = self._no_log

    def _show_log(self, i):
        sys.stdout.write(f"\r state {i}")
        sys.stdout.flush()

    def _no_log(self, i):
        pass

    @property
    def trainable_variables(
        self,
    ):
        return {"A": self.A, "b": self.b, "W_out": self.W_out}

    def set_activation(self, activation):
        self.activation = activation
        self.activation_op = self._get_activation_function(self.activation)

    def _get_activation_function(self, activation):
        method = (
            getattr(np, activation, None)
            or getattr(np.math, activation, None)
            or getattr(self, "_" + activation, None)
        )
        if method:
            return method
        else:
            raise Exception(
                "The activation {} is still not available.".format(activation)
            )

    # Linear activation
    def _linear(self, data):
        return data

    # Sigmoid activation
    def _sigmoid(self, data):
        return 1 / (1 + np.exp(-data))

    """
        BEGIN Transformations that can be applied to the hidden state

        Note: A shift = 1 is applied (ii + 1) in order to respect the Python
        0-based system.
    """

    def set_transformation(self, transformation):
        self.transformation = transformation
        self._construct_r_til = getattr(self, "_" + transformation)

    # ByPass transformation
    def _T0(self, r):
        r_til = r.copy()

        return r_til

    def _T1(self, r):
        n_rows = r.shape[0]
        r_til = r.copy()

        for ii in range(n_rows):
            if (ii + 1) % 2 != 0:
                r_til[ii] = (r[ii] * r[ii]).copy()
            else:
                pass
        return r_til

    def _T2(self, r):
        n_rows = r.shape[0]
        r_til = r.copy()

        for ii in range(2, n_rows):
            if (ii + 1) % 2 != 0:
                r_til[ii] = r[ii - 1] * r[ii - 2]
            else:
                pass
        return r_til

    def _T3(self, r):
        n_rows = r.shape[0]
        r_til = r.copy()

        for ii in range(1, n_rows - 1):
            if (ii + 1) % 2 != 0:
                r_til[ii] = r[ii - 1] * r[ii + 1]
            else:
                pass
        return r_til

    """
            END Transformations that can be applied to the hidden state
    """

    def set_tau(self, tau):
        self.tau = tau
        self.kappa = None

    def _default_kappa(self, size):
        x = np.arange(size) / size

        # tau is a positive number in (0, inf] preferably close to 0, e.g., 0.1
        # y[0] == 0 and y is monotonically increasing, i.e, y[k+1] > y[k] and y[-1] < 1
        # we choose tau such that 0.99 ~= y[int(tau*size)]

        y = 1 - 10 ** (-2 * x / self.tau)
        y = np.reshape(y, [1, -1])
        return y

    def _update_state(self, previous_state, input_series_state):
        reservoir_input = previous_state[: self.reservoir_dim]
        state = (1 - self.leak_rate) * reservoir_input
        state += self.leak_rate * self.activation_op(
            self.s_reservoir_matrix.dot(reservoir_input)
            + np.dot(self.W_in, input_series_state)
        )

        if self.input_augmented_reservoir:
            state = np.hstack((state, input_series_state))

        return state

    # TODO this stage is considerably expensive. It is necessary to fix it in anyway.
    def _reservoir_layer(self, input_series, data_size):
        r_states = np.zeros((self.state_dim, data_size))
        r_state = self.default_state
        for i in range(data_size):
            r_states[:, i] = self._update_state(r_state, input_series[:, i])
            r_state = r_states[:, i]

            self.logger(i)

        return r_states

    # This method is used for initializing a raw model
    def _initialize_parameters(self):
        # Sparse uniform random matrix
        sparse_reservoir_matrix = self.create_reservoir()

        sparse_reservoir_matrix += self.eta * sparse.eye(self.reservoir_dim)

        dense_sparse_reservoir_matrix = sparse_reservoir_matrix.todense()

        # Normalizing with the maximum eigenvalue
        # TODO It could be done using a sparse matrix algorithm, but
        #  there is some issue with the SciPy sparse library. it is necessary to check it.
        if self.reservoir_dim == 0:
            max_eig = 1.0
        else:
            eigvals = eigs(
                dense_sparse_reservoir_matrix,
                k=1,
                which="LM",
                return_eigenvectors=False,
            )
            max_eig = np.max(np.abs(eigvals))

        sparse_reservoir_matrix = sparse_reservoir_matrix * (self.radius / max_eig)

        if self.Win_init == "global":
            Win = np.random.uniform(
                *self.Win_interval, size=[self.reservoir_dim, self.number_of_inputs]
            )

        # TODO the blockwise option can be slow. Maybe it can be parallelized.
        elif self.Win_init == "blockwise":
            n_b = int(self.reservoir_dim / self.number_of_inputs)
            Win = np.zeros((self.reservoir_dim, self.number_of_inputs))

            for j in range(self.number_of_inputs):
                np.random.seed(seed=j)
                Win[j * n_b : (j + 1) * n_b, j] = np.random.uniform(
                    *self.Win_interval, size=[1, n_b]
                )[0]
        else:
            raise Exception(
                f"The initialization method {self.Win_init} for W_in is not supported."
            )

        Win = self.sigma * Win

        return sparse_reservoir_matrix, np.array(Win)

    def set_sigma(self, sigma):
        self.W_in = self.W_in * (sigma / self.sigma)
        self.sigma = sigma

    def set_radius(self, radius):
        self.s_reservoir_matrix = self.s_reservoir_matrix * (radius / self.radius)
        self.radius = radius

    def set_solver(self, solver):
        self.solver = solver
        if isinstance(solver, str):
            try:
                self.solver_op = getattr(self, "_" + solver)
            except Exception:
                raise Exception(f"The option {solver} is not implemented")
        else:
            print(
                f"The object {solver} provided is not an attribute of {self}.\
                    Let us consider it as a functional wrapper.\
                    Good luck in using it."
            )

            if callable(solver):
                self.solver_op = solver
            else:
                raise RuntimeError(f"object {solver} is not callable.")

    # It solves a linear system via direct matrix inversion
    def _direct_inversion(self, data_out_, U, r_til):
        print("Solving the system via direct matrix inversion.")

        Uinv = np.linalg.inv(U)

        Wout = (data_out_ @ r_til.T) @ Uinv

        return Wout

    # It solves a linear system using the SciPy algorithms
    # Note that U is symmetric positive definite
    def _linear_system(self, data_out_, U, r_til):
        print("Solving the linear system using the most proper algorithm.")

        Wout = linalg.solve(U, (r_til @ data_out_.T)).T

        return Wout

    def _construct_global_matrix_direct(self, r_til, idenmat):
        U = np.zeros((r_til.shape[0], r_til.shape[0]))

        for ii in range(r_til.shape[1]):
            U += np.outer(r_til[:, ii], r_til[:, ii])

        U += idenmat

        return U

    def _construct_global_matrix_numba(self, r_til, idenmat):
        U = np.zeros((r_til.shape[0], r_til.shape[0]))

        dot = outer_dot(engine="numba")
        U = dot(r_til, U)

        U += idenmat

        return U

    def _construct_global_matrix_multiprocessing(self, r_til, idenmat):
        assert self.n_workers is not None, (
            "If you are using multiprocessing," "it is necessary to define n_workers."
        )

        assert self.memory_percent is not None, (
            "If you are using multiprocessing,"
            "it is necessary to define memory_percent."
        )

        single_matrix_size = self.size_default * r_til.shape[0] ** 2

        available_memory = self.memory_percent * psutil.virtual_memory().available

        # Evaluating the maximum number of partial U matrices to be allocated
        max_n_workers = int(available_memory / single_matrix_size)

        if max_n_workers >= self.n_workers:
            n_workers = self.n_workers
        else:
            n_workers = max_n_workers

        r_til_chunks = np.array_split(r_til, n_workers, axis=1)

        pool = Pool(processes=self.n_workers)

        U_chunks = pool.map(multiprocessing_dot, r_til_chunks)

        U = sum(U_chunks)

        U += idenmat

        return U

    @staticmethod
    def _prepare_inputs(concat: bool = True):
        def _concat(main, aux, index):
            return np.hstack([main, aux[index : index + 1]])

        def _by_pass(main, aux, index):
            return main

        if concat is True:
            return _concat
        else:
            return _by_pass

    # Setting up new parameters for a restored model
    def set_parameters(self, parameters):
        for key, value in parameters.items():
            if key in [
                "radius",
                "sigma",
                "tau",
                "activation",
                "transformation",
                "solver",
            ]:
                getattr(self, f"set_{key}")(value)
            else:
                setattr(self, key, value)

    def fit(self, input_data=None, target_data=None):
        data_in = input_data.T
        data_out = target_data.T

        if not self.estimate_bias_transition and not self.estimate_linear_transition:
            # dont estimate linear of bias terms
            pass
        elif self.estimate_bias_transition and not self.estimate_linear_transition:
            print("Estimating the bias")
            if self.A is not None:
                self.b = np.linalg.lstsq(
                    np.ones((target_data.shape[0], 1), dtype=target_data.dtype),
                    target_data - (self.A @ data_in).T,
                    rcond=None,
                )[0]
            else:
                self.b = np.linalg.lstsq(
                    np.ones((target_data.shape[0], 1), dtype=target_data.dtype),
                    target_data,
                    rcond=None,
                )[0]
            self.b = np.reshape(self.b, (-1,))
        elif not self.estimate_bias_transition and self.estimate_linear_transition:
            print("Estimating the linear transition")
            if self.b is not None:
                self.A = np.linalg.lstsq(
                    input_data, target_data - np.reshape(self.b, (1, -1)), rcond=None
                )[0].T
            else:
                self.A = np.linalg.lstsq(input_data, target_data, rcond=None)[0].T
        elif self.estimate_bias_transition and self.estimate_linear_transition:
            print("Estimating the linear transition and bias")
            self.A, self.b = np.hsplit(
                np.linalg.lstsq(
                    np.hstack((input_data, np.ones((target_data.shape[0], 1)))),
                    target_data,
                    rcond=None,
                )[0].T,
                [input_data.shape[1]],
            )
            self.b = np.reshape(self.b, (-1,))
        else:
            raise RuntimeError("Unreachable line of code")

        affine_term = 0
        if self.A is not None:
            affine_term += self.A @ data_in
        if self.b is not None:
            affine_term += np.reshape(self.b, (-1, 1))

        if self.state_dim > 0:
            data_out = data_out - affine_term

            data_size = data_in.shape[1]

            print("Evaluating the hidden state ...")

            r = self._reservoir_layer(data_in, data_size)

            print("\nApplying transformation ...")

            r_til = self._construct_r_til(r[: self.reservoir_dim, ...])
            if self.input_augmented_reservoir:
                r_til = np.vstack((r_til, r[self.reservoir_dim :, ...]))

            # Verifying the adjusting array kappa
            if isinstance(self.kappa, np.ndarray):
                self.kappa = np.reshape(self.kappa, [1, -1])
                assert (
                    self.kappa.shape[1] == r_til.shape[1]
                ), f"kappa and r-til must have the same samples dimension \
                but sizes are {self.kappa.shape[1]} and {r_til.shape[1]}"
            elif self.tau is not None and self.tau > 0:
                self.kappa = self._default_kappa(r_til.shape[1])
            else:
                self.kappa = np.ones((1, r_til.shape[1]))

            r_til *= self.kappa
            data_out_ = self.kappa * data_out

            idenmat = self.beta * sparse.identity(self.state_dim)

            print("Constructing W_out ...")

            # Constructing and solving the global linear system
            U = self.global_matrix_constructor(r_til, idenmat)

            Wout = self.solver_op(data_out_, U, r_til)

            self.W_out = np.array(Wout)

            self.current_state = r[:, -1].copy()
            self.reference_state = r[:, -1].copy()
        else:
            self.W_out = np.zeros(
                (target_data.shape[1], self.state_dim), dtype=target_data.dtype
            )
            self.current_state = np.zeros((self.state_dim,), dtype=target_data.dtype)
            self.reference_state = np.zeros((self.state_dim,), dtype=target_data.dtype)

        print("Fitting concluded.")

    def step(self, data=None):
        affine_term = 0
        if self.A is not None:
            affine_term += self.A @ data
        if self.b is not None:
            affine_term += self.b

        state = self.current_state.copy()

        r_ = self._update_state(state, data)

        state_ = self._construct_r_til(r_[: self.reservoir_dim, ...])
        if self.input_augmented_reservoir:
            state_ = np.hstack((state_, r_[self.reservoir_dim :, ...]))

        out = self.W_out @ state_ + affine_term

        self.current_state = r_.copy()

        return out

    @property
    def state_dim(self):
        if self.input_augmented_reservoir:
            return self.reservoir_dim + self.number_of_inputs
        else:
            return self.reservoir_dim

    @property
    def default_state(self):
        return np.zeros((self.state_dim,), dtype=self.s_reservoir_matrix.dtype)

    def set_reference(self, reference=None):
        if reference is None:
            self.reference_state = self.default_state
        else:
            if isinstance(reference, (int, float)):
                reference = np.full_like(self.default_state, reference)
            assert np.array_equal(
                (self.state_dim,), reference.shape
            ), f"Invalid shape {(self.state_dim, )} != {reference.shape}"
            self.reference_state = reference

    def reset(self):
        if self.reference_state is None:
            self.reference_state = self.default_state
        self.current_state = self.reference_state

    def predict(self, initial_data=None, horizon=None, auxiliary_data=None):
        if auxiliary_data is not None:
            concat = True
        else:
            concat = False

        prepare_data = self._prepare_inputs(concat=concat)

        output_data = np.zeros((self.W_out.shape[0], horizon))

        state = self.current_state.copy()  # End of fit stage

        data = prepare_data(initial_data, auxiliary_data, 0)

        for tt in range(horizon):
            print("Extrapolating for the timestep {}".format(tt))

            r_ = self._update_state(state, data)

            r_til = self._construct_r_til(r_[: self.reservoir_dim, ...])
            if self.input_augmented_reservoir:
                r_til = np.hstack((r_til, r_[self.reservoir_dim :, ...]))

            affine_term = 0
            if self.A is not None:
                affine_term += self.A @ data
            if self.b is not None:
                affine_term += self.b

            out = np.array(self.W_out) @ r_til + affine_term
            output_data[:, tt] = out

            state = r_
            data = prepare_data(out, auxiliary_data, tt)

        self.current_state = state

        return output_data.T

    def save(self, save_path=None, model_name=None):
        configs = {
            k: getattr(self, k) for k in signature(self.__init__).parameters.keys()
        }

        to_save = {
            "current_state": self.current_state,
            "reference_state": self.reference_state,
            "W_out": self.W_out,
            "configs": configs,
        }
        path = os.path.join(save_path, model_name + ".pkl")
        try:
            with open(path, "wb") as fp:
                pickle.dump(to_save, fp, protocol=4)
        except Exception as e:
            print(e, e.args)

    # Saving the entire model
    def save_model(self, save_path=None, model_name=None):
        path = os.path.join(save_path, model_name + ".pkl")
        try:
            with open(path, "wb") as fp:
                pickle.dump(self, fp, protocol=4)
        except Exception as e:
            print(e, e.args)

    @classmethod
    def restore(cls, model_path, model_name):
        path = os.path.join(model_path, model_name + ".pkl")
        with open(path, "rb") as fp:
            d = pickle.load(fp)

        self = cls(**d["configs"])
        d.pop("configs")
        for k, v in d.items():
            setattr(self, k, v)

        return self


# Deep version of the Echo State network
class DeepEchoStateNetwork:

    """
    Echo State Network, a subclass of the Reservoir Computing methods
    """

    def __init__(
        self,
        reservoir_dim=None,
        sparsity_level=None,
        n_layers=2,
        radius=None,
        number_of_inputs=None,
        sigma=None,
        beta=None,
        kappa=None,
        leak_rate=1,
        activation="tanh",
        Win_interval=[-1, 1],
        tau=None,
        Win_init="global",
        transformation="T1",
        solver="direct_inversion",
        s_reservoir_matrix=None,
        W_in=None,
        eta=0,
        global_matrix_constructor_str="multiprocessing",
        A=None,
        b=None,
        estimate_linear_transition=False,
        estimate_bias_transition=False,
        n_workers=1,
        memory_percent=0.5,
        layerwise_train=None,
        all_for_input=1,
        show_log=False,
        input_augmented_reservoir=False,
        model_id=None,
    ):
        """
        :param reservoir_dim: dimension of the reservoir matrix
        :type reservoir_dim: int
        :param sparsity_level: level of sparsity at the matrices initialization
        :type sparsity_level: float
        :param radius:
        :param number_of_inputs: number of variables used as input
        :type number_of_inputs: int
        :param sigma: multiplicative constant for the interval of the input matrix initialization
        :type sigma: float
        :param beta: Ridge regularization parameter
        :type beta: float
        :param kappa:
        :param tau:
        :param activation:
        :param Win_interval:
        :param transformation:
        :param model_id:
        """

        self.sparsity_tolerance = 0.0025  # Default choice
        self.n_layers = n_layers
        self.beta = beta
        self.number_of_inputs = number_of_inputs
        self.all_for_input = all_for_input

        # Some variables must be converted in lists with length self.n_layers
        self.reservoir_dim = self._make_list(reservoir_dim)
        self.sparsity_level = self._make_list(sparsity_level)
        self.radius = self._make_list(radius)
        self.sigma = self._make_list(sigma)
        self.leak_rate = self._make_list(leak_rate)
        self.n_inputs_list = self._make_n_inputs_list()

        self.set_activation(activation)
        self.Win_interval = Win_interval
        self.set_tau(tau)
        self.kappa = kappa
        self.Win_init = Win_init
        self.set_transformation(transformation)
        self.eta = eta
        self.global_matrix_constructor_str = global_matrix_constructor_str
        self.n_workers = n_workers
        self.memory_percent = memory_percent
        self.layerwise_train = layerwise_train
        self.input_augmented_reservoir = input_augmented_reservoir
        self.show_log = show_log

        self.s_reservoir_matrix = None
        self.W_in = None
        self.W_out = None
        self.current_state = None
        self.reference_state = None
        self.A = A
        self.b = b
        self.estimate_linear_transition = estimate_linear_transition
        self.estimate_bias_transition = estimate_bias_transition

        self.model_id = model_id

        # Echo state networks does not use history during fitting
        self.depends_on_history = False

        # Linear system solver
        self.solver_op = None
        self.solver = None
        self.set_solver(solver)

        if self.global_matrix_constructor_str == "direct":
            self.global_matrix_constructor = self._construct_global_matrix_direct
        elif self.global_matrix_constructor_str == "numba":
            if platform.system() == "Darwin":
                warnings.warn(
                    "You are using MacOS. Numba have presented some bugs in this platform."
                )
                warnings.warn(
                    "You are using MacOS. If you got some issue, test to use multiprocessing instead."
                )

            self.global_matrix_constructor = self._construct_global_matrix_numba
        elif self.global_matrix_constructor_str == "multiprocessing":
            self.global_matrix_constructor = (
                self._construct_global_matrix_multiprocessing
            )
        else:
            raise Exception(
                f"Case {self.global_matrix_constructor_str} not available for creating the"
                f"global matrix."
            )

        # The matrices input-to-reservoir and reservoir can be
        # directly passed to the class instance
        if s_reservoir_matrix is None and W_in is None:
            print("Initializing ESN matrices ...")

            self.s_reservoir_matrix, self.W_in = self._initialize_parameters()
        else:
            self.s_reservoir_matrix = s_reservoir_matrix
            self.W_in = W_in

        self.size_default = np.array([0]).astype("float64").itemsize

        if show_log == True:
            self.logger = self._show_log
        else:
            self.logger = self._no_log

    # The number of inputs is variable with the layer
    def _make_n_inputs_list(self):
        if self.all_for_input:
            extra_dim = self.number_of_inputs
        else:
            extra_dim = 0

        n_inputs_list = [self.number_of_inputs]
        for ll in range(1, self.n_layers):
            n_inputs_list.append(self.reservoir_dim[ll - 1] + extra_dim)

        return n_inputs_list

    # It converts an individual variable to list
    def _make_list(self, var):
        if type(var) is list:
            return var
        elif type(var) in (str, float, int):
            return self.n_layers * [var]
        else:
            raise Exception(
                f"The variable var can be list, str, int or float but received {type(var)}."
            )

    @property
    def global_reservoir_dim(self):
        return sum(self.reservoir_dim)

    def _show_log(self, i):
        sys.stdout.write(f"\r state {i}")
        sys.stdout.flush()

    def _no_log(self, i):
        pass

    @property
    def trainable_variables(
        self,
    ):
        return {"A": self.A, "b": self.b, "W_out": self.W_out}

    def set_activation(self, activation):
        self.activation = activation
        self.activation_op = self._get_activation_function(self.activation)

    def _get_activation_function(self, activation):
        method = (
            getattr(np, activation, None)
            or getattr(np.math, activation, None)
            or getattr(self, "_" + activation, None)
        )
        if method:
            return method
        else:
            raise Exception(
                "The activation {} is still not available.".format(activation)
            )

    # Linear activation
    def _linear(self, data):
        return data

    # Sigmoid activation
    def _sigmoid(self, data):
        return 1 / (1 + np.exp(-data))

    """
        BEGIN Transformations that can be applied to the hidden state

        Note: A shift = 1 is applied (ii + 1) in order to respect the Python
        0-based system.
    """

    def set_transformation(self, transformation):
        self.transformation = transformation
        self._construct_r_til = getattr(self, "_" + transformation)

    # ByPass transformation
    def _T0(self, r):
        r_til = r.copy()

        return r_til

    def _T1(self, r):
        n_rows = r.shape[0]
        r_til = r.copy()

        for ii in range(n_rows):
            if (ii + 1) % 2 != 0:
                r_til[ii] = (r[ii] * r[ii]).copy()
            else:
                pass
        return r_til

    def _T2(self, r):
        n_rows = r.shape[0]
        r_til = r.copy()

        for ii in range(2, n_rows):
            if (ii + 1) % 2 != 0:
                r_til[ii] = r[ii - 1] * r[ii - 2]
            else:
                pass
        return r_til

    def _T3(self, r):
        n_rows = r.shape[0]
        r_til = r.copy()

        for ii in range(1, n_rows - 1):
            if (ii + 1) % 2 != 0:
                r_til[ii] = r[ii - 1] * r[ii + 1]
            else:
                pass
        return r_til

    """
            END Transformations that can be applied to the hidden state
    """

    def set_tau(self, tau):
        self.tau = tau
        self.kappa = None

    def _default_kappa(self, size):
        x = np.arange(size) / size

        # tau is a positive number in (0, inf] preferably close to 0, e.g., 0.1
        # y[0] == 0 and y is monotonically increasing, i.e, y[k+1] > y[k] and y[-1] < 1
        # we choose tau such that 0.99 ~= y[int(tau*size)]

        y = 1 - 10 ** (-2 * x / self.tau)
        y = np.reshape(y, [1, -1])
        return y

    def _update_state(self, previous_state, input_series_state, layer_id=None):
        reservoir_input = previous_state[: self.reservoir_dim[layer_id]]
        state = (1 - self.leak_rate[layer_id]) * reservoir_input
        state += self.leak_rate[layer_id] * self.activation_op(
            self.s_reservoir_matrix[layer_id].dot(reservoir_input)
            + np.dot(self.W_in[layer_id], input_series_state)
        )

        if self.input_augmented_reservoir:
            state = np.hstack((state, input_series_state))

        return state

    # TODO this stage is considerably expensive. It is necessary to fix it in anyway.
    def _reservoir_layer(self, input_series, data_size):
        r_states_list = self._initialize_r_states(data_size)

        hidden_states = list()
        input_series_ = input_series

        # The states updating are executed for each layer
        for ll in range(self.n_layers):
            print(f"Applying layer {ll}")
            r_state = np.zeros(
                (self.state_dim(ll),), dtype=self.s_reservoir_matrix[ll].dtype
            )
            r_states = r_states_list[ll]

            for i in range(data_size):
                r_states[:, i] = self._update_state(
                    r_state, input_series_[:, i], layer_id=ll
                )
                r_state = r_states[:, i]

                self.logger(i)

            print("\n")
            if self.all_for_input:
                input_series_ = np.vstack([r_states, input_series])
            else:
                input_series_ = r_states

            hidden_states.append(r_states)

        # The output states for each layer ll are concatenate into a single global state
        return np.vstack(hidden_states)

    def _reservoir_dim_corrected_sparsity_level(
        self, sparsity_level=None, reservoir_dim=None
    ):
        # Guaranteeing a minimum sparsity tolerance
        dim = (
            max(sparsity_level, self.sparsity_tolerance)
            if reservoir_dim == 0
            else reservoir_dim
        )
        effective_sparsity = sparsity_level / dim
        if effective_sparsity < self.sparsity_tolerance:
            return self.sparsity_tolerance
        else:
            return sparsity_level / dim

    # It creates a sparse and randomly distributed reservoir matrix
    def _create_reservoir(self, sparsity_level=None, reservoir_dim=None):
        return sparse.rand(
            reservoir_dim,
            reservoir_dim,
            density=self._reservoir_dim_corrected_sparsity_level(
                sparsity_level=sparsity_level, reservoir_dim=reservoir_dim
            ),
        )

    # Each layerwise state is initialized as zero
    def _initialize_r_states(self, data_size):
        r_states_list = list()

        for ll in range(self.n_layers):
            r_states = np.zeros((self.state_dim(ll), data_size))
            r_states_list.append(r_states)

        return r_states_list

    # This method is used for initializing a raw model
    def _initialize_layer_parameters(self, layer_id=None):
        sparsity_level = self.sparsity_level[layer_id]
        reservoir_dim = self.reservoir_dim[layer_id]
        radius = self.radius[layer_id]
        number_of_inputs = self.n_inputs_list[layer_id]
        sigma = self.sigma[layer_id]
        # Sparse uniform random matrix
        sparse_reservoir_matrix = self._create_reservoir(
            sparsity_level=sparsity_level, reservoir_dim=reservoir_dim
        )

        sparse_reservoir_matrix += self.eta * sparse.eye(reservoir_dim)

        dense_sparse_reservoir_matrix = sparse_reservoir_matrix.todense()

        # Normalizing with the maximum eigenvalue
        # TODO It could be done using a sparse matrix algorithm, but
        #  there is some issue with the SciPy sparse library. it is necessary to check it.
        if reservoir_dim == 0:
            max_eig = 1.0
        else:
            eigvals = eigs(
                dense_sparse_reservoir_matrix,
                k=1,
                which="LM",
                return_eigenvectors=False,
            )
            max_eig = np.max(np.abs(eigvals))

        sparse_reservoir_matrix = sparse_reservoir_matrix * (radius / max_eig)

        if self.Win_init == "global":
            Win = np.random.uniform(
                *self.Win_interval, size=[reservoir_dim, number_of_inputs]
            )

        # TODO the blockwise option can be slow. Maybe it can be parallelized.
        elif self.Win_init == "blockwise":
            n_b = int(reservoir_dim / number_of_inputs)
            Win = np.zeros((reservoir_dim, number_of_inputs))

            for j in range(self.number_of_inputs):
                np.random.seed(seed=j)
                Win[j * n_b : (j + 1) * n_b, j] = np.random.uniform(
                    *self.Win_interval, size=[1, n_b]
                )[0]
        else:
            raise Exception(
                f"The initialization method {self.Win_init} for W_in is not supported."
            )

        Win = sigma * Win

        return sparse_reservoir_matrix, np.array(Win)

    def _initialize_parameters(self):
        sparse_reservoir_matrix_list = list()
        W_in_list = list()

        for ll in range(self.n_layers):
            sparse_reservoir_matrix, W_in = self._initialize_layer_parameters(
                layer_id=ll
            )

            sparse_reservoir_matrix_list.append(sparse_reservoir_matrix)
            W_in_list.append(W_in)

        return sparse_reservoir_matrix_list, W_in_list

    def set_sigma(self, sigma):
        self.W_in = self.W_in * (sigma / self.sigma)
        self.sigma = sigma

    def set_radius(self, radius):
        self.s_reservoir_matrix = self.s_reservoir_matrix * (radius / self.radius)
        self.radius = radius

    def set_solver(self, solver):
        self.solver = solver
        if isinstance(solver, str):
            try:
                self.solver_op = getattr(self, "_" + solver)
            except Exception:
                raise Exception(f"The option {solver} is not implemented")
        else:
            print(
                f"The object {solver} provided is not an attribute of {self}.\
                    Let us consider it as a functional wrapper.\
                    Good luck in using it."
            )

            if callable(solver):
                self.solver_op = solver
            else:
                raise RuntimeError(f"object {solver} is not callable.")

    # It solves a linear system via direct matrix inversion
    def _direct_inversion(self, data_out_, U, r_til):
        print("Solving the system via direct matrix inversion.")

        Uinv = np.linalg.inv(U)

        Wout = (data_out_ @ r_til.T) @ Uinv

        return Wout

    # It solves a linear system using the SciPy algorithms
    # Note that U is symmetric positive definite
    def _linear_system(self, data_out_, U, r_til):
        print("Solving the linear system using the most proper algorithm.")

        Wout = linalg.solve(U, (r_til @ data_out_.T)).T

        return Wout

    def _construct_global_matrix_direct(self, r_til, idenmat):
        U = np.zeros((r_til.shape[0], r_til.shape[0]))

        for ii in range(r_til.shape[1]):
            U += np.outer(r_til[:, ii], r_til[:, ii])

        U += idenmat

        return U

    def _construct_global_matrix_numba(self, r_til, idenmat):
        U = np.zeros((r_til.shape[0], r_til.shape[0]))

        dot = outer_dot(engine="numba")
        U = dot(r_til, U)

        U += idenmat

        return U

    def _construct_global_matrix_multiprocessing(self, r_til, idenmat):
        assert self.n_workers is not None, (
            "If you are using multiprocessing," "it is necessary to define n_workers."
        )

        assert self.memory_percent is not None, (
            "If you are using multiprocessing,"
            "it is necessary to define memory_percent."
        )

        single_matrix_size = self.size_default * r_til.shape[0] ** 2

        available_memory = self.memory_percent * psutil.virtual_memory().available

        # Evaluating the maximum number of partial U matrices to be allocated
        max_n_workers = int(available_memory / single_matrix_size)

        if max_n_workers >= self.n_workers:
            n_workers = self.n_workers
        else:
            n_workers = max_n_workers

        r_til_chunks = np.array_split(r_til, n_workers, axis=1)

        pool = Pool(processes=self.n_workers)

        U_chunks = pool.map(multiprocessing_dot, r_til_chunks)

        U = sum(U_chunks)

        U += idenmat

        return U

    # Setting up new parameters for a restored model
    def set_parameters(self, parameters):
        for key, value in parameters.items():
            if key in [
                "radius",
                "sigma",
                "tau",
                "activation",
                "transformation",
                "solver",
            ]:
                getattr(self, f"set_{key}")(value)
            else:
                setattr(self, key, value)

    # It constructs each layerwise states array and executes the W_out fitting
    def fit(self, input_data=None, target_data=None):
        data_in = input_data.T
        data_out = target_data.T

        if not self.estimate_bias_transition and not self.estimate_linear_transition:
            # Do not estimate linear of bias terms
            pass
        elif self.estimate_bias_transition and not self.estimate_linear_transition:
            print("Estimating the bias")
            if self.A is not None:
                self.b = np.linalg.lstsq(
                    np.ones((target_data.shape[0], 1), dtype=target_data.dtype),
                    target_data - (self.A @ data_in).T,
                    rcond=None,
                )[0]
            else:
                self.b = np.linalg.lstsq(
                    np.ones((target_data.shape[0], 1), dtype=target_data.dtype),
                    target_data,
                    rcond=None,
                )[0]
            self.b = np.reshape(self.b, (-1,))
        elif not self.estimate_bias_transition and self.estimate_linear_transition:
            print("Estimating the linear transition")
            if self.b is not None:
                self.A = np.linalg.lstsq(
                    input_data, target_data - np.reshape(self.b, (1, -1)), rcond=None
                )[0].T
            else:
                self.A = np.linalg.lstsq(input_data, target_data, rcond=None)[0].T
        elif self.estimate_bias_transition and self.estimate_linear_transition:
            print("Estimating the linear transition and bias")
            self.A, self.b = np.hsplit(
                np.linalg.lstsq(
                    np.hstack((input_data, np.ones((target_data.shape[0], 1)))),
                    target_data,
                    rcond=None,
                )[0].T,
                [input_data.shape[1]],
            )
            self.b = np.reshape(self.b, (-1,))
        else:
            raise RuntimeError("Unreachable line of code")

        affine_term = 0
        if self.A is not None:
            affine_term += self.A @ data_in
        if self.b is not None:
            affine_term += np.reshape(self.b, (-1, 1))

        if self.global_state_dim > 0:
            data_out = data_out - affine_term

            data_size = data_in.shape[1]

            print("Evaluating the hidden state ...")

            r = self._reservoir_layer(data_in, data_size)

            print("\nApplying transformation ...")

            r_til = self._construct_r_til(r[: self.global_reservoir_dim, ...])
            if self.input_augmented_reservoir:
                r_til = np.vstack((r_til, r[self.global_reservoir_dim :, ...]))

            # Verifying the adjusting array kappa
            if isinstance(self.kappa, np.ndarray):
                self.kappa = np.reshape(self.kappa, [1, -1])
                assert (
                    self.kappa.shape[1] == r_til.shape[1]
                ), f"kappa and r-til must have the same samples dimension \
                but sizes are {self.kappa.shape[1]} and {r_til.shape[1]}"
            elif self.tau is not None and self.tau > 0:
                self.kappa = self._default_kappa(r_til.shape[1])
            else:
                self.kappa = np.ones((1, r_til.shape[1]))

            r_til *= self.kappa
            data_out_ = self.kappa * data_out

            idenmat = self.beta * sparse.identity(self.global_state_dim)

            print("Constructing W_out ...")

            # Constructing and solving the global linear system
            U = self.global_matrix_constructor(r_til, idenmat)

            Wout = self.solver_op(data_out_, U, r_til)

            self.W_out = np.array(Wout)

            self.current_state = r[:, -1].copy()
            self.reference_state = r[:, -1].copy()
        else:
            self.W_out = np.zeros(
                (target_data.shape[1], self.global_state_dim), dtype=target_data.dtype
            )
            self.current_state = np.zeros(
                (self.global_state_dim,), dtype=target_data.dtype
            )
            self.reference_state = np.zeros(
                (self.global_state_dim,), dtype=target_data.dtype
            )

        print("Fitting concluded.")

    def step(self, data=None):
        affine_term = 0
        if self.A is not None:
            affine_term += self.A @ data
        if self.b is not None:
            affine_term += self.b

        state = self.current_state.copy()

        hidden_states = list()
        data_ = data

        for ll in range(self.n_layers):
            r_ = self._update_state(state, data_, layer_id=ll)

            if self.all_for_input:
                data_ = np.hstack([r_, data])
            else:
                data_ = r_

            hidden_states.append(r_)

        global_r_ = np.hstack(hidden_states)

        state_ = self._construct_r_til(global_r_[: self.global_reservoir_dim, ...])
        if self.input_augmented_reservoir:
            state_ = np.hstack((state_, global_r_[self.global_reservoir_dim :, ...]))

        out = self.W_out @ state_ + affine_term

        self.current_state = global_r_.copy()

        return out

    def state_dim(self, layer_id=None):
        if self.input_augmented_reservoir:
            return self.reservoir_dim[layer_id] + self.number_of_inputs
        else:
            return self.reservoir_dim[layer_id]

    @property
    def global_state_dim(self):
        if self.input_augmented_reservoir:
            return self.global_reservoir_dim + self.number_of_inputs
        else:
            return self.global_reservoir_dim

    @property
    def default_state(self):
        return np.zeros((self.state_dim(0),), dtype=self.s_reservoir_matrix[0].dtype)

    def set_reference(self, reference=None):
        if reference is None:
            self.reference_state = self.default_state
        else:
            if isinstance(reference, (int, float)):
                reference = np.full_like(self.default_state, reference)
            assert np.array_equal(
                (self.state_dim,), reference.shape
            ), f"Invalid shape {(self.state_dim, )} != {reference.shape}"
            self.reference_state = reference

    def reset(self):
        if self.reference_state is None:
            self.reference_state = self.default_state
        self.current_state = self.reference_state

    def predict(self, initial_data=None, horizon=None):
        output_data = np.zeros((self.W_out.shape[0], horizon))

        state = self.current_state.copy()  # End of fit stage

        data = initial_data

        for tt in range(horizon):
            print("Extrapolating for the timestep {}".format(tt))

            r_ = self._update_state(state, data)

            r_til = self._construct_r_til(r_[: self.reservoir_dim, ...])
            if self.input_augmented_reservoir:
                r_til = np.hstack((r_til, r_[self.reservoir_dim :, ...]))

            affine_term = 0
            if self.A is not None:
                affine_term += self.A @ data
            if self.b is not None:
                affine_term += self.b

            out = np.array(self.W_out) @ r_til + affine_term
            output_data[:, tt] = out

            state = r_
            data = out

        self.current_state = state

        return output_data.T

    def save(self, save_path=None, model_name=None):
        configs = {
            k: getattr(self, k) for k in signature(self.__init__).parameters.keys()
        }

        to_save = {
            "current_state": self.current_state,
            "reference_state": self.reference_state,
            "W_out": self.W_out,
            "configs": configs,
        }
        path = os.path.join(save_path, model_name + ".pkl")
        try:
            with open(path, "wb") as fp:
                pickle.dump(to_save, fp, protocol=4)
        except Exception as e:
            print(e, e.args)

    @classmethod
    def restore(cls, model_path, model_name):
        path = os.path.join(model_path, model_name + ".pkl")
        with open(path, "rb") as fp:
            d = pickle.load(fp)

        self = cls(**d["configs"])
        d.pop("configs")
        for k, v in d.items():
            setattr(self, k, v)

        return self


# It corresponds to the grouped DeepESN
class WideEchoStateNetwork:

    """
    Echo State Network, a subclass of the Reservoir Computing methods
    """

    def __init__(
        self,
        reservoir_dim=None,
        sparsity_level=None,
        n_layers=2,
        radius=None,
        number_of_inputs=None,
        sigma=None,
        beta=None,
        kappa=None,
        leak_rate=1,
        activation="tanh",
        Win_interval=[-1, 1],
        tau=None,
        Win_init="global",
        transformation="T1",
        solver="direct_inversion",
        s_reservoir_matrix=None,
        W_in=None,
        eta=0,
        global_matrix_constructor_str="multiprocessing",
        A=None,
        b=None,
        estimate_linear_transition=False,
        estimate_bias_transition=False,
        n_workers=1,
        memory_percent=0.5,
        layerwise_train=None,
        show_log=False,
        input_augmented_reservoir=False,
        model_id=None,
    ):
        """
        :param reservoir_dim: dimension of the reservoir matrix
        :type reservoir_dim: int
        :param sparsity_level: level of sparsity at the matrices initialization
        :type sparsity_level: float
        :param radius:
        :param number_of_inputs: number of variables used as input
        :type number_of_inputs: int
        :param sigma: multiplicative constant for the interval of the input matrix initialization
        :type sigma: float
        :param beta: Ridge regularization parameter
        :type beta: float
        :param kappa:
        :param tau:
        :param activation:
        :param Win_interval:
        :param transformation:
        :param model_id:
        """

        self.sparsity_tolerance = 0.0025  # Default choice
        self.n_layers = n_layers

        self.reservoir_dim = self._make_list(reservoir_dim)
        self.sparsity_level = self._make_list(sparsity_level)
        self.radius = self._make_list(radius)
        self.sigma = self._make_list(sigma)
        self.beta = beta
        self.leak_rate = self._make_list(leak_rate)

        self.number_of_inputs = number_of_inputs
        self.n_inputs_list = self._make_n_inputs_list()

        self.set_activation(activation)
        self.Win_interval = Win_interval
        self.set_tau(tau)
        self.kappa = kappa
        self.Win_init = Win_init
        self.set_transformation(transformation)
        self.eta = eta
        self.global_matrix_constructor_str = global_matrix_constructor_str
        self.n_workers = n_workers
        self.memory_percent = memory_percent
        self.layerwise_train = layerwise_train
        self.input_augmented_reservoir = input_augmented_reservoir
        self.show_log = show_log

        self.s_reservoir_matrix = None
        self.W_in = None
        self.W_out = None
        self.current_state = None
        self.reference_state = None
        self.A = A
        self.b = b
        self.estimate_linear_transition = estimate_linear_transition
        self.estimate_bias_transition = estimate_bias_transition

        self.model_id = model_id

        # Echo state networks does not use history during fitting
        self.depends_on_history = False

        # Linear system solver
        self.solver_op = None
        self.solver = None
        self.set_solver(solver)

        if self.global_matrix_constructor_str == "direct":
            self.global_matrix_constructor = self._construct_global_matrix_direct
        elif self.global_matrix_constructor_str == "numba":
            if platform.system() == "Darwin":
                warnings.warn(
                    "You are using MacOS. Numba have presented some bugs in this platform."
                )
                warnings.warn(
                    "You are using MacOS. If you got some issue, test to use multiprocessing instead."
                )

            self.global_matrix_constructor = self._construct_global_matrix_numba
        elif self.global_matrix_constructor_str == "multiprocessing":
            self.global_matrix_constructor = (
                self._construct_global_matrix_multiprocessing
            )
        else:
            raise Exception(
                f"Case {self.global_matrix_constructor_str} not available for creating the"
                f"global matrix."
            )

        # The matrices input-to-reservoir and reservoir can be
        # directly passed to the class instance
        if s_reservoir_matrix is None and W_in is None:
            print("Initializing ESN matrices ...")

            self.s_reservoir_matrix, self.W_in = self._initialize_parameters()
        else:
            self.s_reservoir_matrix = s_reservoir_matrix
            self.W_in = W_in

        self.size_default = np.array([0]).astype("float64").itemsize

        if show_log == True:
            self.logger = self._show_log
        else:
            self.logger = self._no_log

    def _make_n_inputs_list(self):
        n_inputs_list = self.n_layers * [self.number_of_inputs]

        return n_inputs_list

    def _make_list(self, var):
        if type(var) is list:
            return var
        elif type(var) in (str, float, int):
            return self.n_layers * [var]
        else:
            raise Exception(
                f"The variable var can be list, str, int or float but received {type(var)}."
            )

    @property
    def global_reservoir_dim(self):
        return sum(self.reservoir_dim)

    def _show_log(self, i):
        sys.stdout.write(f"\r state {i}")
        sys.stdout.flush()

    def _no_log(self, i):
        pass

    @property
    def trainable_variables(
        self,
    ):
        return {"A": self.A, "b": self.b, "W_out": self.W_out}

    def set_activation(self, activation):
        self.activation = activation
        self.activation_op = self._get_activation_function(self.activation)

    def _get_activation_function(self, activation):
        method = (
            getattr(np, activation, None)
            or getattr(np.math, activation, None)
            or getattr(self, "_" + activation, None)
        )
        if method:
            return method
        else:
            raise Exception(
                "The activation {} is still not available.".format(activation)
            )

    # Linear activation
    def _linear(self, data):
        return data

    # Sigmoid activation
    def _sigmoid(self, data):
        return 1 / (1 + np.exp(-data))

    """
        BEGIN Transformations that can be applied to the hidden state

        Note: A shift = 1 is applied (ii + 1) in order to respect the Python
        0-based system.
    """

    def set_transformation(self, transformation):
        self.transformation = transformation
        self._construct_r_til = getattr(self, "_" + transformation)

    # ByPass transformation
    def _T0(self, r):
        r_til = r.copy()

        return r_til

    def _T1(self, r):
        n_rows = r.shape[0]
        r_til = r.copy()

        for ii in range(n_rows):
            if (ii + 1) % 2 != 0:
                r_til[ii] = (r[ii] * r[ii]).copy()
            else:
                pass
        return r_til

    def _T2(self, r):
        n_rows = r.shape[0]
        r_til = r.copy()

        for ii in range(2, n_rows):
            if (ii + 1) % 2 != 0:
                r_til[ii] = r[ii - 1] * r[ii - 2]
            else:
                pass
        return r_til

    def _T3(self, r):
        n_rows = r.shape[0]
        r_til = r.copy()

        for ii in range(1, n_rows - 1):
            if (ii + 1) % 2 != 0:
                r_til[ii] = r[ii - 1] * r[ii + 1]
            else:
                pass
        return r_til

    """
            END Transformations that can be applied to the hidden state
    """

    def set_tau(self, tau):
        self.tau = tau
        self.kappa = None

    def _default_kappa(self, size):
        x = np.arange(size) / size

        # tau is a positive number in (0, inf] preferably close to 0, e.g., 0.1
        # y[0] == 0 and y is monotonically increasing, i.e, y[k+1] > y[k] and y[-1] < 1
        # we choose tau such that 0.99 ~= y[int(tau*size)]

        y = 1 - 10 ** (-2 * x / self.tau)
        y = np.reshape(y, [1, -1])
        return y

    def _update_state(self, previous_state, input_series_state, layer_id=None):
        reservoir_input = previous_state[: self.reservoir_dim[layer_id]]
        state = (1 - self.leak_rate[layer_id]) * reservoir_input
        state += self.leak_rate[layer_id] * self.activation_op(
            self.s_reservoir_matrix[layer_id].dot(reservoir_input)
            + np.dot(self.W_in[layer_id], input_series_state)
        )

        if self.input_augmented_reservoir:
            state = np.hstack((state, input_series_state))

        return state

    # TODO this stage is considerably expensive. It is necessary to fix it in anyway.
    def _reservoir_layer(self, input_series, data_size):
        r_states_list = self._initialize_r_states(data_size)

        hidden_states = list()
        input_series_ = input_series
        for ll in range(self.n_layers):
            print(f"Applying layer {ll}")
            r_state = np.zeros(
                (self.state_dim(ll),), dtype=self.s_reservoir_matrix[ll].dtype
            )
            r_states = r_states_list[ll]

            for i in range(data_size):
                r_states[:, i] = self._update_state(
                    r_state, input_series_[:, i], layer_id=ll
                )
                r_state = r_states[:, i]

                self.logger(i)

            hidden_states.append(r_states)

        return np.vstack(hidden_states)

    def _reservoir_dim_corrected_sparsity_level(
        self, sparsity_level=None, reservoir_dim=None
    ):
        # Guaranteeing a minimum sparsity tolerance
        dim = (
            max(sparsity_level, self.sparsity_tolerance)
            if reservoir_dim == 0
            else reservoir_dim
        )
        effective_sparsity = sparsity_level / dim
        if effective_sparsity < self.sparsity_tolerance:
            return self.sparsity_tolerance
        else:
            return sparsity_level / dim

    # It creates a sparse and randomly distributed reservoir matrix
    def _create_reservoir(self, sparsity_level=None, reservoir_dim=None):
        return sparse.rand(
            reservoir_dim,
            reservoir_dim,
            density=self._reservoir_dim_corrected_sparsity_level(
                sparsity_level=sparsity_level, reservoir_dim=reservoir_dim
            ),
        )

    def _initialize_r_states(self, data_size):
        r_states_list = list()

        for ll in range(self.n_layers):
            r_states = np.zeros((self.state_dim(ll), data_size))
            r_states_list.append(r_states)

        return r_states_list

    # This method is used for initializing a raw model
    def _initialize_layer_parameters(self, layer_id=None):
        sparsity_level = self.sparsity_level[layer_id]
        reservoir_dim = self.reservoir_dim[layer_id]
        radius = self.radius[layer_id]
        number_of_inputs = self.n_inputs_list[layer_id]
        sigma = self.sigma[layer_id]
        # Sparse uniform random matrix
        sparse_reservoir_matrix = self._create_reservoir(
            sparsity_level=sparsity_level, reservoir_dim=reservoir_dim
        )

        sparse_reservoir_matrix += self.eta * sparse.eye(reservoir_dim)

        dense_sparse_reservoir_matrix = sparse_reservoir_matrix.todense()

        # Normalizing with the maximum eigenvalue
        # TODO It could be done using a sparse matrix algorithm, but
        #  there is some issue with the SciPy sparse library. it is necessary to check it.
        if reservoir_dim == 0:
            max_eig = 1.0
        else:
            eigvals = eigs(
                dense_sparse_reservoir_matrix,
                k=1,
                which="LM",
                return_eigenvectors=False,
            )
            max_eig = np.max(np.abs(eigvals))

        sparse_reservoir_matrix = sparse_reservoir_matrix * (radius / max_eig)

        if self.Win_init == "global":
            Win = np.random.uniform(
                *self.Win_interval, size=[reservoir_dim, number_of_inputs]
            )

        # TODO the blockwise option can be slow. Maybe it can be parallelized.
        elif self.Win_init == "blockwise":
            n_b = int(reservoir_dim / number_of_inputs)
            Win = np.zeros((reservoir_dim, number_of_inputs))

            for j in range(self.number_of_inputs):
                np.random.seed(seed=j)
                Win[j * n_b : (j + 1) * n_b, j] = np.random.uniform(
                    *self.Win_interval, size=[1, n_b]
                )[0]
        else:
            raise Exception(
                f"The initialization method {self.Win_init} for W_in is not supported."
            )

        Win = sigma * Win

        return sparse_reservoir_matrix, np.array(Win)

    def _initialize_parameters(self):
        sparse_reservoir_matrix_list = list()
        W_in_list = list()

        for ll in range(self.n_layers):
            sparse_reservoir_matrix, W_in = self._initialize_layer_parameters(
                layer_id=ll
            )

            sparse_reservoir_matrix_list.append(sparse_reservoir_matrix)
            W_in_list.append(W_in)

        return sparse_reservoir_matrix_list, W_in_list

    def set_sigma(self, sigma):
        self.W_in = self.W_in * (sigma / self.sigma)
        self.sigma = sigma

    def set_radius(self, radius):
        self.s_reservoir_matrix = self.s_reservoir_matrix * (radius / self.radius)
        self.radius = radius

    def set_solver(self, solver):
        self.solver = solver
        if isinstance(solver, str):
            try:
                self.solver_op = getattr(self, "_" + solver)
            except Exception:
                raise Exception(f"The option {solver} is not implemented")
        else:
            print(
                f"The object {solver} provided is not an attribute of {self}.\
                    Let us consider it as a functional wrapper.\
                    Good luck in using it."
            )

            if callable(solver):
                self.solver_op = solver
            else:
                raise RuntimeError(f"object {solver} is not callable.")

    # It solves a linear system via direct matrix inversion
    def _direct_inversion(self, data_out_, U, r_til):
        print("Solving the system via direct matrix inversion.")

        Uinv = np.linalg.inv(U)

        Wout = (data_out_ @ r_til.T) @ Uinv

        return Wout

    # It solves a linear system using the SciPy algorithms
    # Note that U is symmetric positive definite
    def _linear_system(self, data_out_, U, r_til):
        print("Solving the linear system using the most proper algorithm.")

        Wout = linalg.solve(U, (r_til @ data_out_.T)).T

        return Wout

    def _construct_global_matrix_direct(self, r_til, idenmat):
        U = np.zeros((r_til.shape[0], r_til.shape[0]))

        for ii in range(r_til.shape[1]):
            U += np.outer(r_til[:, ii], r_til[:, ii])

        U += idenmat

        return U

    def _construct_global_matrix_numba(self, r_til, idenmat):
        U = np.zeros((r_til.shape[0], r_til.shape[0]))

        dot = outer_dot(engine="numba")
        U = dot(r_til, U)

        U += idenmat

        return U

    def _construct_global_matrix_multiprocessing(self, r_til, idenmat):
        assert self.n_workers is not None, (
            "If you are using multiprocessing," "it is necessary to define n_workers."
        )

        assert self.memory_percent is not None, (
            "If you are using multiprocessing,"
            "it is necessary to define memory_percent."
        )

        single_matrix_size = self.size_default * r_til.shape[0] ** 2

        available_memory = self.memory_percent * psutil.virtual_memory().available

        # Evaluating the maximum number of partial U matrices to be allocated
        max_n_workers = int(available_memory / single_matrix_size)

        if max_n_workers >= self.n_workers:
            n_workers = self.n_workers
        else:
            n_workers = max_n_workers

        r_til_chunks = np.array_split(r_til, n_workers, axis=1)

        pool = Pool(processes=self.n_workers)

        U_chunks = pool.map(multiprocessing_dot, r_til_chunks)

        U = sum(U_chunks)

        U += idenmat

        return U

    # Setting up new parameters for a restored model
    def set_parameters(self, parameters):
        for key, value in parameters.items():
            if key in [
                "radius",
                "sigma",
                "tau",
                "activation",
                "transformation",
                "solver",
            ]:
                getattr(self, f"set_{key}")(value)
            else:
                setattr(self, key, value)

    def fit(self, input_data=None, target_data=None):
        data_in = input_data.T
        data_out = target_data.T

        if not self.estimate_bias_transition and not self.estimate_linear_transition:
            # dont estimate linear of bias terms
            pass
        elif self.estimate_bias_transition and not self.estimate_linear_transition:
            print("Estimating the bias")
            if self.A is not None:
                self.b = np.linalg.lstsq(
                    np.ones((target_data.shape[0], 1), dtype=target_data.dtype),
                    target_data - (self.A @ data_in).T,
                    rcond=None,
                )[0]
            else:
                self.b = np.linalg.lstsq(
                    np.ones((target_data.shape[0], 1), dtype=target_data.dtype),
                    target_data,
                    rcond=None,
                )[0]
            self.b = np.reshape(self.b, (-1,))
        elif not self.estimate_bias_transition and self.estimate_linear_transition:
            print("Estimating the linear transition")
            if self.b is not None:
                self.A = np.linalg.lstsq(
                    input_data, target_data - np.reshape(self.b, (1, -1)), rcond=None
                )[0].T
            else:
                self.A = np.linalg.lstsq(input_data, target_data, rcond=None)[0].T
        elif self.estimate_bias_transition and self.estimate_linear_transition:
            print("Estimating the linear transition and bias")
            self.A, self.b = np.hsplit(
                np.linalg.lstsq(
                    np.hstack((input_data, np.ones((target_data.shape[0], 1)))),
                    target_data,
                    rcond=None,
                )[0].T,
                [input_data.shape[1]],
            )
            self.b = np.reshape(self.b, (-1,))
        else:
            raise RuntimeError("Unreachable line of code")

        affine_term = 0
        if self.A is not None:
            affine_term += self.A @ data_in
        if self.b is not None:
            affine_term += np.reshape(self.b, (-1, 1))

        if self.global_state_dim > 0:
            data_out = data_out - affine_term

            data_size = data_in.shape[1]

            print("Evaluating the hidden state ...")

            r = self._reservoir_layer(data_in, data_size)

            print("\nApplying transformation ...")

            r_til = self._construct_r_til(r[: self.global_reservoir_dim, ...])
            if self.input_augmented_reservoir:
                r_til = np.vstack((r_til, r[self.global_reservoir_dim :, ...]))

            # Verifying the adjusting array kappa
            if isinstance(self.kappa, np.ndarray):
                self.kappa = np.reshape(self.kappa, [1, -1])
                assert (
                    self.kappa.shape[1] == r_til.shape[1]
                ), f"kappa and r-til must have the same samples dimension \
                but sizes are {self.kappa.shape[1]} and {r_til.shape[1]}"
            elif self.tau is not None and self.tau > 0:
                self.kappa = self._default_kappa(r_til.shape[1])
            else:
                self.kappa = np.ones((1, r_til.shape[1]))

            r_til *= self.kappa
            data_out_ = self.kappa * data_out

            idenmat = self.beta * sparse.identity(self.global_state_dim)

            print("Constructing W_out ...")

            # Constructing and solving the global linear system
            U = self.global_matrix_constructor(r_til, idenmat)

            Wout = self.solver_op(data_out_, U, r_til)

            self.W_out = np.array(Wout)

            self.current_state = r[:, -1].copy()
            self.reference_state = r[:, -1].copy()
        else:
            self.W_out = np.zeros(
                (target_data.shape[1], self.global_state_dim), dtype=target_data.dtype
            )
            self.current_state = np.zeros(
                (self.global_state_dim,), dtype=target_data.dtype
            )
            self.reference_state = np.zeros(
                (self.global_state_dim,), dtype=target_data.dtype
            )

        print("Fitting concluded.")

    def step(self, data=None):
        affine_term = 0
        if self.A is not None:
            affine_term += self.A @ data
        if self.b is not None:
            affine_term += self.b

        state = self.current_state.copy()

        hidden_states = list()
        data_ = data

        for ll in range(self.n_layers):
            r_ = self._update_state(state, data_, layer_id=ll)

            hidden_states.append(r_)

        global_r_ = np.hstack(hidden_states)

        state_ = self._construct_r_til(global_r_[: self.global_reservoir_dim, ...])
        if self.input_augmented_reservoir:
            state_ = np.hstack((state_, global_r_[self.global_reservoir_dim :, ...]))

        out = self.W_out @ state_ + affine_term

        self.current_state = global_r_.copy()

        return out

    def state_dim(self, layer_id=None):
        if self.input_augmented_reservoir:
            return self.reservoir_dim[layer_id] + self.number_of_inputs
        else:
            return self.reservoir_dim[layer_id]

    @property
    def global_state_dim(self):
        if self.input_augmented_reservoir:
            return self.global_reservoir_dim + self.number_of_inputs
        else:
            return self.global_reservoir_dim

    @property
    def default_state(self):
        return np.zeros((self.state_dim(0),), dtype=self.s_reservoir_matrix[0].dtype)

    def set_reference(self, reference=None):
        if reference is None:
            self.reference_state = self.default_state
        else:
            if isinstance(reference, (int, float)):
                reference = np.full_like(self.default_state, reference)
            assert np.array_equal(
                (self.state_dim,), reference.shape
            ), f"Invalid shape {(self.state_dim, )} != {reference.shape}"
            self.reference_state = reference

    def reset(self):
        if self.reference_state is None:
            self.reference_state = self.default_state
        self.current_state = self.reference_state

    def predict(self, initial_data=None, horizon=None):
        output_data = np.zeros((self.W_out.shape[0], horizon))

        state = self.current_state.copy()  # End of fit stage

        data = initial_data

        for tt in range(horizon):
            print("Extrapolating for the timestep {}".format(tt))

            r_ = self._update_state(state, data)

            r_til = self._construct_r_til(r_[: self.reservoir_dim, ...])
            if self.input_augmented_reservoir:
                r_til = np.hstack((r_til, r_[self.reservoir_dim :, ...]))

            affine_term = 0
            if self.A is not None:
                affine_term += self.A @ data
            if self.b is not None:
                affine_term += self.b

            out = np.array(self.W_out) @ r_til + affine_term
            output_data[:, tt] = out

            state = r_
            data = out

        self.current_state = state

        return output_data.T

    def save(self, save_path=None, model_name=None):
        configs = {
            k: getattr(self, k) for k in signature(self.__init__).parameters.keys()
        }

        to_save = {
            "current_state": self.current_state,
            "reference_state": self.reference_state,
            "W_out": self.W_out,
            "configs": configs,
        }
        path = os.path.join(save_path, model_name + ".pkl")
        try:
            with open(path, "wb") as fp:
                pickle.dump(to_save, fp, protocol=4)
        except Exception as e:
            print(e, e.args)

    @classmethod
    def restore(cls, model_path, model_name):
        path = os.path.join(model_path, model_name + ".pkl")
        with open(path, "rb") as fp:
            d = pickle.load(fp)

        self = cls(**d["configs"])
        d.pop("configs")
        for k, v in d.items():
            setattr(self, k, v)

        return self


"""
    END Reservoir Computing (RC) and children classes
"""
