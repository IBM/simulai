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
from typing import Optional

import numpy as np
from scipy.linalg import solve as solve


class ELM:
    def __init__(
        self,
        n_i: int = None,
        h: int = None,
        n_o: int = None,
        activation="tanh",
        form: str = "primal",
        solver: str = "lstsq",
    ) -> None:
        """Extreme Learning Machine

        :param n_i: number of input features
        :type n_i: int
        :param n_o: number of output features
        :type n_o: int
        :param h: number of hidden cells
        :type h: int
        :param activation: activation to be used in the hidden layer
        :type activation: str
        :param form: the kind of approach, primal or dual
        :type form:str
        :param solver: the linear system solver to be used
        :type solver: str
        """

        self.n_i = n_i
        self.h = h
        self.n_o = n_o
        self.n_samples = None

        self.activation = activation
        self.form = form
        self.solver = solver

        self.W_i = np.random.standard_normal((self.h, self.n_i))
        self.b = np.random.standard_normal((1, self.h))

        self.W_o = None

        self.L_operator = None
        self.R_matrix = None

        self._matrix = getattr(self, "_" + self.form + "_matrix")
        self._right_side = getattr(self, "_" + self.form + "_right_side")

        if self.activation in dir(np):
            self.activation_func = getattr(np, self.activation)
        elif "_" + self.activation in dir(self):
            self.activation_func = getattr(self, "_" + self.activation)
        else:
            raise Exception(f"It was not possible to find the actvation {activation}.")

    def _sigmoid(self, input_data: np.ndarray = None) -> np.ndarray:
        return 1 / (1 + np.exp(-input_data))

    def _is_symmetric(self, matrix: np.ndarray = None) -> bool:
        """It checks if the system matrix is symmetric

        :param matrix: the global system matrix
        :type matrix: np.ndarray
        :return: Is the matrix symmetric ? True or False
        :rtype: bool
        """

        return np.array_equal(matrix, matrix.T)

    def f_h(self, input_data: np.ndarray = None) -> np.ndarray:
        """Evaluating hidden state
        :param input_data: dataset for the input data
        :type input_data: np.ndarray
        :return: Hidden state
        :rtype: np.ndarray
        """

        return self.activation_func(input_data @ self.W_i.T + self.b)

    def _primal_matrix(self, H: np.ndarray = None) -> np.ndarray:
        """Primal version of the linear system matrix
        :param H: hidden state matrix
        :type H: np.ndarray
        :return: primal matrix
        :rtype: np.ndarray
        """

        return H.T @ H

    def _dual_matrix(self, H: np.ndarray = None) -> np.ndarray:
        """Dual version of the linear system matrix
        :param H: hidden state matrix
        :type H: np.ndarray
        :return: dual matrix
        :rtype: np.ndarray
        """

        return H @ H.T

    def _primal_right_side(
        self, H: np.ndarray = None, target_data: np.ndarray = None
    ) -> np.ndarray:
        """Primal version of the right-hand side
        :param H: hidden state matrix
        :type H: np.ndarray
        :param target_data: target dataset
        :type target_data: np.ndarray
        :return: primal right-hand side
        :rtype: np.ndarray
        """

        return H.T @ target_data

    def _dual_right_side(
        self, H: Optional[np.ndarray] = None, target_data: np.ndarray = None
    ) -> np.ndarray:
        """Primal version of the right-hand side
        :param target_data: target dataset
        :type target_data: np.ndarray
        :return: dual right-hand side
        :rtype: np.ndarray
        """

        return target_data

    def fit(
        self,
        input_data: np.ndarray = None,
        target_data: np.ndarray = None,
        lambd: float = 0,
    ) -> None:
        """Fitting the ELM
        :param input_data: dataset for the input data
        :type input_data: np.ndarray
        :param target_data: dataset for the target data
        :type target_data: np.ndarray
        :param lambd: regularization penalty
        :type lambd: float
        :return: nothing
        """

        H = self.f_h(input_data=input_data)
        self.n_samples = input_data.shape[0]

        if self.form == "primal":
            sys_dim = self.h
        else:
            sys_dim = self.n_samples

        if self.solver != "pinv":
            if self.L_operator is None and self.R_matrix is None:
                self.L_operator = self._matrix(H=H)
                self.R_matrix = self._right_side(H=H, target_data=target_data)

            else:
                pass

            if self._is_symmetric(self.L_operator) and self.solver is None:
                print("L_operator is symmetric.")
                solution = solve(
                    self.L_operator + lambd * np.eye(sys_dim),
                    self.R_matrix,
                    assume_a="sym",
                )
            else:
                solution = np.linalg.lstsq(
                    self.L_operator + lambd * np.eye(sys_dim), self.R_matrix, rcond=None
                )[0]
        else:
            H_pinv = np.linalg.pinv(H)
            solution = H_pinv @ target_data

        if self.form == "dual":
            solution = H.T @ solution

        self.W_o = solution

    def eval(self, input_data: np.ndarray = None) -> np.ndarray:
        """Evaluating using ELM
        :param input_data: dataset for the input data
        :type input_data: np.ndarray
        :return: the output evaluation
        :rtype: np.ndarray
        """

        H = self.f_h(input_data=input_data)

        return H @ self.W_o

    def save(self, name: str = None, path: str = None) -> None:
        """Complete saving
        :param path: path to the saving directory
        :type path: str
        :param name: name for the model
        :type name: str
        :return: nothing
        """

        with open(os.path.join(path, name + ".pkl"), "wb") as fp:
            pickle.dump(self, fp, protocol=4)
