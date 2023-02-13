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

import numpy as np
import sympy as sp
from sympy import MatrixExpr


class OpInfDeviation:
    def __init__(self, A_hat: np.ndarray = None, H_hat: np.ndarray = None) -> None:
        """Evaluating the deviation evolution in an OpInf model
        :param A_hat: linear OpInf operator
        :type A_hat: np.ndarray
        :param H_hat: quadratic OpInf operator
        :type: np.ndarray
        :return: nothing
        """

        self.A_hat = A_hat
        self.H_hat = H_hat

        self.n = A_hat.shape[0]

        epsilon = sp.MatrixSymbol("epsilon", self.n, 1)
        u = sp.MatrixSymbol("u", self.n, 1)

        # u*e.T
        ue = sp.MatMul(u, epsilon.T)
        # (u*e.T).T
        ue_T = ue.T
        # e*e.T
        ee = sp.MatMul(epsilon, epsilon.T)

        # U(u*e.T + (u*e.T).T) + e X e
        v_u = np.array(ue + ue_T + ee)[np.triu_indices(self.n)]
        v = sp.Matrix(v_u)

        H_symb = sp.Matrix(self.H_hat)
        A_symb = sp.Matrix(self.A_hat)

        v_jac = sp.Matrix.jacobian(H_symb @ v, epsilon)

        # A + H*(U(u*e.T + (u*e.T).T) + e X e)
        self.jac_expressions = A_symb + v_jac
        self.error_expressions = A_symb @ epsilon + H_symb @ v

        self.epsilon = epsilon
        self.u = u

        self.jac = None
        self.error = None

        self.compile()

    def compile(self) -> None:
        self.jac = self.lambdify(expression=self.jac_expressions)
        self.error = self.lambdify(expression=self.error_expressions)

    def lambdify(self, expression: MatrixExpr = None) -> callable:
        return sp.lambdify([self.epsilon, self.u], expression, "numpy")

    def eval_jacobian(
        self, u: np.ndarray = None, epsilon: np.ndarray = None
    ) -> np.ndarray:
        """Evaluating error Jacobian
        :param u: reference solution
        :type u: np.ndarray
        :param epsilon: error associated to u
        :type epsilon: np.ndarray
        :return: error Jacobian
        :rtype: np.ndarray
        """

        u = u.T
        epsilon = epsilon.T

        return self.jac(epsilon, u)

    def eval_error(self, u: np.ndarray = None, epsilon: np.array = None) -> np.ndarray:
        """Evaluating error
        :param u: reference solution
        :type u: np.ndarray
        :param epsilon: error associated to u
        :type epsilon: np.ndarray
        :return: error
        :rtype: np.ndarray
        """

        u = u.T
        epsilon = epsilon.T

        return self.error(epsilon, u)

    def save(self, name: str = None, path: str = None) -> None:
        """Complete saving
        :param path: path to the saving directory
        :type path: str
        :param name: name for the model
        :type name: str
        :return: nothing
        """

        blacklist = ["jac", "error"]

        for item in blacklist:
            delattr(self, item)

        with open(os.path.join(path, name + ".pkl"), "wb") as fp:
            pickle.dump(self, fp, protocol=4)
