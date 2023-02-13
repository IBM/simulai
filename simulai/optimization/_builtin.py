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

import sys

import numpy as np


# Sparse Regression Algorithm
class SpaRSA:
    def __init__(
        self,
        lambd: float = None,
        alpha_0: float = None,
        epsilon: float = 1e-10,
        sparsity_tol: float = 1e-15,
        use_mean: bool = False,
        transform: callable = None,
    ) -> None:
        self.lambd = lambd
        self.alpha_0 = alpha_0
        self.epsilon = epsilon
        self.sparsity_tol = sparsity_tol
        self.size = 1

        if transform is not None:
            self.transform = transform
        else:
            self.transform = self._bypass

        if use_mean is True:
            self.norm = lambda x: x / self.size
        else:
            self.norm = lambda x: x

        self.m = 0
        self.r = 0
        self.ref_step = 5
        self.lr_reduction = 1 / 2
        self.lr_increase = 3 / 2

        self.W = None
        self.target_data = None

    def _bypass(self, data: np.ndarray) -> np.ndarray:
        return data

    def _F_lambda(self, V_bar: np.ndarray = None) -> np.ndarray:
        residual = (
            np.linalg.norm(self._WV_bar(W=self.W, V_bar=V_bar) - self.target_data, None)
            ** 2
        )

        regularization = self.lambd * np.sum(np.linalg.norm(V_bar, 2, axis=1))

        return (1 / 2) * residual + self.lambd * regularization

    def R_alpha(
        self,
        W: np.ndarray = None,
        V_bar: np.ndarray = None,
        target_data: np.ndarray = None,
        alpha: float = 0,
    ) -> np.ndarray:
        return V_bar - alpha * W.T @ (self._WV_bar(W=W, V_bar=V_bar) - target_data)

    def _WV_bar(self, W: np.ndarray = None, V_bar: np.ndarray = None) -> np.ndarray:
        return W @ V_bar

    def _no_null_V_plus(
        self, R_alpha: np.ndarray = None, alpha: float = 0
    ) -> np.ndarray:
        return (1 - self.lambd * alpha / np.linalg.norm(R_alpha, None)) * R_alpha

    def V_plus(self, R_alpha: np.ndarray = None, alpha: float = None):
        # Zeroing lines according to the regularization criteria
        def _row_function(vector: np.ndarray = None) -> np.ndarray:
            norm = np.linalg.norm(vector, None)

            if norm <= self.lambd * alpha:
                return np.zeros(vector.shape)
            else:
                return self._no_null_V_plus(R_alpha=vector, alpha=alpha)

        rows = np.apply_along_axis(_row_function, 1, R_alpha)

        return rows

    def fit(
        self, input_data: np.ndarray = None, target_data: np.ndarray = None
    ) -> None:
        self.W = self.transform(data=input_data)
        self.target_data = target_data

        self.q = self.W.shape[-1]
        self.m = target_data.shape[-1]
        self.size = self.target_data.size

        V_0 = np.random.rand(self.q, self.m)

        V_k = V_0
        F_lambda_list = list()

        alpha = self.alpha_0
        stopping_criterion = False
        k = 0

        while not stopping_criterion:
            V_bar = V_k
            R_alpha = self.R_alpha(
                W=self.W, V_bar=V_bar, target_data=target_data, alpha=alpha
            )
            V_plus = self.V_plus(R_alpha=R_alpha, alpha=alpha)

            F_lambda_V_plus = self._F_lambda(V_bar=V_plus)
            F_lambda_V_bar = self._F_lambda(V_bar=V_bar)

            while F_lambda_V_plus >= F_lambda_V_bar:
                residual = F_lambda_V_plus - F_lambda_V_bar

                sys.stdout.write(
                    ("\ralpha: {}, discrepancy: {}").format(alpha, residual)
                )
                sys.stdout.flush()

                alpha = alpha * self.lr_reduction
                R_alpha = self.R_alpha(
                    W=self.W, V_bar=V_bar, target_data=target_data, alpha=alpha
                )
                V_plus = self.V_plus(R_alpha=R_alpha, alpha=alpha)

                F_lambda_V_plus = self._F_lambda(V_bar=V_plus)

            F_lambda = F_lambda_V_bar
            F_lambda_list.append(F_lambda)

            V_k = V_plus
            alpha = min(self.lr_increase * alpha, self.alpha_0)

            if k > self.ref_step:
                F_lambda_ref = F_lambda_list[-self.ref_step - 1]

                if np.abs(F_lambda - F_lambda_ref) / F_lambda_ref <= self.epsilon:
                    stopping_criterion = True

            sys.stdout.write(("\rresidual loss: {}").format(self.norm(F_lambda)))
            sys.stdout.flush()

            k += 1

        V_k = np.where(np.abs(V_k) < self.sparsity_tol, 0, V_k)

        return V_k
