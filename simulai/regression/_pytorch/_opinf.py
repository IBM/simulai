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
from typing import Union

import numpy as np
import torch

from simulai.abstract import Regression
from simulai.regression import Linear
from simulai.templates import NetworkTemplate


# Neural network version of the OpInf algorithm
class OpInfNetwork(NetworkTemplate):
    def __init__(
        self,
        n_inputs: int = None,
        n_outputs: int = None,
        n_forcing: int = None,
        forcing: str = None,
        engine="torch",
        name: str = None,
        setup_architecture: bool = True,
    ) -> None:
        super(OpInfNetwork, self).__init__()

        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_forcing = n_forcing
        self.forcing = forcing
        self.engine = engine
        self.name = name

        if self.forcing == "linear":
            self.n_cross_variables = self.n_inputs
            self.prepare = self._with_stacking
            self.forward = self._forward_with_forcing
            self.eval = self._eval_with_forcing

        elif self.forcing == "nonlinear":
            self.n_cross_variables = self.n_inputs + self.n_forcing
            self.prepare = self._without_stacking
            self.forward = self._forward_with_forcing
            self.eval = self._eval_with_forcing

        elif self.forcing is None:
            self.n_cross_variables = self.n_inputs
            self.forward = self._forward_without_forcing
            self.eval = self._eval_without_forcing
            print("No forcing term is being used.")

        else:
            raise Exception(f"The option {self.forcing} for forcing is not supported.")

        self.i_u, self.j_u = np.triu_indices(self.n_cross_variables)
        self.n_quadratic_inputs = self.i_u.shape[0]

        if self.n_outputs is None:
            self.n_outputs = self.n_inputs

        if setup_architecture is True:
            self._setup_architecture()

    def _setup_architecture(self) -> None:
        # The bias c is placed in the operator Linear for A_op
        self.A_op = Linear(
            input_size=self.n_inputs, output_size=self.n_outputs, name="A_op"
        )  # Linear operator for the field variables
        self.H_op = Linear(
            input_size=self.n_quadratic_inputs,
            output_size=self.n_outputs,
            bias=False,
            name="H_op",
        )  # Linear operator for the quadratic terms

        if self.forcing is not None:
            self.B_op = Linear(
                input_size=self.n_forcing,
                output_size=self.n_outputs,
                bias=False,
                name="B_op",
            )  # Linear operator for the forcing variables
            self.operators = [self.A_op, self.B_op, self.H_op]
        else:
            self.operators = [self.A_op, self.H_op]

        # Setting up the network operators
        self._setup_operators()

    def _without_stacking(self, input_field: np.ndarray = None, **kwargs) -> np.ndarray:
        return input_field

    def _with_stacking(
        self, input_field: np.ndarray = None, input_forcing: np.ndarray = None
    ) -> np.ndarray:
        return np.hstack([input_field, input_forcing])

    # Adding each operator as submodule, according to the torch syntax
    def _setup_operators(self) -> None:
        for op in self.operators:
            self.add_module(self.name + "_" + op.name, op)

    @property
    def weights(self) -> list:
        return sum([net.weights for net in self.operators], [])

    @property
    def linear_weights_l2(self):
        return sum([torch.norm(weight, p=2) for weight in self.A_op.weights]) + sum(
            [torch.norm(bias, p=2) for bias in self.A_op.bias]
        )

    @property
    def forcing_liner_weights_l2(self):
        if self.forcing is not None:
            return sum([torch.norm(weight, p=2) for weight in self.B_op.weights])
        else:
            return 0.0

    @property
    def nonlinear_weights_l2(self):
        return sum([torch.norm(weight, p=2) for weight in self.H_op.weights])

    #############################################################################
    # The weights norms are customized for the OpInf case                       #
    # The L2-norm is dedicated to H_op and the L1 one to A_op                   #
    #############################################################################
    #
    @property  #
    def weights_l2(self) -> (torch.Tensor, torch.Tensor):  #
        #
        return (
            self.linear_weights_l2 + self.forcing_liner_weights_l2,  #
            self.nonlinear_weights_l2,
        )  #

    @property  #
    def weights_l1(self) -> float:  #
        return 0.0  #

    #############################################################################

    def _forward_with_forcing(
        self,
        input_field: Union[np.ndarray, torch.Tensor] = None,
        input_forcing: Union[np.ndarray, torch.Tensor] = None,
    ) -> torch.Tensor:
        A_output = self.A_op.forward(input_field)  # Matrix for the linear field terms
        B_output = self.B_op.forward(
            input_forcing
        )  # Matrix  for the linear forcing terms

        data = self.prepare(input_field=input_field, input_forcing=input_forcing)
        quadratic_inputs = self.kronecker_dot(data=data)
        H_output = self.H_op.forward(quadratic_inputs)

        return A_output + B_output + H_output

    def _forward_without_forcing(
        self, input_field: Union[np.ndarray, torch.Tensor] = None
    ) -> torch.Tensor:
        A_output = self.A_op.forward(input_field)  # Matrix for the linear field terms

        data = input_field
        quadratic_inputs = self.kronecker_dot(data=data)
        H_output = self.H_op.forward(quadratic_inputs)

        return A_output + H_output

    def _eval_with_forcing(
        self, input_data: np.ndarray = None, forcing_data: np.ndarray = None
    ) -> np.ndarray:
        output_tensor = self._forward_with_forcing(
            input_field=input_data, input_forcing=forcing_data
        )

        return output_tensor.detach().numpy()

    def _eval_without_forcing(self, input_data: np.ndarray = None) -> np.ndarray:
        output_tensor = self._forward_without_forcing(input_field=input_data)

        return output_tensor.detach().numpy()

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

    # The Kronecker dot is used for generating the quadratic component to be
    # used together the matrix H
    def kronecker_dot(
        self, data: Union[torch.Tensor, np.ndarray] = None
    ) -> torch.Tensor:
        if type(data) == np.ndarray:
            data_ = torch.from_numpy(data)
        else:
            data_ = data

        kronecker_output = torch.einsum("bi,bj->bij", data_, data_)

        return kronecker_output[:, self.i_u, self.j_u].float()

    ###############################################################
    # The forward method is executed via the self.forward attribute
    ###############################################################

    ###############################################################
    # The eval method is executed via the self.eval attribute
    ###############################################################

    def to_numpy(self):
        return NumpyOpInfNetwork(opinf_net=self, forcing=self.forcing)

    def _get_array_from_linear_operator(
        self, op: Linear = None, component: str = "weight"
    ) -> np.ndarray:
        assert type(op) is Linear, f"It is expected Linear but received type(op)."
        return getattr(op.layers[0], component).detach().numpy()

    @property
    def A_hat(self) -> np.ndarray:
        return self._get_array_from_linear_operator(self.A_op, "weight")

    @property
    def B_hat(self) -> np.ndarray:
        return self._get_array_from_linear_operator(self.B_op, "weight")

    @property
    def H_hat(self) -> np.ndarray:
        return self._get_array_from_linear_operator(self.H_op, "weight")

    @property
    def c_hat(self) -> np.ndarray:
        return self._get_array_from_linear_operator(self.A_op, "bias")


# Numpy version of the OpInf network
class NumpyOpInfNetwork(Regression):
    def __init__(self, opinf_net: torch.nn.Module = None, forcing: str = None) -> None:
        super().__init__()

        self.forcing = forcing

        self.A_op = opinf_net.A_op.to_numpy()
        self.H_op = opinf_net.H_op.to_numpy()
        self.K_op = opinf_net.K_op

        self.n_inputs = opinf_net.n_inputs

        if hasattr(opinf_net, "B_op"):
            self.B_op = opinf_net.B_op.to_numpy()
        else:
            self.B_op = None

        if self.forcing is None:
            self.n_cross_variables = self.n_inputs
            self.forward = self._forward_without_forcing
            self.eval = self._eval_without_forcing
            print("No forcing term is being used.")

            self.operators = [self.A_op, self.H_op]

        else:
            raise Exception(f"The option {self.forcing} for forcing is not supported.")

        self.i_u, self.j_u = opinf_net.i_u, opinf_net.j_u

    def _builtin_jacobian(self, x):
        return self.A_op.weights + (self.K_op @ x.T)

    def _external_jacobian(self, x):
        return self.jacobian_op(x)

    def construct_K_op(self, op: callable = None):
        if op is None:
            self.jacobian = self._builtin_jacobian

        else:
            self.jacobian_op = op

            self.jacobian = self._external_jacobian

    def parametric_jacobian(self, input_data: np.ndarray = None):
        A_jac = np.tile(input_data, (self.n_inputs, 1))
        c_jac = np.eye(self.n_inputs)

        quadratic_input_data = self.kronecker_dot(data=input_data)
        H_jac = np.tile(quadratic_input_data, (self.n_inputs, 1))

        return np.hstack([A_jac, c_jac, H_jac])

    # The Kronecker dot is used for generating the quadratic component to be
    # used together the matrix H
    def kronecker_dot(self, data: np.ndarray = None) -> np.ndarray:
        kronecker_output = np.einsum("bi,bj->bij", data, data)

        return kronecker_output[:, self.i_u, self.j_u]

    def _forward_without_forcing(self, input_field: np.ndarray = None) -> np.ndarray:
        A_output = self.A_op.forward(input_field)  # Matrix for the linear field terms

        data = input_field
        quadratic_inputs = self.kronecker_dot(data=data)
        H_output = self.H_op.forward(quadratic_inputs)

        return A_output + H_output

    def _eval_without_forcing(self, input_field: np.ndarray = None) -> np.ndarray:
        output_tensor = self._forward_without_forcing(input_field=input_field)

        return output_tensor

    # Saving to disk the complete model
    def save(self, save_path: str = None, model_name: str = None) -> None:
        path = os.path.join(save_path, model_name + ".pkl")
        try:
            with open(path, "wb") as fp:
                pickle.dump(self, fp, protocol=4)
        except Exception as e:
            print(e, e.args)

    def _get_array_from_linear_operator(
        self, op=None, component: str = "weights"
    ) -> np.ndarray:
        return getattr(op, component)

    @property
    def A_hat(self) -> np.ndarray:
        return self._get_array_from_linear_operator(self.A_op, "weights")

    @property
    def B_hat(self) -> np.ndarray:
        return self._get_array_from_linear_operator(self.B_op, "weights")

    @property
    def H_hat(self) -> np.ndarray:
        return self._get_array_from_linear_operator(self.H_op, "weights")

    @property
    def c_hat(self) -> np.ndarray:
        return self._get_array_from_linear_operator(self.A_op, "bias")
