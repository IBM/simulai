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
from typing import List, Union

import numpy as np
import torch

from simulai.abstract import Regression
from simulai.math.expressions import FromSymbol2FLambda
from simulai.templates import NetworkTemplate, as_tensor, guarantee_device

from ._opinf import OpInfNetwork


# Wrapper class for constructing Koopman operators using tensor algebra
class KoopmanNetwork(OpInfNetwork):
    def __init__(
        self,
        n_inputs: int = None,
        observables: List[str] = None,
        fobservables: List[str] = None,
        variables: List[str] = None,
        intervals: List[Union[int, list]] = None,
        fintervals: List[Union[int, list]] = None,
        operator_config: dict = None,
    ) -> None:
        self.observables_expressions = observables
        self.fobservables_expressions = fobservables

        self.built_observables = False

        if intervals is None:
            intervals = len(observables) * [-1]

        assert len(intervals) == len(
            observables
        ), "The number of intervals must be the same number of observables"

        # Intervals for indicating the modes in which each observable must be applied
        self.intervals = [self._construct_intervals(ii) for ii in intervals]

        if fintervals is not None:
            self.fintervals = [self._construct_intervals(ii) for ii in fintervals]
        else:
            self.fintervals = None

        if variables is not None:
            self.variables = variables
        else:
            self.variables = ["x"]

        self._build_observables_spaces(
            observables=self.observables_expressions,
            fobservables=self.fobservables_expressions,
        )

        n_inputs_ext = self._get_n_inputs(n_inputs)
        operator_config["n_inputs"] = n_inputs_ext
        operator_config["n_outputs"] = n_inputs

        self.black_list = ["observables", "fobservables", "funcgen", "all_observables"]

        super().__init__(**operator_config)

        self._setup_architecture()

    def _get_n_inputs(self, n_inputs: int = None) -> int:
        test = np.ones(n_inputs)[None, :]
        return self._generate_observables(
            data=test, observables=self.observables
        ).shape[1]

    def _build_observables_spaces(
        self, observables: List[str] = None, fobservables: List[str] = None
    ) -> None:
        # Generating observables lambda functions from string expressions
        # Some of this objects are not pickleable, so they must re-generated when necessary

        self.funcgen = FromSymbol2FLambda(variables=self.variables)

        self.observables = [
            self.funcgen.convert(ob) for ob in observables
        ]  # Observables for the field variables

        self.fobservables = fobservables  # Observables for the forcing variables

        if self.fobservables is not None:
            self.fobservables = [self.funcgen.convert(ob) for ob in self.fobservables]
            self.all_observables = self.observables + self.fobservables
        else:
            self.all_observables = self.observables

        # Checking up if all the observables are callable functions
        assert all(
            [callable(ob) for ob in self.all_observables]
        ), f"All the observable must be callable, but received {self.all_observables}"

        self.built_observables = True

    def _construct_intervals(self, interval: Union[int, list] = None) -> slice:
        if interval == -1:
            return slice(0, None)
        elif type(interval) and len(interval) == 2:
            return slice(*interval)
        elif type(interval) and len(interval) == 1:
            return slice(interval[0], None)
        else:
            raise Exception(
                f"It is expected 'interval' to be int or list, but received {type(interval)}."
            )

    # Generating the observables data using a properly defined transformation
    def _generate_observables(
        self, data: np.ndarray = None, observables: list = None
    ) -> np.ndarray:
        # The object data is guaranteed as a 2D matrix
        return np.hstack(
            [ob(data[:, self.intervals[oi]]) for oi, ob in enumerate(observables)]
        )

    def _forward_with_forcing(
        self,
        input_field: Union[np.ndarray, torch.Tensor] = None,
        input_forcing: Union[np.ndarray, torch.Tensor] = None,
    ) -> torch.Tensor:
        forcing_observables = self._generate_observables(
            data=input_forcing, observables=self.fobservables
        )
        input_observables = self._generate_observables(
            data=input_field, observables=self.observables
        )

        A_output = self.A_op.forward(
            input_observables
        )  # Matrix for the linear field terms
        B_output = self.B_op.forward(
            forcing_observables
        )  # Matrix  for the linear forcing terms

        data = self.prepare(
            input_field=input_observables, input_forcing=forcing_observables
        )
        quadratic_inputs = self.kronecker_dot(data=data)
        H_output = self.H_op.forward(quadratic_inputs)

        return A_output + B_output + H_output

    def _forward_without_forcing(
        self, input_field: Union[np.ndarray, torch.Tensor] = None
    ) -> torch.Tensor:
        input_observables = self._generate_observables(
            data=input_field, observables=self.observables
        )

        A_output = self.A_op.forward(
            input_observables
        )  # Matrix for the linear field terms

        data = input_observables
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
        if len(x.shape) == 1:
            x = x[None, :]

        x_observables = self._generate_observables(data=x, observables=self.observables)

        return self.A_hat + (self.K_op @ x_observables[0].T)

    def eval(self, input_data: np.ndarray = None, **kwargs) -> np.ndarray:
        input_observables = self._generate_observables(
            data=input_data, observables=self.observables
        )

        return super().eval(input_data=input_observables, **kwargs)

    def save(self, save_path: str = None, model_name: str = None) -> None:
        for item in self.black_list:
            setattr(self, item, None)

        self.built_observables = False

        super().save(save_path=save_path, model_name=model_name)

    def lean_save(self, save_path: str = None, model_name: str = None) -> None:
        for item in self.black_list:
            setattr(self, item, None)

        self.built_observables = False

        super().lean_save(save_path=save_path, model_name=model_name)


# Neural network version of the Koopman operator theory
class AutoEncoderKoopman(NetworkTemplate):
    name = "autoencoderkoopman"
    engine = "torch"

    def __init__(
        self,
        encoder: NetworkTemplate = None,
        decoder: NetworkTemplate = None,
        devices: Union[str, list] = "cpu",
        opinf_net: NetworkTemplate = None,
    ) -> None:
        super(AutoEncoderKoopman, self).__init__()

        # Determining the kind of device to be used for allocating the
        # subnetworks used in the DeepONet model
        if type(devices) == list():
            raise Exception("In construction.")
        elif type(str):
            if devices == "gpu":
                if torch.cuda.is_available():
                    self.device = "cuda"
                else:
                    self.device = "cpu"
            else:
                self.device = "cpu"

        self.encoder = encoder.to(self.device)
        self.decoder = decoder.to(self.device)
        self.opinf_net = opinf_net.to(self.device)

        # The encoder output must be one-dimensional
        assert self._latent_dimension_is_correct(self.encoder.output_size), (
            "The trunk network must have"
            " one-dimensional output , "
            " but received"
            f"{self.trunk_network.output_size}"
        )

        # The decoder input must be one-dimensional
        assert self._latent_dimension_is_correct(self.decoder.input_size), (
            "The branch network must have"
            " one-dimensional output,"
            " but received"
            f"{self.branch_network.input_size}"
        )

    def _latent_dimension_is_correct(self, dim: Union[int, tuple]) -> bool:
        if type(dim) == int:
            return True
        elif type(dim) == tuple:
            if len(tuple) == 1:
                return True
            else:
                return False

    @as_tensor
    def forward(
        self, input_data: Union[torch.Tensor, np.ndarray] = None
    ) -> torch.Tensor:
        encoder_output = self.encoder.forward(input_data=input_data)
        opinf_output = self.opinf_net.forward(input_field=encoder_output)
        decoder_output = self.decoder.forward(input_data=opinf_output)

        return decoder_output

    @property
    def weights_l2(self) -> (torch.Tensor, torch.Tensor):
        return (
            (self.encoder.weights_l2,)
            + self.opinf_net.weights_l2
            + (self.decoder.weights_l2,)
        )

    @property
    def weights_l1(self) -> float:
        return 0.0

    @guarantee_device
    def eval(self, input_data: Union[torch.Tensor, np.ndarray] = None) -> np.ndarray:
        output = self.forward(input_data=input_data)

        return output.cpu().detach().numpy()


# Numpy version of the OpInf network
class NumpyKoopmanNetwork(Regression):
    def __init__(self, opinf_net: torch.nn.Module = None, forcing: str = None) -> None:
        super().__init__()

        self.forcing = forcing

        self.A_op = opinf_net.A_op.to_numpy()
        self.H_op = opinf_net.H_op.to_numpy()
        self.K_op = opinf_net.K_op

        self.n_inputs = opinf_net.n_inputs

        self._build_observables_spaces(
            observables=self.observables_expressions,
            fobservables=self.fobservables_expressions,
        )

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
