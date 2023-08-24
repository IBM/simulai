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
from typing import Callable, List, Tuple, Union
from time import sleep

import numpy as np
import torch
import torch.nn.functional as F

from simulai import ARRAY_DTYPE
from simulai.io import IntersectingBatches
from simulai.models import AutoencoderKoopman, AutoencoderVariational, DeepONet
from simulai.residuals import SymbolicOperator
from simulai.optimization._adjusters import AnnealingWeights

class LossBasics:

    def __init__(self):
        """
        Loss functions parent class
        """
        self.loss_states = None
        self.tol = 1e-16

    def _single_term_loss(self, res:torch.Tensor) -> torch.Tensor:

        return torch.square(res)

    def _two_term_loss(self, res_a:torch.Tensor, res_b:torch.Tensor) -> torch.Tensor:

        return torch.square(res_a - res_b)

    def _two_term_log_loss(self, res_a:torch.Tensor, res_b:torch.Tensor) -> torch.Tensor:

        if torch.all(res_a <= self.tol) or torch.all(res_b <= self.tol):

            return self._two_term_loss(res_a, res_b)
        else:
            return torch.square(torch.log(res_a) - torch.log(res_b))

    # Choosing the kind of multiplication to be done for each
    # type of lambda penalties and regularization terms
    def _exec_multiplication_in_regularization(
        self, lambda_type: type, term_type: type
    ) -> Callable:
        """
        It executes the multiplications involved in the construction of the L*-penalty terms

        :param lambda_type: the kind of lambda to be used, float or Tuple(float)
        :type: Union[float, Tuple(float)]
        :param term_type: the kind of term to be used
        :type term_type: Union[float, Tuple(float)]
        :returns: a function properly configured to execute the multiplication
        :rtype: Callable

        """

        if (lambda_type, term_type) in [(float, torch.Tensor), (float, float)]:

            def multiplication(lambd, term):
                return lambd * term

            return multiplication

        elif (lambda_type, term_type) == (tuple, tuple):

            def multiplication(lambd, term):
                return sum([ll * tt for ll, tt in zip(lambd, term)])

            return multiplication

        elif (lambda_type, term_type) == (float, tuple):

            def multiplication(lambd, term):
                lambd_ = len(term) * (lambd,)
                return sum([ll * tt for ll, tt in zip(lambd_, term)])

            return multiplication

        else:
            raise Exception(
                "lambda_weight and term must be both float or both"
                f" tuple but received {(lambda_type, term_type)}"
            )

    @staticmethod
    def _eval_weighted_loss(losses:List[torch.tensor], weights:List[float]) -> torch.tensor:

        residual_loss = [
                weight * loss
                for weight, loss in zip(weights, losses)
            ]

        return [sum(residual_loss)]

    @staticmethod
    def _bypass_weighted_loss(losses:List[torch.tensor], *args) -> torch.tensor:

        return losses

    @staticmethod
    def _aggregate_terms(*args) -> List[torch.tensor]:

        return list(args)

    @staticmethod
    def _formatter(value:torch.tensor=None, n_decimals:int=2) -> str:

        value = torch.tensor(value)

        if value > 0:

            exp = float(torch.log10(value))
            base = int(exp)
            res = round(exp - base, n_decimals)

            return f"{round(10**res, n_decimals)}e{base}"
        else:
            return str(round(float(value)))

    @staticmethod
    def _pprint_simple(loss_str:str=None,
                       losses_list:List[torch.tensor]=None,
                       call_back:str=None,
                       loss_indices:List[int]=None, **kwargs) -> None:

       sys.stdout.write((loss_str).format(*losses_list[loss_indices]) + call_back)

       sys.stdout.flush()

    def _pprint_verbose(self, loss_terms:List[torch.tensor]=None, loss_weights:List[torch.tensor]=None, **kwargs) -> None:

        terms_str_list = [f"|L_{i}: {{}} | w_{i}: {{}}|" for i in range(len(loss_terms))] 

        formatted_loss_terms = [self._formatter(value=l) for l in loss_terms]
        formatted_weights = [self._formatter(value=w) for w in loss_weights]

        terms_list = [str_term.format(l, w) for str_term, l, w in zip(terms_str_list,
                                                                      formatted_loss_terms,
                                                                      formatted_weights)]

        print((len(terms_list))*"\033[F" + '\n'.join(terms_list), end='\n', flush=True)

# Classic RMSE Loss with regularization for PyTorch
class RMSELoss(LossBasics):
    def __init__(self, operator: torch.nn.Module = None) -> None:
        """
        Vanilla mean-squared error loss function

        :param operator: the operator used for evaluating the loss function (usually a neural network)
        :type operator: torch.nn.Module

        """
        super().__init__()

        self.operator = operator
        self.loss_states = {"loss": list()}

    def _data_loss(
        self,
        output_tilde: torch.Tensor = None,
        norm_value: torch.Tensor = None,
        target_data_tensor: torch.Tensor = None,
    ) -> torch.Tensor:
        """

        It executes the evaluation of the data-driven mean-squared error

        :param output_tilde: the output generated by self.operator
        :type output_tilde: torch.Tensor
        :param norm_value: the value used for normalizing the loss evaluation
        :type norm_value: torch.Tensor
        :param target_data_tensor: the target tensor to be compared with output_tilde
        :type target_data_tensor: torch.Tensor
        :returns: the loss function value for a given state
        :rtype: torch.Tensor

        """

        if norm_value is not None:
            data_loss = torch.mean(
                torch.square((output_tilde - target_data_tensor) / norm_value)
            )
        else:
            data_loss = torch.mean(torch.square((output_tilde - target_data_tensor)))

        return data_loss

    def __call__(
        self,
        input_data: Union[dict, torch.Tensor] = None,
        target_data: torch.Tensor = None,
        call_back: str = "",
        norm_value: list = None,
        lambda_1: float = 0.0,
        device: str = "cpu",
        lambda_2: float = 0.0,
    ) -> Callable:
        """
        Main function for generating complete loss function workflow

        :param input_data: the data used as input for self.operator
        :type input_data: Union[dict, torch.Tensor]
        :param target_data: the target data used for training self.oeprator
        :type target_data: torch.Tensor
        :param call_back: a string used for composing the logging of the optimization process
        :type call_back:str
        :param norm_value: a list of values used for normalizing the loss temrms
        :type norm_value: list
        :param lambda_1: the penalty for the L^1  regularization term
        :type lambda_1: float
        :param lambda_2: the penalty for the L^2  regularization term
        :type lambda_2: float
        :param device: the device in which the loss evaluation will be executed, 'cpu' or 'gpu'
        :type device: str
        :returns: the closure function used for evaluating the loss value
        :rtype: Callable

        """

        l1_reg_multiplication = self._exec_multiplication_in_regularization(
            lambda_type=type(lambda_1), term_type=type(self.operator.weights_l1)
        )

        l2_reg_multiplication = self._exec_multiplication_in_regularization(
            lambda_type=type(lambda_2), term_type=type(self.operator.weights_l2)
        )

        def closure():
            output_tilde = self.operator.forward(**input_data)

            data_loss = self._data_loss(
                output_tilde=output_tilde,
                norm_value=norm_value,
                target_data_tensor=target_data,
            )

            # L² and L¹ regularization term
            weights_l2 = self.operator.weights_l2
            weights_l1 = self.operator.weights_l1

            # beta *||W||_2 + alpha * ||W||_1
            l2_reg = l2_reg_multiplication(lambda_2, weights_l2)
            l1_reg = l1_reg_multiplication(lambda_1, weights_l1)

            # Loss = ||Ũ_t - U_t||_2  +
            #         lambda_1 *||W||_2 + lambda_2 * ||W||_1

            loss = data_loss + l2_reg + l1_reg

            # Back-propagation
            loss.backward()

            self.loss_states["loss"].append(float(loss.detach().data))

            sys.stdout.write(("\rloss: {} {}").format(loss, call_back))
            sys.stdout.flush()

            return loss

        return closure


# Weighted RMSE Loss with regularization for PyTorch
class WRMSELoss(LossBasics):
    def __init__(self, operator=None):
        """
        Weighted mean-squared error loss function

        :param operator: the operator used for evaluating the loss function (usually a neural network)
        :type operator: torch.nn.Module

        """

        super().__init__()

        self.operator = operator
        self.split_dim = 1
        self.tol = 1e-25

        self.loss_evaluator = None
        self.norm_evaluator = None

        self.axis_loss_evaluator = lambda res: torch.mean(torch.square((res)), dim=1)

        self.loss_states = {"loss": list()}

    def _data_loss(
        self,
        output_tilde: torch.Tensor = None,
        weights: list = None,
        target_data_tensor: torch.Tensor = None,
        axis: int = -1,
    ) -> List:
        """

        It executes the evaluation of the data-driven mean-squared error

        :param output_tilde: the output generated by self.operator
        :type output_tilde: torch.Tensor
        :param norm_value: the value used for normalizing the loss evaluation
        :type norm_value: torch.Tensor
        :param target_data_tensor: the target tensor to be compared with output_tilde
        :type target_data_tensor: torch.Tensor
        :returns: the loss function value for a given state
        :rtype: torch.Tensor

        """

        output_split = torch.split(output_tilde, self.split_dim, dim=axis)
        target_split = torch.split(target_data_tensor, self.split_dim, dim=axis)

        data_losses = [
            weights[i]
            * self.loss_evaluator(out_split - tgt_split)
            / self.norm_evaluator(tgt_split)
            for i, (out_split, tgt_split) in enumerate(zip(output_split, target_split))
        ]

        return data_losses

    def _no_data_loss_wrapper(
        self,
        output_tilde: torch.Tensor = None,
        weights: list = None,
        target_data_tensor: torch.Tensor = None,
        axis: int = -1,
    ) -> torch.Tensor:
        """

        It executes the evaluation of the data-driven mean-squared error without considering causality preserving

        :param output_tilde: the output generated by self.operator
        :type output_tilde: torch.Tensor
        :param weights: weights for rescaling each variable outputted by self.operator
        :type weights: list
        :param target_data_tensor: the target tensor to be compared with output_tilde
        :type target_data_tensor: torch.Tensor
        :param axis: the axis in which the variables are split
        :type axis: int
        :returns: the loss function value for a given state
        :rtype: torch.Tensor

        """

        return self.data_loss(
            output_tilde=output_tilde,
            weights=weights,
            target_data_tensor=target_data_tensor,
            axis=axis,
        )

    def __call__(
        self,
        input_data: Union[dict, torch.Tensor] = None,
        target_data: torch.Tensor = None,
        call_back: str = "",
        lambda_1: float = 0.0,
        lambda_2: float = 0.0,
        axis: int = -1,
        relative: bool = False,
        device: str = "cpu",
        weights: list = None,
        use_mean: bool = True,
    ) -> Callable:
        """

        Main function for generating complete loss function workflow

        :param input_data: the data used as input for self.operator
        :type input_data: Union[dict, torch.Tensor]
        :param target_data: the target data used for training self.oeprator
        :type target_data: torch.Tensor
        :param call_back: a string used for composing the logging of the optimization process
        :type call_back:str
        :param norm_value: a list of values used for normalizing the loss temrms
        :type norm_value: list
        :param lambda_1: the penalty for the L^1  regularization term
        :type lambda_1: float
        :param lambda_2: the penalty for the L^2  regularization term
        :type lambda_2: float
        :param device: the device in which the loss evaluation will be executed, 'cpu' or 'gpu'
        :type device: str
        :param weights: a list of weights for rescaling each variable outputted by self.operator
        :type weights: list
        :param use_mean: use mean for evaluating the losses or not (the alternative is sum)
        :type use_mean: bool
        :returns: the closure function used for evaluating the loss value
        :rtype: Callable

        """

        self.data_loss = self._data_loss

        l1_reg_multiplication = self._exec_multiplication_in_regularization(
            lambda_type=type(lambda_1), term_type=type(self.operator.weights_l1)
        )

        l2_reg_multiplication = self._exec_multiplication_in_regularization(
            lambda_type=type(lambda_2), term_type=type(self.operator.weights_l2)
        )

        # Using mean evaluation or not
        if use_mean == True:
            self.loss_evaluator = lambda res: torch.mean(torch.square((res)))
        else:
            self.loss_evaluator = lambda res: torch.sum(torch.square((res)))

        # Relative norm or not
        if relative == True:
            if use_mean == True:
                self.norm_evaluator = lambda ref: torch.mean(torch.square((ref)))
            else:
                self.norm_evaluator = lambda ref: torch.sum(torch.square((ref)))
        else:
            self.norm_evaluator = lambda ref: 1

        self.data_loss_wrapper = self._no_data_loss_wrapper

        def closure():
            output_tilde = self.operator.forward(**input_data)

            data_losses = self.data_loss_wrapper(
                output_tilde=output_tilde,
                weights=weights,
                target_data_tensor=target_data,
                axis=axis,
            )

            # L² and L¹ regularization term
            weights_l2 = self.operator.weights_l2
            weights_l1 = self.operator.weights_l1

            # beta *||W||_2 + alpha * ||W||_1
            l2_reg = l2_reg_multiplication(lambda_2, weights_l2)
            l1_reg = l1_reg_multiplication(lambda_1, weights_l1)

            # Loss = ||Ũ_t - U_t||_2  +
            #         lambda_1 *||W||_2 + lambda_2 * ||W||_1
            loss = sum(data_losses) + l2_reg + l1_reg

            # Back-propagation
            loss.backward()

            self.loss_states["loss"].append(float(loss.detach().data))

            sys.stdout.write(("\rloss: {} {}").format(loss, call_back))
            sys.stdout.flush()

        return closure


# RMSE Loss for equation-based residuals
class PIRMSELoss(LossBasics):
    def __init__(self, operator: torch.nn.Module = None) -> None:
        """
        Physics-Informed mean-squared error loss function

        :param operator: the operator used for evaluating the loss function (usually a neural network)
        :type operator: torch.nn.Module

        """

        super().__init__()

        self.split_dim = 1
        self.operator = operator
        self.loss_evaluator = None
        self.residual = None
        self.tol = 1e-15
        self.device = None

        self.axis_loss_evaluator = lambda res: torch.mean(torch.square((res)), dim=1)

        self.loss_states = {
            "pde": list(),
            "init": list(),
            "bound": list(),
            "extra_data": list(),
        }
        self.loss_tags = list(self.loss_states.keys())
        self.hybrid_data_pinn = False

        self.losses_terms_indices = {
            "pde": 0,
            "init": 1,
            "bound": 2,
            "extra_data": 3,
            "causality_weights": 4,
        }

    def _convert(
        self, input_data: Union[dict, np.ndarray] = None, device: str = None
    ) -> Union[dict, torch.Tensor]:
        """

        It converts a dataset to the proper format (torch.Tensor) and send it to
        the chosen execution device ('gpu' or 'cpu')

        :param input_data: the data structure to be converted
        :type input_data: Union[dict, np.ndarray]
        :param device: the device in which the converted dataset must be placed
        :returns: the converted data structure
        :rtype: Union[dict, torch.Tensor]

        """

        if type(input_data) == dict:
            return {
                key: torch.from_numpy(item.astype(ARRAY_DTYPE)).to(device)
                for key, item in input_data.items()
            }

        else:
            return torch.from_numpy(input_data.astype(ARRAY_DTYPE)).to(device)

    def _to_tensor(self, *args, device: str = "cpu") -> List[torch.Tensor]:
        """

        It converted a size indefined list of arrays to tensors

        :param *args: list of arrays to be converted
        :type: np.array, np.array, ..., np.array
        :type input_data: Union[dict, np.ndarray]
        :param device: the device in which the converted dataset must be placed
        :returns: a list of tensors
        :rtype: List[torch.Tensor]

        """
        return [self._convert(input_data=arg, device=device) for arg in args]

    def _data_loss(
        self, output_tilde: torch.Tensor = None,
              target_data_tensor: torch.Tensor = None,
              weights: List[float]=None,
    ) -> torch.Tensor:
        """

        It executes the evaluation of the data-driven mean-squared error

        :param output_tilde: the output generated by self.operator
        :type output_tilde: torch.Tensor
        :param target_data_tensor: the target tensor to be compared with output_tilde
        :type target_data_tensor: torch.Tensor
        :returns: the loss function value
        :rtype: torch.Tensor

        """

        output_split = torch.split(output_tilde, self.split_dim, dim=-1)
        target_split = torch.split(target_data_tensor, self.split_dim, dim=-1)

        data_losses = [
            self.loss_evaluator_data((out_split, tgt_split)) / (self.norm_evaluator(tgt_split) or torch.tensor(1.0).to(self.device))
            for i, (out_split, tgt_split) in enumerate(zip(output_split, target_split))
        ]

        return self.weighted_loss_evaluator(data_losses, weights)

    def _data_loss_adaptive(
        self, output_tilde: torch.Tensor = None,
              target_data_tensor: torch.Tensor = None,
              **kwargs,
    ) -> torch.Tensor:
        """

        It executes the evaluation of the data-driven mean-squared error

        :param output_tilde: the output generated by self.operator
        :type output_tilde: torch.Tensor
        :param target_data_tensor: the target tensor to be compared with output_tilde
        :type target_data_tensor: torch.Tensor
        :returns: the loss function value
        :rtype: torch.Tensor

        """

        output_split = torch.split(output_tilde, self.split_dim, dim=-1)
        target_split = torch.split(target_data_tensor, self.split_dim, dim=-1)

        data_discrepancy = [
            out_split - tgt_split
            for i, (out_split, tgt_split) in enumerate(zip(output_split, target_split))
        ]

        weights = self.data_weights_estimator(residual=data_discrepancy,
                                         loss_evaluator=self.loss_evaluator,
                                         loss_history=self.loss_states,
                                         operator=self.operator)

        data_losses = [
            weights[i]*self.loss_evaluator_data((out_split, tgt_split))
            for i, (out_split, tgt_split) in enumerate(zip(output_split, target_split))
        ]

        return [sum(data_losses)]

    def _global_weights_bypass(self, initial_penalty:float=None, **kwargs) -> List[float]:

        return [1.0, initial_penalty, 1.0, 1.0]

    def _global_weights_estimator(self, **kwargs) -> List[float]:

        weights = self.global_weights_estimator(**kwargs)

        return weights

    def _residual_loss(
        self, residual_approximation: List[torch.Tensor] = None, weights: list = None
    ) -> List[torch.Tensor]:
        """

        It evaluates the physics-driven residual loss

        :param residual_approximation: a list of tensors containing the evaluation for
                                       the physical residual for each sample in the dataset
        :type residual_approximation: List[torch.Tensor]
        :param weights: a list of weights used for rescaling the residuals of each variable
        :type weights: list
        :returns: the list of residual losses
        :rtype: torch.Tensor

        """
        residual_losses = [
            self.loss_evaluator(res)
            for res in residual_approximation
        ]

        return self.weighted_loss_evaluator(residual_losses, weights)

    def _residual_loss_adaptive(
        self, residual_approximation: List[torch.Tensor] = None, weights: list = None
    ) -> List[torch.Tensor]:
        """

        It evaluates the physics-driven residual loss

        :param residual_approximation: a list of tensors containing the evaluation for
                                       the physical residual for each sample in the dataset
        :type residual_approximation: List[torch.Tensor]
        :param weights: a list of weights used for rescaling the residuals of each variable
        :type weights: list
        :returns: the list of residual losses
        :rtype: torch.Tensor

        """

        weights = self.residual_weights_estimator(residual=residual_approximation,
                                         loss_evaluator=self.loss_evaluator,
                                         loss_history=self.loss_states,
                                         operator=self.operator)

        residual_loss = [
            weight * self.loss_evaluator(res)
            for weight, res in zip(weights, residual_approximation)
        ]

        return [sum(residual_loss)]

    def _extra_data(
        self, input_data: torch.Tensor = None, target_data: torch.Tensor = None
    ) -> torch.Tensor:
        # Evaluating data for the initial condition
        output_tilde = self.operator(input_data=input_data)

        # Evaluating loss approximation for extra data
        data_loss = self._data_loss(
            output_tilde=output_tilde, target_data_tensor=target_data
        )

        return data_loss

    def _boundary_penalisation(
        self, boundary_input: dict = None, residual: SymbolicOperator = None
    ) -> List[torch.Tensor]:
        """

        It applies the boundary conditions

        :param boundary_input: a dictionary containing the coordinates of the boundaries
        :type boundary_input:dict
        :param residual: a symbolic expression for the boundary condition
        :type residual: SymbolicOperator
        :returns: the evaluation of each boundary condition
        :rtype: list

        """
        return [
            residual.eval_expression(k, boundary_input[k])
            for k in boundary_input.keys()
        ]

    def _no_boundary_penalisation(
        self, boundary_input: dict = None, residual: object = None
    ) -> List[torch.Tensor]:
        """

        It is used for cases in which no boundary condition is applied

        """

        return [torch.Tensor([0.0]).to(self.device) for k in boundary_input.keys()]

    def _no_boundary(
        self, boundary_input: dict = None, residual: object = None
    ) -> List[torch.Tensor]:
        """

        It is used for cases where there are not boundaries

        """

        return torch.Tensor([0.0]).to(self.device)

    def _no_extra_data(
        self, input_data: torch.Tensor = None, target_data: torch.Tensor = None
    ) -> torch.Tensor:
        return torch.Tensor([0.0]).to(self.device)

    def _no_residual_wrapper(self, input_data: torch.Tensor = None) -> torch.Tensor:
        return self.residual(input_data)

    def _causality_preserving_residual_wrapper(
        self, input_data: torch.Tensor = None
    ) -> List:
        return self.causality_preserving(self.residual(input_data))

    def _filter_necessary_loss_terms(self, residual: SymbolicOperator = None):
        tags = ["pde", "init"]
        indices = [0, 1]

        if residual.g_expressions:
            tags.append("bound")
            indices.append(2)
        else:
            pass

        if self.hybrid_data_pinn:
            tags.append("extra_data")
            indices.append(3)
        else:
            pass

        return tags, indices

    def _losses_states_str(self, tags: List[str] = None):
        losses_str = "\r"
        for item in tags:
            losses_str += f"{item}:{{}} "

        return losses_str

    def __call__(
        self,
        input_data: Union[dict, torch.Tensor] = None,
        target_data: Union[dict, torch.Tensor] = None,
        verbose:bool=False,
        call_back: str = "",
        residual: Callable = None,
        initial_input: Union[dict, torch.Tensor] = None,
        initial_state: Union[dict, torch.Tensor] = None,
        boundary_input: dict = None,
        boundary_penalties: list = [1],
        extra_input_data: Union[dict, torch.Tensor] = None,
        extra_target_data: Union[dict, torch.Tensor] = None,
        initial_penalty: float = 1,
        axis: int = -1,
        relative: bool = False,
        lambda_1: float = 0.0,
        lambda_2: float = 0.0,
        weights=None,
        weights_residual=None,
        device: str = "cpu",
        split_losses: bool = False,
        causality_preserving: Callable = None,
        global_weights_estimator: Callable = None,
        residual_weights_estimator: Callable = None,
        data_weights_estimator: Callable = None,
        use_mean: bool = True,
        use_data_log: bool = False,
    ) -> Callable:
        self.residual = residual

        self.device = device

        self.causality_preserving = causality_preserving

        # Handling expection when AnnealingWeights and split_losses
        # are used together.
        if isinstance(global_weights_estimator, AnnealingWeights):
            if split_losses:
                raise RuntimeError("Global weights estimator, AnnealingWeights, is not"+\
                                   "compatible with split loss terms.")
            else:
                pass

        self.global_weights_estimator = global_weights_estimator

        self.residual_weights_estimator = residual_weights_estimator

        self.data_weights_estimator = data_weights_estimator

        if split_losses:
            self.weighted_loss_evaluator = self._bypass_weighted_loss
        else:
            self.weighted_loss_evaluator = self._eval_weighted_loss

        if (
            isinstance(extra_input_data, np.ndarray)
            == isinstance(extra_target_data, np.ndarray)
            == True
        ):
            self.hybrid_data_pinn = True
        else:
            pass

        # When no weight is provided, they are
        # set to the default choice
        if weights is None:
            weights = len(residual.output_names) * [1]

        if weights_residual is None:
            weights_residual = len(residual.output_names) * [1]

        loss_tags, loss_indices = self._filter_necessary_loss_terms(residual=residual)
        loss_str = self._losses_states_str(tags=loss_tags)

        # Boundary conditions are optional, since they are not
        # defined in some cases, as ODE, for example.
        if residual.g_expressions:
            boundary = self._boundary_penalisation
        else:
            if boundary_input == None:
                boundary = self._no_boundary
            else:
                boundary = self._no_boundary_penalisation

        if self.causality_preserving:
            call_back = f", causality_weights: {self.causality_preserving.call_back}"
            self.residual_wrapper = self._causality_preserving_residual_wrapper

        else:
            self.residual_wrapper = self._no_residual_wrapper

        l1_reg_multiplication = self._exec_multiplication_in_regularization(
            lambda_type=type(lambda_1), term_type=type(self.operator.weights_l1)
        )

        l2_reg_multiplication = self._exec_multiplication_in_regularization(
            lambda_type=type(lambda_2), term_type=type(self.operator.weights_l2)
        )
        if type(input_data) is dict:
            try:
                input_data = input_data["input_data"]
            except Exception:
                pass

        initial_input, initial_state = self._to_tensor(
            initial_input, initial_state, device=device
        )

        # Preparing extra data, when necessary
        if self.hybrid_data_pinn:
            extra_input_data, extra_target_data = self._to_tensor(
                extra_input_data, extra_target_data, device=device
            )
            self.extra_data = self._extra_data
        else:
            self.extra_data = self._no_extra_data

        if use_data_log == True:
            self.inner_square = self._two_term_log_loss
        else:
            self.inner_square = self._two_term_loss

        if use_mean == True:
            self.loss_evaluator = lambda res: torch.mean(self._single_term_loss(res))
        else:
            self.loss_evaluator = lambda res: torch.sum(self._single_term_loss(res))

        if use_mean == True:
            self.loss_evaluator_data = lambda res: torch.mean(self.inner_square(*res))
        else:
            self.loss_evaluator_data = lambda res: torch.sum(self.inner_square(*res))

        # Relative norm or not
        if relative == True:
            if use_mean == True:
                self.norm_evaluator = lambda ref: torch.mean(torch.square((ref)))
            else:
                self.norm_evaluator = lambda ref: torch.sum(torch.square((ref)))
        else:
            self.norm_evaluator = lambda ref: 1

        # Determing the usage of special residual loss weighting
        if residual_weights_estimator:
            self.residual_loss = self._residual_loss_adaptive
        else:
            self.residual_loss = self._residual_loss

        # Determing the usage of special data loss weighting
        if data_weights_estimator:
            self.data_loss = self._data_loss_adaptive
        else:
            self.data_loss = self._data_loss

        # Determining the usage of special global loss weighting
        if global_weights_estimator:
            self.global_weights = self._global_weights_estimator
        else:
            self.global_weights = self._global_weights_bypass

        if verbose:
            self.pprint = self._pprint_verbose
        else:
            self.pprint = self._pprint_simple

        def closure():
            # Executing the symbolic residual evaluation
            residual_approximation = self.residual_wrapper(input_data)

            # Boundary, if appliable
            boundary_approximation = boundary(
                boundary_input=boundary_input, residual=residual
            )

            # Evaluating data for the initial condition
            initial_output_tilde = self.operator(input_data=initial_input)

            # Evaluating loss function for residual
            residual_loss = self.residual_loss(
                residual_approximation=residual_approximation, weights=weights_residual
            )

            # Evaluating loss for the boundary approaximation, if appliable
            boundary_loss = self._residual_loss(
                residual_approximation=boundary_approximation,
                weights=boundary_penalties,
            )

            # Evaluating loss approximation for initial condition
            initial_data_loss = self.data_loss(
                output_tilde=initial_output_tilde,
                target_data_tensor=initial_state, weights=weights,
            )

            # Evaluating extra data loss, when appliable
            extra_data = self.extra_data(
                input_data=extra_input_data, target_data=extra_target_data
            )

            # L² and L¹ regularization term
            weights_l2 = self.operator.weights_l2
            weights_l1 = self.operator.weights_l1

            # beta *||W||_2 + alpha * ||W||_1
            l2_reg = l2_reg_multiplication(lambda_2, weights_l2)
            l1_reg = l1_reg_multiplication(lambda_1, weights_l1)

            # The complete loss function
            pde = residual_loss
            init = initial_data_loss
            bound = boundary_loss

            loss_terms = self._aggregate_terms(*pde, *init, *bound, *extra_data)

            # Updating the loss weights if necessary
            loss_weights = self.global_weights(initial_penalty=initial_penalty,
                                               operator=self.operator,
                                               loss_evaluator=self.loss_evaluator,
                                               residual=loss_terms)
            # Overall loss function
            loss = sum(self._eval_weighted_loss(loss_terms, loss_weights)) + l2_reg + l1_reg

            # Back-propagation
            loss.backward()

            pde_detach = float(sum(pde).detach().data)
            init_detach = float(sum(init).detach().data)
            bound_detach = float(sum(bound).detach().data)
            extra_data_detach = float(sum(extra_data).detach().data)

            self.loss_states["pde"].append(pde_detach)
            self.loss_states["init"].append(init_detach)
            self.loss_states["bound"].append(bound_detach)
            self.loss_states["extra_data"].append(extra_data_detach)

            losses_list = np.array(
                [pde_detach, init_detach, bound_detach, extra_data_detach]
            )

            self.pprint(loss_str=loss_str,
                        losses_list=losses_list,
                        call_back=call_back,
                        loss_indices=loss_indices,
                        loss_terms=loss_terms,
                        loss_weights=loss_weights)

            _current_loss = loss

            return _current_loss

        return closure


# Customized RMSE Loss for equation residuals in PyTorch dedicated to DeepONets
class OPIRMSELoss(LossBasics):
    def __init__(self, operator: DeepONet = None) -> None:
        super().__init__()

        self.split_dim = 1
        self.operator = operator
        self.loss_evaluator = None
        self.residual = None
        self.tol = 1e-25

        self.axis_loss_evaluator = lambda res: torch.mean(torch.square((res)), dim=1)

        self.min_causality_weight = self.tol
        self.mean_causality_weight = 0

        self.loss_states = {
            "pde": list(),
            "init": list(),
            "bound": list(),
        }

        self.loss_tags = list(self.loss_states.keys())
        self.hybrid_data_pinn = False

        self.losses_terms_indices = {
            "pde": 0,
            "init": 1,
            "bound": 2,
            "causality_weights": 3,
        }

    def _convert(
        self, input_data: Union[dict, np.ndarray] = None, device: str = None
    ) -> Union[dict, torch.Tensor]:
        if type(input_data) == dict:
            return {
                key: torch.from_numpy(item.astype(ARRAY_DTYPE)).to(device)
                for key, item in input_data.items()
            }

        else:
            return torch.from_numpy(input_data.astype(ARRAY_DTYPE)).to(device)

    def _to_tensor(self, *args, device="cpu"):
        return [self._convert(input_data=arg, device=device) for arg in args]

    def _data_loss(self, output_tilde=None, weights=None, target_data_tensor=None):
        output_split = torch.split(output_tilde, self.split_dim, dim=-1)
        target_split = torch.split(target_data_tensor, self.split_dim, dim=-1)

        data_losses = [
            weights[i]
            * self.loss_evaluator(out_split - tgt_split)
            / self.norm_evaluator(tgt_split)
            for i, (out_split, tgt_split) in enumerate(zip(output_split, target_split))
        ]

        return sum(data_losses)

    def _residual_loss(self, residual_approximation=None, weights=None):
        residual_loss = [
            weight * self.loss_evaluator(res)
            for weight, res in zip(weights, residual_approximation)
        ]

        return residual_loss

    def _no_boundary_penalisation(
        self, boundary_input: dict = None, residual: object = None
    ) -> torch.Tensor:
        return [torch.Tensor([0.0])]

    def _boundary_penalisation(
        self, boundary_input: dict = None, residual: SymbolicOperator = None
    ) -> torch.Tensor:
        return [
            residual.eval_expression(k, boundary_input[k])
            for k in boundary_input.keys()
        ]

    def _no_residual_wrapper(self, input_data: torch.Tensor = None) -> torch.Tensor:
        return self.residual(input_data)

    def _causality_preserving_residual_wrapper(
        self, input_data: torch.Tensor = None
    ) -> List:
        return self.causality_preserving(self.residual(input_data))

    @property
    def causality_weights_interval(self):
        return self.min_causality_weight, self.mean_causality_weight

    def _filter_necessary_loss_terms(self, residual: SymbolicOperator = None):
        tags = ["pde", "init"]
        indices = [0, 1]

        if residual.g_expressions:
            tags.append("bound")
            indices.append(2)
        else:
            pass

        if self.hybrid_data_pinn:
            tags.append("extra_data")
            indices.append(3)
        else:
            pass

        return tags, indices

    def _losses_states_str(self, tags: List[str] = None):
        losses_str = "\r"
        for item in tags:
            losses_str += f"{item}:{{}} "

        if self.causality_preserving != None:
            loss_str += "{{}}"

        return losses_str

    def __call__(
        self,
        input_data: Union[dict, torch.Tensor] = None,
        target_data: Union[dict, torch.Tensor] = None,
        call_back: str = "",
        residual: Callable = None,
        initial_input: Union[dict, torch.Tensor] = None,
        initial_state: Union[dict, torch.Tensor] = None,
        boundary_input: list = None,
        boundary_penalties: list = None,
        initial_penalty: float = 1,
        axis: int = -1,
        relative: bool = False,
        lambda_1: float = 0.0,
        lambda_2: float = 0.0,
        weights=None,
        weights_residual=None,
        device: str = "cpu",
        causality_preserving: Callable = None,
        use_mean: bool = True,
    ) -> Callable:
        self.residual = residual

        self.causality_preserving = causality_preserving

        if weights is None:
            weights = len(residual.output_names) * [1]

        if weights_residual is None:
            weights_residual = len(residual.output_names) * [1]

        loss_tags, loss_indices = self._filter_necessary_loss_terms(residual=residual)
        loss_str = self._losses_states_str(tags=loss_tags)

        if residual.g_expressions:
            boundary = self._boundary_penalisation
        else:
            boundary = self._no_boundary_penalisation
            boundary_penalties = [0]

        if weights is None:
            weights = len(residual.output_names) * [1]

        if self.causality_preserving:
            call_back = f", causality_weights: {self.causality_weights_interval}"
            self.residual_wrapper = self._causality_preserving_residual_wrapper

        else:
            self.residual_wrapper = self._no_residual_wrapper

        l1_reg_multiplication = self._exec_multiplication_in_regularization(
            lambda_type=type(lambda_1), term_type=type(self.operator.weights_l1)
        )

        l2_reg_multiplication = self._exec_multiplication_in_regularization(
            lambda_type=type(lambda_2), term_type=type(self.operator.weights_l2)
        )

        initial_input, initial_state = self._to_tensor(
            initial_input, initial_state, device=device
        )

        if use_mean == True:
            self.loss_evaluator = lambda res: torch.mean(torch.square((res)))
        else:
            self.loss_evaluator = lambda res: torch.sum(torch.square((res)))

        # Relative norm or not
        if relative == True:
            if use_mean == True:
                self.norm_evaluator = lambda ref: torch.mean(torch.square((ref)))
            else:
                self.norm_evaluator = lambda ref: torch.sum(torch.square((ref)))
        else:
            self.norm_evaluator = lambda ref: 1

        def closure():
            residual_approximation = self.residual_wrapper(input_data)

            boundary_approximation = boundary(
                boundary_input=boundary_input, residual=residual
            )

            initial_output_tilde = self.operator(**initial_input)

            residual_loss = self._residual_loss(
                residual_approximation=residual_approximation, weights=weights_residual
            )

            boundary_loss = self._residual_loss(
                residual_approximation=boundary_approximation,
                weights=boundary_penalties,
            )

            initial_data_loss = self._data_loss(
                output_tilde=initial_output_tilde,
                weights=weights,
                target_data_tensor=initial_state,
            )

            # L² and L¹ regularization term
            weights_l2 = self.operator.weights_l2
            weights_l1 = self.operator.weights_l1

            # beta *||W||_2 + alpha * ||W||_1
            l2_reg = l2_reg_multiplication(lambda_2, weights_l2)
            l1_reg = l1_reg_multiplication(lambda_1, weights_l1)

            # The complete loss function
            pde = sum(residual_loss)
            init = initial_data_loss
            bound = sum(boundary_loss)

            loss = pde + initial_penalty * init + bound + l2_reg + l1_reg

            # Back-propagation
            loss.backward()

            self.loss_states["pde"].append(float(loss.detach().data))
            self.loss_states["init"].append(float(init.detach().data))
            self.loss_states["bound"].append(float(bound.detach().data))

            pde_detach = float(pde.detach().data)
            init_detach = float(init.detach().data)
            bound_detach = float(bound.detach().data)

            losses_list = np.array([pde_detach, init_detach, bound_detach, call_back])

            sys.stdout.write((loss_str).format(*losses_list[loss_indices]))

            sys.stdout.flush()

        return closure


# Customized RMSE Loss for equation residuals in PyTorch dedicated to Koopman Autoencoders
class KAERMSELoss(LossBasics):
    def __init__(self, operator: AutoencoderKoopman = None) -> None:
        super().__init__()

        self.split_dim = 1
        self.operator = operator
        self.loss_evaluator = None
        self.indices = None
        self.batchers = dict()
        self.shifted_indices = dict()

    def _convert(
        self, input_data: Union[dict, np.ndarray] = None, device: str = None
    ) -> Union[dict, torch.Tensor]:
        if type(input_data) == dict:
            return {
                key: torch.from_numpy(item.astype(ARRAY_DTYPE)).to(device)
                for key, item in input_data.items()
            }

        else:
            return torch.from_numpy(input_data.astype(ARRAY_DTYPE)).to(device)

    def _to_tensor(self, *args, device="cpu"):
        return [self._convert(input_data=arg, device=device) for arg in args]

    def _data_loss(self, output_tilde=None, target_data_tensor=None):
        return self.loss_evaluator(
            output_tilde - target_data_tensor
        ) / self.norm_evaluator(target_data_tensor)

    def __call__(
        self,
        input_data: Union[dict, torch.Tensor] = None,
        target_data: Union[dict, torch.Tensor] = None,
        call_back: str = "",
        initial_penalty: float = 1,
        axis: int = -1,
        relative: bool = False,
        lambda_1: float = 0.0,
        lambda_2: float = 0.0,
        m: int = 1,
        S_p: float = None,
        T: float = 4,
        alpha_1: float = 1.0,
        alpha_2: float = 1.0,
        alpha_3: float = 1.0,
        device: str = "cpu",
        use_mean: bool = True,
    ) -> Callable:
        self.indices = [1, 2, m, m + 1]

        for n in self.indices:
            self.batchers[n] = IntersectingBatches(
                skip_size=1, batch_size=n, full=False
            )

            indices, shifted_indices = self.batchers[n].get_indices(
                dim=target_data.shape[0]
            )
            self.shifted_indices[n] = [indices, shifted_indices]

        l1_reg_multiplication = self._exec_multiplication_in_regularization(
            lambda_type=type(lambda_1), term_type=type(self.operator.weights_l1)
        )

        l2_reg_multiplication = self._exec_multiplication_in_regularization(
            lambda_type=type(lambda_2), term_type=type(self.operator.weights_l2)
        )

        shifted_target_data_1 = target_data[self.shifted_indices[1][1]]
        shifted_target_data_2 = target_data[self.shifted_indices[2][1]]
        shifted_target_data_m = target_data[self.shifted_indices[m][1]]
        shifted_target_data_m_1 = target_data[self.shifted_indices[m + 1][1]]

        if use_mean == True:
            self.loss_evaluator = lambda res: torch.mean(torch.square((res)))
        else:
            self.loss_evaluator = lambda res: torch.sum(torch.square((res)))

        # Relative norm or not
        if relative == True:
            if use_mean == True:
                self.norm_evaluator = lambda ref: torch.mean(torch.square((ref)))
            else:
                self.norm_evaluator = lambda ref: torch.sum(torch.square((ref)))
        else:
            self.norm_evaluator = lambda ref: 1

        def closure():
            output_tilde = self.operator.reconstruction_forward(**input_data)

            latent_space_ = self.operator.projection(**input_data)

            latent_space_1 = latent_space_[self.shifted_indices[1][0]]
            shifted_latent_space_1 = latent_space_[self.shifted_indices[1][1]]

            latent_space_2 = latent_space_[self.shifted_indices[2][0]]
            shifted_latent_space_2 = latent_space_[self.shifted_indices[2][1]]

            latent_space_m = latent_space_[self.shifted_indices[m][0]]
            shifted_latent_space_m = latent_space_[self.shifted_indices[m][1]]

            latent_space_m_1 = latent_space_[self.shifted_indices[m + 1][0]]
            shifted_latent_space_m_1 = latent_space_[self.shifted_indices[m + 1][1]]

            # Time-extrapolating in the latent space
            latent_space_1_tilde = self.operator.latent_forward_m(
                input_data=latent_space_1, m=1
            )
            latent_space_2_tilde = self.operator.latent_forward_m(
                input_data=latent_space_2, m=2
            )
            latent_space_m_tilde = self.operator.latent_forward_m(
                input_data=latent_space_m, m=m
            )
            latent_space_m_1_tilde = self.operator.latent_forward_m(
                input_data=latent_space_m_1, m=m + 1
            )

            # Reconstruction loss for multiple shifts
            output_tilde_1 = self.operator.reconstruction(
                input_data=latent_space_1_tilde
            )
            output_tilde_2 = self.operator.reconstruction(
                input_data=latent_space_2_tilde
            )
            output_tilde_m = self.operator.reconstruction(
                input_data=latent_space_m_tilde
            )
            output_tilde_m_1 = self.operator.reconstruction(
                input_data=latent_space_m_1_tilde
            )

            # Reconstruction loss
            loss_rec = self._data_loss(
                output_tilde=output_tilde, target_data_tensor=target_data
            )

            # Prediction losses in full space
            data_loss_1 = self._data_loss(
                output_tilde=output_tilde_1, target_data_tensor=shifted_target_data_1
            )
            data_loss_2 = self._data_loss(
                output_tilde=output_tilde_2, target_data_tensor=shifted_target_data_2
            )
            data_loss_m = self._data_loss(
                output_tilde=output_tilde_m, target_data_tensor=shifted_target_data_m
            )
            data_loss_m_1 = self._data_loss(
                output_tilde=output_tilde_m_1,
                target_data_tensor=shifted_target_data_m_1,
            )

            # Linearisation losses for the latent space
            linearisation_loss_1 = self._data_loss(
                output_tilde=latent_space_1_tilde,
                target_data_tensor=shifted_latent_space_1,
            )

            linearisation_loss_2 = self._data_loss(
                output_tilde=latent_space_2_tilde,
                target_data_tensor=shifted_latent_space_2,
            )

            linearisation_loss_m = self._data_loss(
                output_tilde=latent_space_m_tilde,
                target_data_tensor=shifted_latent_space_m,
            )

            linearisation_loss_m_1 = self._data_loss(
                output_tilde=latent_space_m_1_tilde,
                target_data_tensor=shifted_latent_space_m_1,
            )

            # L² and L¹ regularization term
            weights_l2 = self.operator.weights_l2
            weights_l1 = self.operator.weights_l1

            # beta *||W||_2 + alpha * ||W||_1
            l2_reg = l2_reg_multiplication(lambda_2, weights_l2)
            l1_reg = l1_reg_multiplication(lambda_1, weights_l1)

            # Loss =  alpha_1*(loss_rec + loss_pred) + alpha_2*(loss_lin) + beta *||W||_2 + alpha * ||W||_1
            loss_pred = data_loss_1 + data_loss_2 + data_loss_m + data_loss_m_1
            loss_lin = (
                linearisation_loss_1
                + linearisation_loss_2
                + linearisation_loss_m
                + linearisation_loss_m_1
            )

            loss = (
                alpha_1 * (loss_rec + loss_pred / T)
                + alpha_3 * loss_lin / T
                + l2_reg
                + l1_reg
            )

            # Back-propagation
            loss.backward()

            sys.stdout.write(("\rresidual loss: {}").format(loss, call_back))
            sys.stdout.flush()

        return closure


# Customized RMSE Loss for equation residuals in PyTorch dedicated to Koopman Autoencoders
class VAERMSELoss(LossBasics):
    def __init__(self, operator: AutoencoderVariational = None) -> None:
        super().__init__()

        self.split_dim = 1
        self.operator = operator
        self.loss_evaluator = None
        self.beta = 1
        self.loss_states = {"loss": list(), "kl_loss": list()}

    def _convert(
        self, input_data: Union[dict, np.ndarray] = None, device: str = None
    ) -> Union[dict, torch.Tensor]:
        if type(input_data) == dict:
            return {
                key: torch.from_numpy(item.astype(ARRAY_DTYPE)).to(device)
                for key, item in input_data.items()
            }

        else:
            return torch.from_numpy(input_data.astype(ARRAY_DTYPE)).to(device)

    def _to_tensor(self, *args, device="cpu"):
        return [self._convert(input_data=arg, device=device) for arg in args]

    def _data_loss(self, output_tilde=None, target_data_tensor=None):
        return self.loss_evaluator(
            output_tilde - target_data_tensor
        ) / self.norm_evaluator(target_data_tensor)

    def _kl_loss(self):
        z_mean, z_log_var = self.operator.mu, self.operator.log_v
        kl_loss = -(self.beta / 2) * torch.mean(
            1.0 + z_log_var - z_mean**2.0 - torch.exp(z_log_var)
        )

        return kl_loss

    def __call__(
        self,
        input_data: Union[dict, torch.Tensor] = None,
        target_data: Union[dict, torch.Tensor] = None,
        call_back: str = "",
        relative: bool = False,
        lambda_1: float = 0.0,
        lambda_2: float = 0.0,
        device: str = "cpu",
        use_mean: bool = True,
        beta: float = 1,
    ) -> Callable:

        """

        Parameters
        ----------
        input_data : Union[dict, torch.Tensor]
            The data used as input for the model.
        target_data : Union[dict, torch.Tensor]
            The target data used to guide the model training.
        call_back : str
            A string to be used for printing log during the training.
        relative : bool
            Use relative norm (dividing by a norm of the target data) or not.
        lambda_1 : float
            Penalty for the L^1 norm of the weights (regularization term).
        lambda_2 : float
            Penalty for the L^2 norm of the weights (regularization term).
        device : str
            Device to be used for executing the method (`cpu` or `gpu`)
        use_mean : bool
            Use a mean operation or not. In negative case, a sum is used.
        beta : float
            Penalty for the Kulback-Leibler term. 
        Returns
        -------
            A callable object used to evaluate the global loss function.
        """

        l1_reg_multiplication = self._exec_multiplication_in_regularization(
            lambda_type=type(lambda_1), term_type=type(self.operator.weights_l1)
        )

        l2_reg_multiplication = self._exec_multiplication_in_regularization(
            lambda_type=type(lambda_2), term_type=type(self.operator.weights_l2)
        )

        self.beta = beta

        if use_mean == True:
            self.loss_evaluator = lambda res: torch.mean(torch.square((res)))
        else:
            self.loss_evaluator = lambda res: torch.sum(torch.square((res)))

        # Relative norm or not
        if relative == True:
            if use_mean == True:
                self.norm_evaluator = lambda ref: torch.mean(torch.square((ref)))
            else:
                self.norm_evaluator = lambda ref: torch.sum(torch.square((ref)))
        else:
            self.norm_evaluator = lambda ref: 1

        def closure():
            output_tilde = self.operator.reconstruction_forward(**input_data)

            data_loss = self._data_loss(
                output_tilde=output_tilde, target_data_tensor=target_data
            )

            # L² and L¹ regularization term
            weights_l2 = self.operator.weights_l2
            weights_l1 = self.operator.weights_l1

            # beta *||W||_2 + alpha * ||W||_1
            l2_reg = l2_reg_multiplication(lambda_2, weights_l2)
            l1_reg = l1_reg_multiplication(lambda_1, weights_l1)

            # Loss = ||residual||_2 + lambda_2 * ||W||_1
            kl_loss = self._kl_loss()
            loss = data_loss + kl_loss + l2_reg + l1_reg

            # Back-propagation
            loss.backward()

            loss_str = ("\rresidual loss: {}, kl_loss:{}").format(
                loss, kl_loss, call_back
            )

            self.loss_states["loss"].append(float(loss.detach().data))
            self.loss_states["kl_loss"].append(float(kl_loss.detach().data))

            sys.stdout.write(loss_str)
            sys.stdout.flush()

        return closure

# Wrapper for the Binary Cross-entropy loss function
class BCELoss(LossBasics):
    def __init__(self, operator: torch.nn.Module) -> None:

        super().__init__()

        self.loss_states = {"loss":list()}
        self.operator = operator

    def __call__(self,
                 input_data: Union[dict, torch.Tensor] = None,
                 target_data: Union[dict, torch.Tensor] = None,
                 call_back: str = "",
                 **kwargs) -> None:
        """

        Parameters
        ----------
        input_data : Union[dict, torch.Tensor]
            The data used as input for the model.
        target_data : Union[dict, torch.Tensor]
            The target data used to guide the model training.
        call_back : str
            A string to be used for printing log during the training.
        Returns
        -------
            A callable object used to evaluate the global loss function.
        """


        def closure():

            output_tilde = self.operator.forward(**input_data)
            loss = F.binary_cross_entropy(output_tilde, target_data)

            loss.backward()

            loss_str = ("\rloss: {}").format(loss)

            self.loss_states["loss"].append(float(loss.detach().data))

            sys.stdout.write(loss_str)
            sys.stdout.flush()

        return closure

