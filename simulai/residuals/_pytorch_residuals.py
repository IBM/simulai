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

import importlib
from typing import List, Union

import numpy as np
import sympy
import torch
from sympy import sympify
from sympy.parsing.sympy_parser import parse_expr
from torch.autograd import grad
from torch.autograd.functional import jacobian

from simulai.io import MakeTensor
from simulai.tokens import D


class SymbolicOperator(torch.nn.Module):
    """
    The SymbolicOperatorClass is a class that constructs tensor operators using symbolic expressions written in PyTorch.

    Args:
        expressions (list, optional): List of expressions representing the operator. Defaults to None.
        input_vars (list, optional): List of input variables. Defaults to None.
        output_vars (list, optional): List of output variables. Defaults to None.
        function (callable, optional): A callable that represents the evaluation of the tensor operator. Defaults to None.
        gradient (callable, optional): A callable that represents the gradient evaluation of the tensor operator. Defaults to None.
        keys (str, optional): The keys for accessing the inputs in the input dictionary. Defaults to None.
        inputs_key (str, optional): The key for accessing the input data in the input dictionary. Defaults to None.
        processing (str, optional): The processing method to be used during the evaluation of the tensor operator. Defaults to 'serial'.
        device (str, optional): The device to be used during the evaluation of the tensor operator. Can be either 'cpu' or 'gpu'. Defaults to 'cpu'.
        auxiliary_expressions (list, optional): Additional expressions that are needed to evaluate the tensor operator. Defaults to None.
        constants (dict, optional): Constants that are required during the evaluation of the tensor operator. Defaults to None.

    Returns:
        object: An instance of the SymbolicOperatorClass.
    """

    def __init__(
        self,
        expressions: List[Union[sympy.Expr, str]] = None,
        input_vars: List[Union[sympy.Symbol, str]] = None,
        output_vars: List[Union[sympy.Symbol, str]] = None,
        function: callable = None,
        gradient: callable = None,
        keys: str = None,
        inputs_key=None,
        constants: dict = None,
        processing: str = "serial",
        device: str = "cpu",
        engine: str = "torch",
        auxiliary_expressions: list = None,
    ) -> None:
        if engine == "torch":
            super(SymbolicOperator, self).__init__()
        else:
            pass

        self.engine = importlib.import_module(engine)

        self.constants = constants
        self.processing = processing
        self.periodic_bc_protected_key = "periodic"

        self.protected_funcs = ["cos", "sin", "sqrt"]
        self.protected_operators = ["L", "Div", "Identity", "Kronecker"]

        self.protected_funcs_subs = self._construct_protected_functions()
        self.protected_operators_subs = self._construct_implict_operators()

        # Configuring the device to be used during the fitting process
        if device == "gpu":
            if not torch.cuda.is_available():
                print("Warning: There is no GPU available, using CPU instead.")
                device = "cpu"
            else:
                device = "cuda"
                print("Using GPU.")
        elif device == "cpu":
            print("Using CPU.")
        else:
            raise Exception(f"The device must be cpu or gpu, but received: {device}")

        self.device = device

        self.expressions = [self._parse_expression(expr=expr) for expr in expressions]

        if isinstance(auxiliary_expressions, dict):
            self.auxiliary_expressions = {
                key: self._parse_expression(expr=expr)
                for key, expr in auxiliary_expressions.items()
            }
        else:
            self.auxiliary_expressions = auxiliary_expressions

        self.input_vars = [self._parse_variable(var=var) for var in input_vars]
        self.output_vars = [self._parse_variable(var=var) for var in output_vars]

        self.input_names = [var.name for var in self.input_vars]
        self.output_names = [var.name for var in self.output_vars]
        self.keys = keys
        self.inputs_key = inputs_key
        self.all_vars = self.input_vars + self.output_vars

        if self.inputs_key is not None:
            self.forward = self._forward_dict
        else:
            self.forward = self._forward_tensor

        self.function = function
        self.diff_symbol = D

        self.output = None

        self.f_expressions = list()
        self.g_expressions = dict()

        self.feed_vars = None

        for name in self.output_names:
            setattr(self, name, None)

        # Defining functions for returning each variable of the regression
        # function
        for index, name in enumerate(self.output_names):
            setattr(
                self,
                name,
                lambda data: self.function.forward(input_data=data)[..., index][
                    ..., None
                ],
            )

        # If no external gradient is provided, use the core gradient evaluator
        if gradient is None:
            gradient_function = self.gradient
        else:
            gradient_function = gradient

        subs = {self.diff_symbol.name: gradient_function}
        subs.update(self.protected_funcs_subs)

        for expr in self.expressions:
            f_expr = sympy.lambdify(self.all_vars, expr, subs)

            self.f_expressions.append(f_expr)

        if self.auxiliary_expressions is not None:
            for key, expr in self.auxiliary_expressions.items():
                g_expr = sympy.lambdify(self.all_vars, expr, subs)

                self.g_expressions[key] = g_expr

        # Method for executing the expressions evaluation
        if self.processing == "serial":
            self.process_expression = self._process_expression_serial
        else:
            raise Exception(f"Processing case {self.processing} not supported.")

    def _construct_protected_functions(self):
        """
        This function creates a dictionary of protected functions from the engine object attribute.

        Args:
            self (object): The instance of the class that calls this function.

        Returns:
            dict: A dictionary of function names and their corresponding function objects.
        """
        protected_funcs = {
            func: getattr(self.engine, func) for func in self.protected_funcs
        }

        return protected_funcs

    def _construct_implict_operators(self):
        """
        This function creates a dictionary of protected operators from the operators engine module.

        Args:
            self (object): The instance of the class that calls this function.

        Returns:
            dict: A dictionary of operator names and their corresponding function objects.
        """
        operators_engine = importlib.import_module("simulai.tokens")

        protected_operators = {
            func: getattr(operators_engine, func) for func in self.protected_operators
        }

        return protected_operators

    def _parse_expression(self, expr=Union[sympy.Expr, str]) -> sympy.Expr:
        """
        Parses the input expression and returns a SymPy expression.

        Parameters
        ----------
        expr : Union[sympy.Expr, str], optional
            The expression to parse, by default None. It can either be a SymPy expression or a string.

        Returns
        -------
        sympy.Expr
            The parsed SymPy expression.

        Raises
        ------
        Exception
            If the `constants` attribute is not defined, and the input expression is a string.
        """
        if isinstance(expr, str):
            try:
                expr_ = sympify(
                    expr, locals=self.protected_operators_subs, evaluate=False
                )

                if self.constants is not None:
                    expr_ = expr_.subs(self.constants)

            except ValueError:
                if self.constants is not None:
                    _expr = expr
                    for key, value in self.constants.items():
                        _expr = _expr.replace(key, str(value))

                    expr_ = parse_expr(_expr, evaluate=0)
                else:
                    raise Exception("It is necessary to define a constants dict.")
        else:
            expr_ = expr

        return expr_

    def _parse_variable(self, var=Union[sympy.Symbol, str]) -> sympy.Symbol:
        """
        Parse the input variable and return a SymPy Symbol.

        Parameters
        ----------
        var : Union[sympy.Symbol, str], optional
            The input variable, either a SymPy Symbol or a string.

        Returns
        -------
        sympy.Symbol
            A SymPy Symbol representing the input variable.
        """
        if isinstance(var, str):
            return sympy.Symbol(var)
        else:
            return var

    def _forward_tensor(self, input_data: torch.Tensor = None) -> torch.Tensor:
        """
        Forward the input tensor through the function.

        Parameters
        ----------
        input_data : torch.Tensor, optional
            The input tensor.

        Returns
        -------
        torch.Tensor
            The output tensor after forward pass.
        """
        return self.function.forward(input_data=input_data)

    def _forward_dict(self, input_data: dict = None) -> torch.Tensor:
        """
        Forward the input dictionary through the function.

        Parameters
        ----------
        input_data : dict, optional
            The input dictionary.

        Returns
        -------
        torch.Tensor
            The output tensor after forward pass.
        """
        return self.function.forward(**input_data)

    def _process_expression_serial(self, feed_vars: dict = None) -> List[torch.Tensor]:
        """
        Process the expression list serially using the given feed variables.

        Parameters
        ----------
        feed_vars : dict, optional
            The feed variables.

        Returns
        -------
        List[torch.Tensor]
            A list of tensors after evaluating the expressions serially.
        """
        return [f(**feed_vars).to(self.device) for f in self.f_expressions]

    def _process_expression_individual(
        self, index: int = None, feed_vars: dict = None
    ) -> torch.Tensor:
        """
        Evaluates a single expression specified by index from the f_expressions list with given feed variables.

        Parameters
        ----------
        index : int, optional
            Index of the expression to be evaluated, by default None
        feed_vars : dict, optional
            Dictionary of feed variables, by default None

        Returns
        -------
        torch.Tensor
            Result of evaluating the specified expression with given feed variables
        """
        return self.f_expressions[index](**feed_vars).to(self.device)

    def __call__(
        self, inputs_data: Union[np.ndarray, dict] = None
    ) -> List[torch.Tensor]:
        """
        Evaluate the symbolic expression.

        This function takes either a numpy array or a dictionary of numpy arrays as input.

        Parameters:
        inputs_data (Union[np.ndarray, dict]): Input data to evaluate the expression. If inputs_data is a numpy array,
        it must have the same length as the input_names attribute. If inputs_data is a dictionary, it must have
        key-value pairs where the keys correspond to the names in the input_names attribute.

        Returns:
        List[torch.Tensor]: A list of tensors containing the evaluated expressions.

        Raises:
        Exception: If inputs_data is not a numpy array or a dictionary, or if the inputs_data is a dictionary and the key
        does not match with the inputs_key attribute.
        """
        constructor = MakeTensor(
            input_names=self.input_names, output_names=self.output_names
        )

        inputs_list = constructor(input_data=inputs_data, device=self.device)

        output = self.forward(input_data=inputs_list)

        output = output.to(self.device)  # TODO Check if it is necessary

        outputs_list = torch.split(output, 1, dim=-1)

        outputs = {key: value for key, value in zip(self.output_names, outputs_list)}

        if type(inputs_list) is list:
            inputs = {key: value for key, value in zip(self.input_names, inputs_list)}

        elif type(inputs_list) is dict:
            inputs_list = [
                inputs_list[self.inputs_key]
            ]  # TODO It it not generic the enough

            assert (
                self.inputs_key is not None
            ), "If inputs_list is dict, \
                                                 it is necessary to provide\
                                                 a key."
            inputs = {key: value for key, value in zip(self.input_names, inputs_list)}
        else:
            raise Exception(
                f"Format {type(inputs_list)} not supported \
                            for inputs_list"
            )

        feed_vars = {**outputs, **inputs}

        # It returns a list of tensors containing the expressions
        # evaluated over a domain
        return self.process_expression(feed_vars=feed_vars)

    def eval_expression(self, key, inputs_list):
        """
        This function evaluates an expression stored in the class attribute 'g_expressions' using the inputs in 'inputs_list'. If the expression has a periodic boundary condition, the function evaluates the expression at the lower and upper boundaries and returns the difference. If the inputs are provided as a list, they are split into individual tensors and stored in a dictionary with the keys as the input names. If the inputs are provided as an np.ndarray, they are converted to tensors and split along the second axis. If the inputs are provided as a dict, they are extracted using the 'inputs_key' attribute. The inputs, along with the outputs obtained from running the function, are then passed as arguments to the expression using the 'g(**feed_vars)' syntax.

        Parameters
        ----------
        key : str
            the key used to retrieve the expression from the 'g_expressions' attribute
        inputs_list: list
            either a list of arrays, an np.ndarray, or a dict containing the inputs to the function

        Raises:
        -------
        Exception: if the expression does not exist in 'g_expressions'
        Exception: if the input format is not supported
        Exception: if the inputs_list is a dict but the 'inputs_key' attribute is not provided

        Returns
        -------
        result (float/torch tensor): the result of evaluating the expression using the inputs.
        """

        try:
            g = self.g_expressions.get(key)
        except:
            raise Exception(f"The expression {key} does not exist.")

        # Periodic boundary conditions
        if self.periodic_bc_protected_key in key:
            assert isinstance(inputs_list, list), (
                "When a periodic boundary expression is used,"
                " the input must be a list of arrays."
            )

            # Lower bound
            constructor = MakeTensor(
                input_names=self.input_names, output_names=self.output_names
            )

            tensors_list = constructor(input_data=inputs_list[0], device=self.device)

            inputs_L = {
                key: value for key, value in zip(self.input_names, tensors_list)
            }

            output = self.function.forward(input_data=tensors_list)

            output = output.to(self.device)  # TODO Check if it is necessary

            outputs_list = torch.split(output, 1, dim=-1)

            outputs_L = {
                key: value for key, value in zip(self.output_names, outputs_list)
            }

            feed_vars_L = {**inputs_L, **outputs_L}

            # Upper bound
            constructor = MakeTensor(
                input_names=self.input_names, output_names=self.output_names
            )

            tensors_list = constructor(input_data=inputs_list[-1], device=self.device)

            inputs_U = {
                key: value for key, value in zip(self.input_names, tensors_list)
            }

            output = self.function.forward(input_data=tensors_list)

            output = output.to(self.device)  # TODO Check if it is necessary

            outputs_list = torch.split(output, 1, dim=-1)

            outputs_U = {
                key: value for key, value in zip(self.output_names, outputs_list)
            }

            feed_vars_U = {**inputs_U, **outputs_U}

            # Evaluating the boundaries equality
            return g(**feed_vars_L) - g(**feed_vars_U)

        # The non-periodic cases
        else:
            output = self.function.forward(input_data=inputs_list)

            outputs_list = torch.split(output, 1, dim=-1)

            outputs = {
                key: value for key, value in zip(self.output_names, outputs_list)
            }

            if type(inputs_list) is list:
                inputs = {
                    key: value for key, value in zip(self.input_names, inputs_list)
                }

            elif type(inputs_list) is np.ndarray:
                arrays_list = np.split(inputs_list, inputs_list.shape[1], axis=1)
                tensors_list = [torch.from_numpy(arr) for arr in arrays_list]

                for t in tensors_list:
                    t.requires_grad = True

                inputs = {
                    key: value for key, value in zip(self.input_names, tensors_list)
                }

            elif type(inputs_list) is dict:
                assert (
                    self.inputs_key is not None
                ), "If inputs_list is dict, \
                                                     it is necessary to provide\
                                                     a key."

                inputs = {
                    key: value
                    for key, value in zip(
                        self.input_names, inputs_list[self.inputs_key]
                    )
                }

            else:
                raise Exception(
                    f"Format {type(inputs_list)} not supported \
                                for inputs_list"
                )

            feed_vars = {**inputs, **outputs}

            return g(**feed_vars)

    @staticmethod
    def gradient(feature, param):
        """
        Calculates the gradient of the given feature with respect to the given parameter.

        Parameters
        ----------
        feature : torch.Tensor
            Tensor with the input feature.
        param : torch.Tensor
            Tensor with the parameter to calculate the gradient with respect to.

        Returns
        -------
        grad_ : torch.Tensor
            Tensor with the gradient of the feature with respect to the given parameter.

        Examples
        --------
        >>> feature = torch.tensor([1, 2, 3], dtype=torch.float32)
        >>> param = torch.tensor([2, 3, 4], dtype=torch.float32)
        >>> gradient(feature, param)
        tensor([1., 1., 1.], grad_fn=<AddBackward0>)
        """
        grad_ = grad(
            feature,
            param,
            grad_outputs=torch.ones_like(feature),
            create_graph=True,
            allow_unused=True,
            retain_graph=True,
        )

        return grad_[0]

    def jac(self, inputs):
        """
        Calculates the Jacobian of the forward function of the model with respect to its inputs.

        Parameters
        ----------
        inputs : torch.Tensor
            Tensor with the input data to the forward function.

        Returns
        -------
        jac_ : torch.Tensor
            Tensor with the Jacobian of the forward function with respect to its inputs.

        Examples
        --------
        >>> inputs = torch.tensor([[1, 2, 3], [2, 3, 4]], dtype=torch.float32)
        >>> jac(inputs)
        tensor([[1., 1., 1.],
                [1., 1., 1.]], grad_fn=<MulBackward0>)
        """

        def inner(inputs):
            return self.forward(input_data=inputs)

        return jacobian(inner, inputs)
