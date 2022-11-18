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

import fsspec.asyn
import torch
import numpy as np
import sympy
from sympy.parsing.sympy_parser import parse_expr
from torch.autograd import grad
from torch.autograd.functional import jacobian
import importlib
from typing import List, Union

from simulai.tokens import D
from simulai.io import MakeTensor

# Factory function for constructing the symbolic operator accordingly
# the numerical engine used
def SymbolicOperator(expressions: list = None, input_vars: list = None,
                     output_vars: list = None, function: callable = None, gradient: callable = None,
                     keys: str = None, inputs_key: str = None, processing:str='serial',
                     device:str='cpu', engine: str = 'torch',
                     auxiliary_expressions: list = None, constants:dict=None) -> object:


    # It constructs tensor operators using symbolic expressions
    class SymbolicOperatorClass(torch.nn.Module if engine == 'torch'
                                else object):

        def __init__(self, expressions:List[Union[sympy.Expr, str]]=None,
                           input_vars:List[Union[sympy.Symbol, str]]=None,
                           output_vars:List[Union[sympy.Symbol, str]]=None,
                           function:callable=None,
                           gradient:callable=None,
                           keys:str=None,
                           inputs_key=None,
                           constants:dict=None,
                           processing: str = 'serial',
                           device:str='cpu',
                           engine:str='torch') -> None:

            if engine == 'torch':
                super(SymbolicOperatorClass, self).__init__()
            else:
                pass

            self.constants = constants
            self.processing = processing
            self.periodic_bc_protected_key = 'periodic'

            # Configuring the device to be used during the fitting process
            if device == 'gpu':
                if not torch.cuda.is_available():
                    print("Warning: There is no GPU available, using CPU instead.")
                    device = 'cpu'
                else:
                    device = "cuda"
                    print("Using GPU.")
            elif device == 'cpu':
                print("Using CPU.")
            else:
                raise Exception(f"The device must be cpu or gpu, but received: {device}")

            self.device = device

            self.engine = importlib.import_module(engine)

            self.expressions = [self._parse_expression(expr=expr)  for expr in expressions]

            if isinstance(auxiliary_expressions, dict):
                self.auxiliary_expressions = {key: self._parse_expression(expr=expr)
                                              for key, expr in auxiliary_expressions.items()}
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

            self.protected_funcs = ['cos', 'sin', 'sqrt']

            self.protected_funcs_subs = {func:getattr(self.engine, func) for func in self.protected_funcs}

            for name in self.output_names:
                setattr(self, name, None)

            # Defining functions for returning each variable of the regression
            # function
            for index, name in enumerate(self.output_names):
                setattr(self, name, lambda data: self.function.forward(input_data=data)[..., index][..., None])

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
            if self.processing == 'serial':
                self.process_expression = self._process_expression_serial
            else:
                raise Exception(f"Processing case {self.processing} not supported.")

        def _parse_expression(self, expr=Union[sympy.Expr, str]) -> sympy.Expr:

            if isinstance(expr, str):
                try:
                    expr_ = parse_expr(expr, evaluate=False)

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

            if isinstance(var, str):
                return sympy.Symbol(var)
            else:
                return var

        def _forward_tensor(self, input_data:torch.Tensor=None) -> torch.Tensor:

            return self.function.forward(input_data=input_data)

        def _forward_dict(self, input_data:dict=None) -> torch.Tensor:

            return self.function.forward(**input_data)

        def _process_expression_serial(self, feed_vars:dict=None) -> List[torch.Tensor]:

            return [f(**feed_vars).to(self.device) for f in self.f_expressions]

        def _process_expression_individual(self, index:int=None, feed_vars:dict=None) -> torch.Tensor:

            return self.f_expressions[index](**feed_vars).to(self.device)

        # Evaluating the symbolic expression
        def __call__(self, inputs_data:Union[np.ndarray, dict]=None) -> List[torch.Tensor]:

            constructor = MakeTensor(input_names=self.input_names,
                                     output_names=self.output_names)

            inputs_list = constructor(input_data=inputs_data, device=self.device)

            output = self.forward(input_data=inputs_list)

            output = output.to(self.device) # TODO Check if it is necessary

            outputs_list = torch.split(output, 1, dim=-1)

            outputs = {key: value for key, value in
                       zip(self.output_names, outputs_list)}

            if type(inputs_list) is list:
                inputs = {key: value for key, value in
                          zip(self.input_names, inputs_list)}

            elif type(inputs_list) is dict:

                inputs_list = [inputs_list[self.inputs_key]] # TODO It it not generic the enough

                assert self.inputs_key is not None, "If inputs_list is dict, \
                                                     it is necessary to provide\
                                                     a key."
                inputs = {key: value for key, value in
                          zip(self.input_names, inputs_list)}
            else:
                raise Exception(f"Format {type(inputs_list)} not supported \
                                for inputs_list")

            feed_vars = {**outputs, **inputs}

            # It returns a list of tensors containing the expressions
            # evaluated over a domain
            return self.process_expression(feed_vars=feed_vars)

        def eval_expression(self, key, inputs_list):

            try:
                g = self.g_expressions.get(key)
            except:
                raise Exception(f"The expression {key} does not exist.")

            # Periodic boundary conditions
            if self.periodic_bc_protected_key in key:

                assert isinstance(inputs_list, list), "When a periodic boundary expression is used," \
                                                      " the input must be a list of arrays."

                # Lower bound
                constructor = MakeTensor(input_names=self.input_names,
                                         output_names=self.output_names)

                tensors_list = constructor(input_data=inputs_list[0], device=self.device)

                inputs_L = {key: value for key, value in
                          zip(self.input_names, tensors_list)}

                output = self.function.forward(input_data=tensors_list)

                output = output.to(self.device)  # TODO Check if it is necessary

                outputs_list = torch.split(output, 1, dim=-1)

                outputs_L = {key: value for key, value in
                             zip(self.output_names, outputs_list)}

                feed_vars_L = {**inputs_L, **outputs_L}

                # Upper bound
                constructor = MakeTensor(input_names=self.input_names,
                                         output_names=self.output_names)

                tensors_list = constructor(input_data=inputs_list[-1], device=self.device)

                inputs_U = {key: value for key, value in
                            zip(self.input_names, tensors_list)}

                output = self.function.forward(input_data=tensors_list)

                output = output.to(self.device)  # TODO Check if it is necessary

                outputs_list = torch.split(output, 1, dim=-1)

                outputs_U = {key: value for key, value in
                             zip(self.output_names, outputs_list)}

                feed_vars_U = {**inputs_U, **outputs_U}

                # Evaluating the boundaries equality
                return g(**feed_vars_L) - g(**feed_vars_U)

            # The non-periodic cases
            else:

                output = self.function.forward(input_data=inputs_list)

                outputs_list = torch.split(output, 1, dim=-1)

                outputs = {key: value for key, value in
                           zip(self.output_names, outputs_list)}

                if type(inputs_list) is list:
                    inputs = {key: value for key, value in
                              zip(self.input_names, inputs_list)}

                elif type(inputs_list) is np.ndarray:

                    arrays_list = np.split(inputs_list, inputs_list.shape[1], axis=1)
                    tensors_list = [torch.from_numpy(arr) for arr in arrays_list]

                    for t in tensors_list:
                        t.requires_grad = True

                    inputs = {key: value for key, value in
                              zip(self.input_names, tensors_list)}

                elif type(inputs_list) is dict:
                    assert self.inputs_key is not None, "If inputs_list is dict, \
                                                         it is necessary to provide\
                                                         a key."

                    inputs = {key: value for key, value in
                              zip(self.input_names, inputs_list[self.inputs_key])}

                else:
                    raise Exception(f"Format {type(inputs_list)} not supported \
                                    for inputs_list")

                feed_vars = {**inputs, **outputs}

                return g(**feed_vars)

        @staticmethod
        def gradient(feature, param):

            grad_ = grad(feature, param, grad_outputs=torch.ones_like(feature),
                         create_graph=True, allow_unused=True,
                         retain_graph=True)

            return grad_[0]

        def jac(self, inputs):

            def inner(inputs):
                return self.forward(input_data=inputs)

            return jacobian(inner, inputs)

    return SymbolicOperatorClass(expressions=expressions,
                                 input_vars=input_vars,
                                 output_vars=output_vars,
                                 function=function,
                                 gradient=gradient,
                                 keys=keys,
                                 inputs_key=inputs_key,
                                 constants=constants,
                                 processing=processing,
                                 device=device,
                                 engine=engine)
