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
from typing import Callable, List, Tuple

import sympy
from sympy.parsing.sympy_parser import parse_expr


class FromSymbol2FLambda:
    def __init__(self, engine: str = "numpy", variables: List[str] = None) -> None:
        """
        Initialize a lambda function from a string.

        Parameters
        ----------
        engine : str, optional
            The low level engine used, e. g. numpy, torch ... The default value is 'numpy'.
        variables : list of str, optional
            The list of definition variables. The default value is None.

        Returns
        -------
        None
        """

        self.engine = engine
        self.aux_engine = "simulai.math.products"

        self.engine_module = importlib.import_module(self.engine)
        self.aux_engine_module = importlib.import_module(self.aux_engine)
        self.tokens_module = importlib.import_module("simulai.tokens")

        self.variables = [sympy.Symbol(vv) for vv in variables]

        self.func_sep = "("

    def _handle_composite_function(
        self, func_expr: str = None
    ) -> Tuple[List[str], bool]:
        """
        Handle composite functions such as g(x) = f_1 o f_2 o ... o f_n(x) = f_1(f_2( ... f_n(x) ... )).

        Parameters
        ----------
        func_expr : str, optional
            String containing the definition expression of a function.

        Returns
        -------
        functions : list of str
            List of functions names.
        success : bool
            Boolean indicating success.

        """

        splits = func_expr.split(self.func_sep)
        functions = [ss.capitalize() for ss in splits[:-1]]

        return functions, True

    from typing import Tuple

    def _get_function_name(self, func_expr: str = None) -> Tuple[str, bool]:
        """
        Get the input name of a function and return its corresponding standard name.

        Parameters
        ----------
        func_expr : str, optional
            Function name provided as input.

        Returns
        -------
        function_name : str
            Corresponding function name.
        is_composite : bool
            Boolean indicating whether the function is composite or not.

        Raises
        ------
        Exception
            If the expression is not valid.
        """
        splits = func_expr.split(self.func_sep)

        if len(splits) == 2:
            return [splits[0].capitalize()], True
        elif len(splits) == 1:
            return splits[0], False
        elif len(splits) > 2:
            return self._handle_composite_function(func_expr=func_expr)
        else:
            raise Exception(f"The expression {func_expr} is not valid.")

    def clean_engines(self) -> None:
        """
        Clean all the pre-defined engines.

        Returns
        -------
        None
            This function does not return anything.
        """
        self.engine_module = None
        self.aux_engine_module = None
        self.tokens_module = None

    def convert(self, expression: str = None) -> Callable:
        """
        Receive a string mathematical expression and convert it into a callable function.

        Parameters
        ----------
        expression : str, optional
            String containing the mathematical expression definition.

        Returns
        -------
        callable
            Callable function equivalent to the string expression.

        """
        expression_names, is_function = self._get_function_name(func_expr=expression)
        symbol_expression = parse_expr(expression, evaluate=0)

        if is_function is True:
            symbol_functions = [
                getattr(self.tokens_module, expression_name, None)
                for expression_name in expression_names
            ]

            assert all([ss != None for ss in symbol_functions]), (
                f"The list of functions {expression_names}"
                f" does not exist in {self.tokens_module} completely."
            )

            op_map = dict()
            for expression_name in expression_names:
                try:
                    engine_function = getattr(
                        self.engine_module, expression_name.lower(), None
                    )
                    assert engine_function is not None
                except:
                    engine_function = getattr(
                        self.aux_engine_module, expression_name.lower(), None
                    )
                    assert engine_function is not None

                op_map[expression_name] = engine_function

            compiled_expr = sympy.lambdify(
                self.variables, symbol_expression, modules=[op_map, self.engine]
            )

        else:
            compiled_expr = sympy.lambdify(self.variables, symbol_expression)

        return compiled_expr
