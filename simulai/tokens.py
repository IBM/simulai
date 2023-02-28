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

import sympy

# Token used for representing the operator differentiation
# It must contain two arguments such df/dy = D(f, y)
D = sympy.Function("D")

Sin = sympy.Function("Sin")
Cos = sympy.Function("Cos")
Tanh = sympy.Function("Tanh")
Identity = sympy.Function("Identity")
Kronecker = sympy.Function("Kronecker")


def L(u: sympy.Symbol, vars: tuple) -> callable:
    """
    Generate a callable object to compute the Laplacian operator.

    The Laplacian operator is a second-order differential operator commonly used in
    fields such as physics and engineering. It is defined as the sum of second
    partial derivatives of a function with respect to each variable.

    Parameters
    ----------
    u : sympy.Symbol
        The function to compute the Laplacian of.
    vars : tuple
        A tuple of variables to compute the Laplacian with respect to.

    Returns
    -------
    callable
        A callable object that computes the Laplacian of a function with respect to
        the given variables.

    Examples
    --------
    >>> x, y, z = sympy.symbols('x y z')
    >>> u = x*y*z
    >>> L(u, (x, y, z))
    x*y + x*z + y*z
    """
    l = sum([D(D(u, var), var) for var in vars])

    return l


def Div(u: sympy.Symbol, vars: tuple) -> callable:
    """
    Generate a callable object to compute the divergence operator.

    The divergence operator is a first-order differential operator that measures the
    magnitude and direction of a flow of a vector field from its source and
    convergence to a point.

    Parameters
    ----------
    u : sympy.Symbol
        The vector field to compute the divergence of.
    vars : tuple
        A tuple of variables to compute the divergence with respect to.

    Returns
    -------
    callable
        A callable object that computes the divergence of a vector field with respect
        to the given variables.

    Examples
    --------
    >>> x, y, z = sympy.symbols('x y z')
    >>> u = sympy.Matrix([x**2, y**2, z**2])
    >>> Div(u, (x, y, z))
    2*x + 2*y + 2*z
    """
    l = sum([D(u, var) for var in vars])

    return l


def diff_op(func: callable) -> callable:
    """
    Decorate a function to indicate that it is an operator.

    This decorator is used to tag a function that represents an operator, such as the
    Laplacian or divergence operators, and makes it possible to recognize it as such.

    Parameters
    ----------
    func : callable
        The function to decorate.

    Returns
    -------
    callable
        The decorated function.

    Examples
    --------
    >>> @diff_op
    ... def my_operator(x):
    ...     return x**2
    ...
    >>> my_operator.is_op
    True
    """
    func.op_method = "D"

    return func


def make_op(func: callable) -> callable:
    """
    Decorates a function as an operator and returns it.

    Parameters:
    -----------
    func: callable
        The function to be decorated as an operator.

    Returns:
    --------
    callable:
        The decorated operator function.

    Notes:
    ------
    This function sets the 'is_op' attribute of the input function to True, indicating
    that it is an operator function.
    """
    func.is_op = True

    return func
