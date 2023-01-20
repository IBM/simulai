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
D = sympy.Function('D')


Sin = sympy.Function('Sin')
Cos = sympy.Function('Cos')
Tanh = sympy.Function('Tanh')
Identity = sympy.Function('Identity')
Kronecker = sympy.Function('Kronecker')

def L(u:sympy.Symbol, vars:tuple) -> callable:

    l = sum([D(D(u, var), var) for var in vars])

    return l

def Div(u:sympy.Symbol, vars:tuple) -> callable:

    l = sum([D(u, var) for var in vars])

    return l

def diff_op(func:callable) -> callable:

    func.op_method = 'D'

    return func

def make_op(func:callable) -> callable:

    func.is_op = True

    return func
