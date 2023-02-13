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


# Geometrical progression
def gp(init, factor, n) -> list:
    """
    Generate a geometric progression of numbers.

    Parameters
    ----------
    init: int or float
        The first number in the progression.
    factor: int or float
        The factor by which each number in the progression is multiplied by to generate the next number.
    n: int
        The number of numbers to generate in the progression.

    Returns
    -------
    list
        A list of `n` numbers in the geometric progression, starting with `init` and multiplying by `factor` for each subsequent number.

    Raises
    ------
    Exception
        If `n` is not higher than 0.
    """
    if n == 1:
        return [init]
    elif n > 1:
        return gp(init, factor, n - 1) + [factor * gp(init, factor, n - 1)[-1]]
    else:
        raise Exception("n must be higher than 0")
