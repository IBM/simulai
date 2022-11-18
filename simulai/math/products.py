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

import numpy as np

def kronecker(a: np.ndarray) -> np.ndarray:

    if len(a.shape) == 2:
        n_inputs = a.shape[1]
    else:
        n_inputs = a.shape[0]

    i_u, j_u = np.triu_indices(n_inputs)

    kron_output = np.einsum('bi, bj->bij', a, a)

    assert np.isnan(kron_output).max() == False, "There are NaN in the Kronecker output"

    # Checking if the Kronecker output tensor is symmetric or not
    if np.array_equal(kron_output, kron_output.transpose((0, 2, 1))):
        return kron_output[:, i_u, j_u]
    else:
        shapes = kron_output.shape[1:]
        return kron_output.reshape(-1, np.prod(shapes))
