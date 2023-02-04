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

from simulai.rom import QQM

t = np.linspace(0, 2 * np.pi, 10000)

# input: s = [t, sin(t), cos(t)]
s = np.hstack([t[:, None], np.sin(t)[:, None], np.cos(t)[:, None]])

# output: e = [1, sin(2*t), cos(2*t)]
e = np.hstack(
    [np.ones(t.shape)[:, None], np.sin(2 * t)[:, None], np.cos(2 * t)[:, None]]
)

# Trying to find a projection matrix V_bar : e = [s X s] * V_bar
# Ideally: V_bar = [0  0  0]
#                  [0  0  0]
#                  [0  0  0]
#                  [1  0 -1]
#                  [0  2  0]
#                  [1  0  1]

INFO = """
input: s = [t, sin(t), cos(t)]
output: e = [1, sin(2*t), cos(2*t)]

Trying to find a projection matrix V_bar : e = [s X s] * V_bar
Ideally: V_bar = [0  0  0]
                 [0  0  0]
                 [0  0  0]
                 [1  0 -1]
                 [0  2  0]
                 [1  0  1]
"""

print(INFO)

# Using the default solver (SpaRSA)
qqm = QQM(n_inputs=3, lambd=1e-3, alpha_0=100, use_mean=True)

# Using the default solver (SpaRSA) without sparsity hard-limiting
qqm.fit(input_data=s, target_data=e)

print("\n Vanilla SpaRSA")
print(qqm.V_bar)

# Using the default solver (SpaRSA) with sparsity hard-limiting
qqm = QQM(n_inputs=3, lambd=1e-3, sparsity_tol=1e-6, alpha_0=100, use_mean=True)

qqm.fit(input_data=s, target_data=e)

print("\n SpaRSA using sparsity threshold of 1e-6")
print(qqm.V_bar)

# Using the Moore-Penrose generalized inverse
qqm.fit(input_data=s, target_data=e, pinv=True)

print("\n Moore-Penrose pseudoinverse")
print(qqm.V_bar)
