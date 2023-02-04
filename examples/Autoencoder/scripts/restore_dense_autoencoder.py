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

from simulai.file import SPFile
from simulai.metrics import L2Norm


def u(t, x, L: float = None, t_max: float = None) -> np.ndarray:
    return np.sin(4 * np.pi * t * (x / L - 1 / 2) ** 2) * np.cos(
        5 * np.pi * (t / t_max - 1 / 2) ** 2
    )


t_max = 10
L = 5
K = 512
N = 10_000

x = np.linspace(0, L, K)
t = np.linspace(0, t_max, N)
T, X = np.meshgrid(t, x, indexing="ij")

data_test = u(T, X, L=L, t_max=t_max)

saver = SPFile(compact=False)

autoencoder_reload = saver.read(model_path="/tmp/autoencoder_mlp")

approximated_data = autoencoder_reload.eval(input_data=data_test)

l2_norm = L2Norm()

projection_error = 100 * l2_norm(
    data=approximated_data, reference_data=data_test, relative_norm=True
)

print(f"Projection error: {projection_error} %")
